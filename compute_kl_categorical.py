""" Script to compute the KL divergence for categorical distribution. """
import os
import math
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from torch import Tensor
from typing import List
from tqdm import tqdm

from data.dataset import get_dataset
from pred_model.model import get_model
from projector.projector import get_IPsi 
from linearized_model.low_rank_laplace import IPsi_predictive
from linearized_model.approximation_metrics import get_kl_categorical

from utils import make_deterministic

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def get_s_list(s_min: int, s_max: int, s_n: int) -> List[int]:
    """ 
    Compute a list of all dimensions of the opterator `P`.
    
    `s_max` and `s_min` are included.
    """
        
    s_step = math.ceil((s_max-s_min) / (s_n-1))
    s_list = np.concatenate((
        np.arange(s_min, s_max, step=s_step),
        np.array([s_max]),
    ))
    return s_list.tolist()


def get_kappa(variances: Tensor) -> Tensor:
    return 1 / torch.sqrt(
            1 + math.pi/8 * torch.diagonal(variances, dim1=1, dim2=2))


@hydra.main(config_path="config", config_name="config")
def run_main(cfg: DictConfig) -> None:
    make_deterministic(cfg.seed)
    torch.set_default_dtype(getattr(torch, cfg.dtype))

    # setting up paths
    print(f"Considering {cfg.data.name}")
    results_path = os.path.join(
        "results", cfg.data.folder_name, cfg.pred_model.name, f"seed{cfg.seed}"
    )
    projector_path = os.path.join(results_path, "projector")
    if cfg.projector.sigma.method.p is not None:
        Path(projector_path).mkdir(parents=True, exist_ok=True)
        print(f"Using projector {cfg.projector.sigma.method.p}")
        name = f"{cfg.projector.sigma.method.p}_" + \
            f"Psi{cfg.projector.sigma.method.psi}{cfg.projector.name_postfix}"
        projector_file = os.path.join(projector_path, f"projector_{name}.npz")
        P = torch.load(
            projector_file, 
            map_location=cfg.device_torch, 
            weights_only=False
        )["P"]
        s_list = get_s_list(
            s_min=cfg.projector.s.min,
            s_max=cfg.projector.s.max,
            s_n=cfg.projector.s.n
    )
    else:
        print("No projector is chosen.")
        P = None
        s_list = [None,]
        

    get_model_kwargs = dict(cfg.pred_model.param) | dict(cfg.data.param)
    get_model_kwargs["name"] = cfg.pred_model.name

    # setting up kwargs for loading of model and data
    data_name = cfg.data.name_corrupt if cfg.data.use_corrupt else cfg.data.name
    get_dataset_kwargs = dict(
        name=data_name, path=cfg.data.path, dtype=cfg.dtype
        )
    print(f'Using data {data_name}')

    # load data and construct DataLoader
    train_data = get_dataset(**get_dataset_kwargs, train=True)
    test_data = get_dataset(**get_dataset_kwargs, train=False)
    dl_test = DataLoader(
        dataset=test_data,
        batch_size=cfg.projector.batch_size,
        shuffle=False
    )

    # load network
    model = get_model(**get_model_kwargs)
    model.eval()
    model.to(cfg.device_torch)
    # switch off layers to ignore
    for module in model.modules():
        if type(module).__name__ in cfg.projector.layers_to_ignore:
            for par in module.parameters():
                par.requires_grad = False

    # load checkpoint
    ckpt_file_name = os.path.join(results_path, "ckpt", cfg.data.model.ckpt)
    with open(ckpt_file_name, "rb") as f:
        state_dict = torch.load(f, map_location=cfg.device_torch,
                                weights_only=True)
    model.load_state_dict(state_dict=state_dict)



    IPsi = get_IPsi(
        method=cfg.projector.sigma.method.psi,
        cfg=cfg,
        model=model,
        data=train_data,
        path=projector_path
    )

    predictive_full = IPsi_predictive(
        model=model,
        IPsi=IPsi,
        P=None,
        chunk_size=cfg.projector.chunk_size,
        regression_likelihood_sigma=None, # script works only for categorical distr
    )
    predictive = IPsi_predictive(
        model=model,
        IPsi=IPsi,
        P=P,
        chunk_size=cfg.projector.chunk_size,
        regression_likelihood_sigma=None,
    )

    # compute kl-divergence
    kl, kl_list ={}, []
    with torch.no_grad():
        for s in s_list:
            n_data, kl[s] = 0, 0.0
            predictive_s = lambda X: predictive(X=X, s=s)
            predictive_full_s = lambda X: predictive_full(X=X, s=None)
            for X, Y in tqdm(dl_test, desc='KL divergence'):
                X, Y = X.to(cfg.device_torch), Y.to(cfg.device_torch)
                predictions_full, variances_full = predictive_full_s(X)
                kappa_full = get_kappa(variances_full)
                log_probs_full = torch.nn.LogSoftmax(dim=-1)(kappa_full * predictions_full)
                predictions, variances = predictive_s(X)
                kappa = get_kappa(variances)
                log_probs = torch.nn.LogSoftmax(dim=-1)(kappa * predictions)
                kl[s] += get_kl_categorical(log_probs_full, log_probs).sum().item()
                n_data += X.size(0)
            # take mean of all samples
            kl[s] /= n_data
            kl_list.append(kl[s])
            print("Dimension: ", s)
            print(f"KL divergence: {kl[s]}")
        results_file_name = f"kl_{cfg.projector.sigma.method.p}" \
            f"_Psi{cfg.projector.sigma.method.psi}" \
            f"{cfg.results_file.name_postfix}.pt"
        results = {'s_list': s_list, 'kl': kl_list}
        torch.save(
            results,
            os.path.join(results_path, results_file_name)
        )
            
if __name__ == "__main__":
    run_main()