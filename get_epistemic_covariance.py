"""Compute the predictive covariance of different methods,
use them to infer a projection operator and compute the epistemic covariances.
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import os
import math

import hydra
from omegaconf import DictConfig, open_dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from pathlib import Path
from typing import List

from utils import estimate_regression_likelihood_sigma
from projector.projector1d import (
    create_jacobian_data_iterator,
    number_of_parameters_with_grad,
)

from data.dataset import get_dataset
from pred_model.model import get_model
from projector.projector import get_P, get_IPsi 
from linearized_model.low_rank_laplace import (
    compute_Sigma,
    compute_Sigma_P,
)

from utils import make_deterministic


def get_s_max(model: nn.Module, dl: DataLoader, n_batches: int, batch_size: int
    ) -> int:
    """ 
    Computes the maximal dimension of the operator `P` possible.
    
    The dimension of `P` is the minimum of the number of differentable 
    parameters of the model or the number of gradients.
    """
    n_parameters = number_of_parameters_with_grad(model)
    device = next(iter(model.parameters())).get_device()

    test_out = model(next(iter(dl))[0].to(device))
    if len(test_out.shape) == 1:
        n_out = 1
    else:
        n_out = test_out.size(-1)
    n_data = min(
        len(dl.dataset), n_batches * batch_size
    )
    return min(n_data * n_out, n_parameters)


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

def get_regression_likelihood_sigma(model, dl, classification, device):
    """ Compute the sigma for non-classification problems. """
    if not classification:
        print('Estimating sigma of likelihood')
        regression_likelihood_sigma = estimate_regression_likelihood_sigma(
            model=model, dataloader=dl, device=device,
        )
    else:
        regression_likelihood_sigma = None
    return regression_likelihood_sigma

@hydra.main(config_path="config", config_name="config")
def run_main(cfg: DictConfig) -> None:
    make_deterministic(cfg.seed)
    torch.set_default_dtype(getattr(torch, cfg.dtype))

    # store all results in this dictionary
    results = {"cfg": cfg}

    # setting up paths
    print(f"Considering {cfg.data.name}")

    results_path = os.path.join(
        "results", cfg.data.folder_name, cfg.pred_model.name, f"seed{cfg.seed}"
    )
    ## for the projector
    projector_path = os.path.join(results_path, "projector")
    projector_filename = f"projector_{cfg.projector.sigma.method.p}" + \
        f"_Psi{cfg.projector.sigma.method.psi}.npz"
    projector_file = os.path.join(projector_path, projector_filename)

    ## for the results
    results_filename = f"SigmaP_{cfg.projector.sigma.method.p}" + \
        f"_Psi{cfg.projector.sigma.method.psi}{cfg.results_file.name_postfix}.pt"
    results_file = os.path.join(results_path, results_filename)
    Path(os.path.join(results_path, "ckpt")).mkdir(parents=True, exist_ok=True)

    
    # setting up kwargs for loading of model and data
    get_model_kwargs = dict(cfg.pred_model.param) | dict(cfg.data.param)
    get_model_kwargs["name"] = cfg.pred_model.name
    results["get_model_kwargs"] = get_model_kwargs

    data_name = cfg.data.name_corrupt if cfg.data.use_corrupt else cfg.data.name
    get_dataset_kwargs = dict(
        name=data_name, path=cfg.data.path, dtype=cfg.dtype
        )
    print(f'Using data {data_name}')
    results["get_dataset_kwargs"] = get_dataset_kwargs

    # load data and construct DataLoader
    train_data = get_dataset(**get_dataset_kwargs, train=True)
    test_data = get_dataset(**get_dataset_kwargs, train=False)
    dl_train = DataLoader(
        dataset=train_data,
        batch_size=cfg.projector.fit.batch_size,
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
    results["ckpt_file_name"] = ckpt_file_name
    with open(ckpt_file_name, "rb") as f:
        state_dict = torch.load(f, map_location=cfg.device_torch, 
                                weights_only=True)
    model.load_state_dict(state_dict=state_dict)

    #  The following objects create upon call an iterator over the jacobian
    def create_proj_jac_it():
        return create_jacobian_data_iterator(
            dataset=test_data,
            model=model,
            batch_size=cfg.projector.batch_size,
            number_of_batches=cfg.projector.n_batches,
            device=cfg.device_torch,
            dtype=getattr(torch, cfg.dtype),
            jacobian_order_seed=cfg.projector.jacobian_seed,
            chunk_size=cfg.projector.chunk_size,
        )

    # Compute s_max and s_List   
    if cfg.projector.s.max is None:
        s_max_regularized = get_s_max(
            model, dl_train, cfg.projector.n_batches, cfg.projector.batch_size
        )
    else:
        s_max_regularized = cfg.projector.s.max
    with open_dict(cfg):
        cfg.projector.s_max_regularized = s_max_regularized
    s_list = get_s_list(
        s_min = cfg.projector.s.min,
        s_max = cfg.projector.s_max_regularized,
        s_n = cfg.projector.s.n
    )
    results["s_list"] = s_list

    if cfg.projector.data_std is None:
    # for regression problems estimate the sigma of the likelihood
        regression_likelihood_sigma = get_regression_likelihood_sigma(
            model, dl_train, cfg.data.is_classification, cfg.device_torch
        )
    elif cfg.projector.data_std > 0:
        regression_likelihood_sigma = cfg.projector.data_std
    else:
        raise ValueError("Insert a None or a positive number for `data_std`.")
    results['regression_likelihood_sigma'] = regression_likelihood_sigma

    print(f'Using prior precision {cfg.projector.sigma.prior_precision}')

    IPsi = get_IPsi(
        method=cfg.projector.sigma.method.psi,
        cfg=cfg,
        model=model,
        data=train_data,
        path=projector_path,
        data_std=regression_likelihood_sigma,
    )

    if cfg.projector.sigma.method.p is None:
        P = None
        Sigma = compute_Sigma(IPsi=IPsi, J_X=create_proj_jac_it)
        create_Sigma_P_s_it = iter([Sigma])
        s_list = [None]
        print("No projector is chosen.")
    else:
        try: 
            P = torch.load(
                projector_file,
                map_location=cfg.device_torch,
                weights_only=False,
            )["P"]
            print(f'Using P from {projector_file}')
        except FileNotFoundError:
            print('No stored P, computing P...')
            P = get_P(
                cfg.projector.sigma.method.p, 
                cfg, 
                model, 
                data_Psi=train_data, 
                data_J=train_data if "lowrankoptimal" not in cfg.projector.sigma.method.p \
                    else test_data, # theoretical optimal solution needs test_data 
                path=projector_path,
                s=s_max_regularized,
                data_std=regression_likelihood_sigma,
            )
        create_Sigma_P_s_it = compute_Sigma_P(
            P=P,
            IPsi=IPsi,
            J_X=create_proj_jac_it,
            s_iterable=s_list,
        )()

    if cfg.projector.store:
        results["P"] = P

    results["s_list"] = s_list
    
    # store Sigma_P_s for each s
    for s, Sigma_P_s in zip(s_list, create_Sigma_P_s_it):
        name_Sigma = f"SigmaP{s}" if s is not None else "SigmaP"
        results[name_Sigma] = Sigma_P_s

    # save results after each seed computation
    print(f"Seed {cfg.seed}! Save results in {results_file}")
    with open(results_file, "wb") as f:
        torch.save(results, f)


if __name__ == "__main__":
    run_main()
