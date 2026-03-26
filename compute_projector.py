""" Script to compute a projection matrix. """
import os
import json
from pathlib import Path
import hydra
from omegaconf import DictConfig, open_dict
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from projector.projector1d import create_jacobian_data_iterator
from data.dataset import get_dataset
from pred_model.model import get_model
from projector.projector import get_P

from utils import make_deterministic, estimate_regression_likelihood_sigma  

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def get_rank_jacobian(model: nn.Module, data: Tensor, cfg: DictConfig) -> int:
    """ Compute the rank of the Jacobian."""
    def create_train_jac_it():
        return create_jacobian_data_iterator(
            dataset=data,
            model=model,
            batch_size=cfg.projector.batch_size,
            number_of_batches=cfg.projector.n_batches,
            device=cfg.device_torch,
            dtype=getattr(torch, cfg.dtype),
            jacobian_order_seed=cfg.projector.jacobian_seed,
            chunk_size=cfg.projector.chunk_size,
        )
    J_X = torch.concat([j for j in create_train_jac_it()], dim=0)
    return torch.linalg.matrix_rank(J_X).item()


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
    Path(projector_path).mkdir(parents=True, exist_ok=True)
    name = f"{cfg.projector.sigma.method.p}_" + \
        f"Psi{cfg.projector.sigma.method.psi}{cfg.projector.name_postfix}"
    results_name = f"projector_{name}.npz"

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
    jacobian_data = train_data if not "lowrankoptimal" in cfg.projector.sigma.method.p \
            else test_data # theoretical optimal solution needs test_data

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

    if cfg.projector.s.max is None:
        s_max_jacobian = get_rank_jacobian(
                model, jacobian_data, cfg
            )
        s_max_regularized = s_max_jacobian
    else:
        # s_max_regularized = min(cfg.projector.s.max, s_max_jacobian)
        s_max_regularized = cfg.projector.s.max
    with open_dict(cfg):
        cfg.projector.s_max_regularized = s_max_regularized

    if cfg.projector.data_std is None:
        if not cfg.data.is_classification:
            # for regression problems estimate the sigma of the likelihood
            dl_train = DataLoader(
                dataset=train_data,
                batch_size=cfg.projector.fit.batch_size,
                shuffle=False
            )
            print('Estimate data std')
            regression_likelihood_sigma = estimate_regression_likelihood_sigma(
                    model=model,
                    dataloader=dl_train,
                    device=cfg.device_torch,
                    )
        else:
            # choose None for classification
            regression_likelihood_sigma = None
        
    elif cfg.projector.data_std > 0:
        regression_likelihood_sigma = cfg.projector.data_std
    else:
        raise ValueError("Insert a None or a positive number for `data_std`.")

    # get projector
    print('Estimate P')
    P = get_P(
            cfg.projector.sigma.method.p,
            cfg,
            model,
            data_Psi=train_data,
            data_J=train_data if not "lowrankoptimal" in cfg.projector.sigma.method.p \
                else test_data, # theoretical optimal solution needs test_data
            path=projector_path,
            s=s_max_regularized,
            data_std=regression_likelihood_sigma,
        )

    # store data
    torch.save(
        {"P": P.detach().cpu()},
        os.path.join(projector_path, results_name)
    )
    
    # store configs
    with open(
        os.path.join(projector_path, "Psize.json"), 
        "w", 
        encoding="utf-8"
    ) as f:
        json.dump({"P_shape": P.shape}, f, indent=4)


if __name__ == "__main__":
    run_main()
