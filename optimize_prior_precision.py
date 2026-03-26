"""Find the optimal prior precision for a given model and dataset.
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import os
from typing import Optional

import numpy as np
import torch.nn as nn
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import laplace

from utils import estimate_regression_likelihood_sigma
from data.dataset import get_dataset
from pred_model.model import get_model

from utils import make_deterministic

seed_list = list(range(1, 6))

def get_regression_likelihood_sigma(
        model: nn.Module,
        dl: DataLoader,
        classification: bool,
        device: torch.device) -> Optional[float]:
    """ Compute the sigma for non-classification problems. """
    if not classification:
        print('Estimating sigma of likelihood')
        regression_likelihood_sigma = estimate_regression_likelihood_sigma(
            model=model, dataloader=dl, device=device,
        )
    else:
        regression_likelihood_sigma = 1.0
    return regression_likelihood_sigma

@hydra.main(config_path="config", config_name="config")
def run_main(cfg: DictConfig) -> None:
    prior_precision_collection = []
    print(f'Using seed list {seed_list}')
    for seed in seed_list:
        make_deterministic(seed)
        torch.set_default_dtype(getattr(torch, cfg.dtype))
        device = torch.device(cfg.device_torch)


        # setting up paths
        get_model_kwargs = dict(cfg.pred_model.param) | dict(cfg.data.param)
        get_model_kwargs["name"] = cfg.pred_model.name

        # setting up kwargs for loading of model and data
        data_name = cfg.data.name_corrupt if cfg.data.use_corrupt else cfg.data.name
        get_dataset_kwargs = dict(
            name=data_name, path=cfg.data.path, dtype=cfg.dtype
            )
        print(f'Using data {data_name}')
        results_path = os.path.join(
            "results", cfg.data.folder_name, cfg.pred_model.name, f"seed{seed}"
        ) # for loading the checkpoint

        # load data and construct DataLoader
        train_data = get_dataset(**get_dataset_kwargs, train=True)
        test_data = get_dataset(**get_dataset_kwargs, train=False)
        dl_train = DataLoader(
            dataset=train_data,
            batch_size=cfg.projector.v.batch_size,
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
        model.to(device)

        data_std = get_regression_likelihood_sigma(
            model=model,
            dl=dl_train,
            classification=cfg.data.is_classification,
            device=device,
        ) 

        # fit prior precision
        la = laplace.Laplace(
            model,
            hessian_structure="kron",
            likelihood='classification' if cfg.data.is_classification else 'regression',
            subset_of_weights="all",
            prior_precision=cfg.projector.sigma.prior_precision,
            sigma_noise=data_std,
        )
        print('Fitting Laplace approximation...')
        la.fit(dl_train)
        prior_precision_method = 'marglik'
        print(f'Optimizing prior precision via {prior_precision_method}..')
        la.optimize_prior_precision(method=prior_precision_method)
        print({la.prior_precision})
        prior_precision = la.prior_precision.item()
        prior_precision_collection.append(prior_precision)
        print(f'Found prior precision {prior_precision} for seed {seed}')
    prior_precision_mean = np.mean(prior_precision_collection)
    prior_precision_error = np.std(prior_precision_collection) \
        / np.sqrt(len(prior_precision_collection))
    print(f'Average prior precision:'
          f' {prior_precision_mean} ({prior_precision_error})')


if __name__ == "__main__":
    run_main()