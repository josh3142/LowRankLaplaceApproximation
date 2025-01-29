""" Creates plots of the jacobian
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import os
import math


import hydra
from tqdm import tqdm
from omegaconf import DictConfig
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


from utils import estimate_regression_likelihood_sigma
from projector.projector1d import (
    create_jacobian_data_iterator,
    number_of_parameters_with_grad,
)
from data.dataset import get_dataset
from pred_model.model import get_model
import matplotlib.pyplot as plt

from utils import make_deterministic


seed_list = [1,2,3,4,5]

@hydra.main(config_path="config", config_name="config")
def run_main(cfg: DictConfig) -> None:
    torch.set_default_dtype(getattr(torch, cfg.dtype))
    use_train = getattr(cfg, 'use_train', True)
    threshold = getattr(cfg, 'threshold', 0.0)
     # setting up kwargs for loading of model and data
    data_name = cfg.data.name_corrupt if cfg.data.use_corrupt else cfg.data.name
    get_dataset_kwargs = dict(
        name=data_name, path=cfg.data.path, dtype=cfg.dtype
        )

    # load data and construct DataLoader
    train_data = get_dataset(**get_dataset_kwargs, train=True)
    test_data = get_dataset(**get_dataset_kwargs, train=False)

    dead_weight_percentage_collection = []
    for seed in tqdm(seed_list):
        make_deterministic(seed)
        # setting up paths
        results_path = os.path.join(
            "results", cfg.data.folder_name, cfg.pred_model.name, f"seed{seed}"
        )

        get_model_kwargs = dict(cfg.pred_model.param) | dict(cfg.data.param)
        get_model_kwargs["name"] = cfg.pred_model.name

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
        with open(ckpt_file_name, "rb") as f:
            state_dict = torch.load(f, map_location=cfg.device_torch)
        model.load_state_dict(state_dict=state_dict)

        #  The following objects create upon call an iterator over the jacobian
        def create_jac_it():
            return create_jacobian_data_iterator(
                dataset=train_data if use_train else test_data,
                model=model,
                batch_size=cfg.projector.batch_size,
                number_of_batches=cfg.projector.n_batches,
                device=cfg.device_torch,
                dtype=getattr(torch, cfg.dtype),
                chunk_size=cfg.projector.chunk_size,
            )

        J_X = torch.concat([j for j in create_jac_it()], dim=0)


        dead_gradient = (J_X.abs() <= threshold)
        dead_weights = torch.all(dead_gradient, dim=0)
        number_of_weights = number_of_parameters_with_grad(model)
        dead_weights_percentage = sum(dead_weights).item()/number_of_weights*100
        dead_weight_percentage_collection.append(dead_weights_percentage)
    print(f'On average {np.mean(dead_weight_percentage_collection)} ({np.std(dead_weight_percentage_collection)}) % dead weights')




if __name__ == "__main__":
    run_main()