""" Creates plots of the jacobian
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import os
import math

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
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


@hydra.main(config_path="config", config_name="config")
def run_main(cfg: DictConfig) -> None:
    make_deterministic(cfg.seed)
    torch.set_default_dtype(getattr(torch, cfg.dtype))
    use_train = getattr(cfg, 'use_train', True)
    threshold = getattr(cfg, 'threshol', 0.0)
    

    # setting up paths
    print(f"Considering {cfg.data.name}")
    results_path = os.path.join(
        "results", cfg.data.folder_name, cfg.pred_model.name, f"seed{cfg.seed}"
    )

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

    print('Computing jacobian...')
    J_X = torch.concat([j for j in create_jac_it()], dim=0)
    print('done')
    plt.figure(1)
    plt.matshow(J_X.abs().cpu().numpy())
    figure_name = os.path.join(results_path, 'J_X.png') 
    plt.xlabel('parameters')
    plt.ylabel('data')
    print(f'Saving J_X under {figure_name}')
    plt.savefig(figure_name)

    print('sorting Jacobian')
    J_X_summary = J_X.abs().mean(dim=0)
    sort_idx = torch.argsort(J_X_summary, descending=True)
    sorted_J_X = J_X[:,sort_idx]
    assert sorted_J_X.shape == J_X.shape
    figure_name = os.path.join(results_path, 'sorted_J_X.png') 

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    axes[0].imshow(J_X[:,sort_idx].abs().cpu().numpy(), aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0].set_title('Matrix J_X Heatmap')
    axes[0].set_ylabel('data')
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)  # Hide x-ticks

    axes[1].plot(range(len(J_X_summary)), J_X_summary[sort_idx].cpu().numpy(), color='red', marker='o', label='Summary Vector')
    axes[1].set_xlabel('parameters (sorted)')
    axes[1].set_ylabel('averaged gradient')

    # Set log scale on the x-axis for both plots
    axes[0].set_xscale('log')
    axes[1].set_xscale('log')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)  # Minimize space between subplots
    plt.show()
    print(f'Saving sorted J_X under {figure_name}')
    plt.savefig(figure_name)     

    print('finding dead weights')
    dead_gradient = (J_X.abs() <= threshold)
    dead_weights = torch.all(dead_gradient, dim=0)
    number_of_weights = number_of_parameters_with_grad(model)
    num_active_weights = number_of_weights - sum(dead_weights).item()
    print(f'Dead weights: {sum(dead_weights).item()}/{number_of_weights}, \
          that is {sum(dead_weights).item()/number_of_weights*100}%')
    print(f'Active weights: {num_active_weights}/{number_of_weights}, \
          that is {num_active_weights/number_of_weights*100}%')




if __name__ == "__main__":
    run_main()
