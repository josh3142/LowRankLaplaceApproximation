"""Prints out some basic information on the specified dataset.
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

from utils import make_deterministic


def get_s_max(
        model: nn.Module,
        dl: DataLoader,
        n_batches: int,
        batch_size: int
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
    compute_rank = getattr(cfg, 'compute_rank', True)
    make_deterministic(cfg.seed)
    torch.set_default_dtype(getattr(torch, cfg.dtype))

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
    def create_train_jac_it():
        return create_jacobian_data_iterator(
            dataset=train_data,
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

    # for regression problems estimate the sigma of the likelihood
    regression_likelihood_sigma = get_regression_likelihood_sigma(
        model, dl_train, cfg.data.is_classification, cfg.device_torch
    )
    # p
    number_of_parameters = number_of_parameters_with_grad(model)
    # rank J_X
    if compute_rank:
        print('Computing rank(J_X)')
        J_X = torch.concat([j for j in create_train_jac_it()], dim=0)
    print('Summary\n .......')

    print(f'len(train_data): {len(train_data)}')
    print(f'len(test_data): {len(test_data)}')
    print(f'len(train_data)+len(test_data): {len(train_data) + len(test_data)}')
    print(f'sigma (for regression): {regression_likelihood_sigma}')
    print(f'number of parameters: {number_of_parameters}')
    print(f's_max: {s_max_regularized}')
    if compute_rank:
        print(f'rank(J_X): {torch.linalg.matrix_rank(J_X)}')


if __name__ == "__main__":
    run_main()
