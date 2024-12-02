"""Compute the predictive covariance of different methods,
use them to infer a projection operator and compare the results
with `update_performance_metrics`.
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
    IPsi_predictive,
)
from linearized_model.approximation_metrics import (
    collect_NLL, 
    update_performance_metrics
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
    results, nll = {"cfg": cfg}, {}

    # setting up paths
    print(f"Considering {cfg.data.name}")
    results_path = os.path.join(
        "results", cfg.data.name, cfg.pred_model.name, f"seed{cfg.seed}"
    )
    projector_path = os.path.join(results_path, "projector")
    results_name = f"SigmaP_{cfg.projector.sigma.method.p}" + \
        f"_Psi{cfg.projector.sigma.method.psi}{cfg.projector.name_postfix}.pt"
    nll_name = f"nll_{cfg.projector.sigma.method.p}" + \
        f"_Psi{cfg.projector.sigma.method.psi}{cfg.projector.name_postfix}.pt"
    results_filename = os.path.join(results_path, results_name)
    nll_filename = os.path.join(results_path, nll_name)

    get_model_kwargs = dict(cfg.pred_model.param) | dict(cfg.data.param)
    get_model_kwargs["name"] = cfg.pred_model.name
    results["get_model_kwargs"] = get_model_kwargs

    # setting up kwargs for loading of model and data
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
    results["ckpt_file_name"] = ckpt_file_name
    with open(ckpt_file_name, "rb") as f:
        state_dict = torch.load(f, map_location=cfg.device_torch)
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

    # for regression problems estimate the sigma of the likelihood
    regression_likelihood_sigma = get_regression_likelihood_sigma(
        model, dl_train, cfg.data.is_classification, cfg.device_torch
    )
    results['regression_likelihood_sigma'] = regression_likelihood_sigma

    # TODO: Check if IPsi_ggn can be deleted.
    # IPsi_ggn = get_Psi("ggnit", cfg, model, train_data, path=projector_path)
    IPsi = get_IPsi(
        method=cfg.projector.sigma.method.psi,
        cfg=cfg,
        model=model,
        data=train_data,
        path=projector_path
    )

    if cfg.projector.sigma.method.p is None:
        P = None
        # Sigma = compute_Sigma(IPsi=IPsi_ggn, J_X=create_proj_jac_it)
        Sigma = compute_Sigma(IPsi=IPsi, J_X=create_proj_jac_it)
        create_Sigma_P_s_it = iter([Sigma])
        s_list = [None]
        print("No projector is chosen.")
    else:
        P = get_P(
            cfg.projector.sigma.method.p, 
            cfg, 
            model, 
            data_Psi=train_data, 
            data_J=train_data, 
            path=projector_path,
            s=s_max_regularized,
        )
        create_Sigma_P_s_it = compute_Sigma_P(
            P=P,
            IPsi=IPsi, #IPsi_ggn,
            J_X=create_proj_jac_it,
            s_iterable=s_list,
        )()

    if cfg.projector.store:
        results["P"] = P

    # TODO: Should is this IPsi supposed to be different from IPsi in
    # create_Sigma_P_s_it?
    predictive = IPsi_predictive(
        model=model,
        IPsi=IPsi,
        P=P,
        chunk_size=cfg.projector.chunk_size,
        regression_likelihood_sigma=regression_likelihood_sigma,
    )
    results["s_list"], nll["s_list"], nll["nll"] = s_list, s_list, []
    for s, Sigma_P_s in zip(s_list, create_Sigma_P_s_it):
        # store Sigma_P
        name_Sigma = f"SigmaP{s}" if s is not None else "SigmaP"
        results[name_Sigma] = Sigma_P_s

        # compute nll
        predictive_s = lambda X: predictive(X=X, s=s)
        nll_value = collect_NLL(
            predictive=predictive_s,
            dataloader=dl_test,
            is_classification=cfg.data.is_classification,
            reduction="mean",
            verbose=False,
            device=cfg.device_torch
        ).item()
        update_performance_metrics(nll, "nll", nll_value)

    # save results after each seed computation
    print(f"Seed {cfg.seed}! Save results in {results_filename}")
    with open(results_filename, "wb") as f:
        torch.save(results, f)
    with open(nll_filename, "wb") as f:
        torch.save(nll, f)


if __name__ == "__main__":
    run_main()
