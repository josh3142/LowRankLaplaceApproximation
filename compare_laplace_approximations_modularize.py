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
from torch.utils.data import DataLoader, Dataset
import numpy as np
import laplace
from laplace import FullLaplace, KronLaplace

from typing import Literal

from utils import estimate_regression_likelihood_sigma
from projector.projector1d import (
    create_jacobian_data_iterator,
    number_of_parameters_with_grad,
)
from projector.fisher import get_V_iterator
from data.dataset import get_dataset
from pred_model.model import get_model
from linearized_model.low_rank_laplace import (
    InvPsi,
    FullInvPsi,
    HalfInvPsi,
    KronInvPsi,
    compute_Sigma,
    compute_optimal_P,
    compute_Sigma_P,
    IPsi_predictive,
)
from linearized_model.subset import subset_indices
from linearized_model.approximation_metrics import collect_NLL


from utils import make_deterministic

def get_Psi(
        method: Literal["ggn_it", "load_file" "kron", "full"], 
        cfg: DictConfig, 
        model: nn.Module, 
        data: Dataset, 
        path: str
    ) -> InvPsi:
    """
    Wrapper to get posterior `Psi`.

    Args:
        method: Method to compute the posterior. 
            `ggn_it` computes generalized Gauss-Newton matrix as an iterator 
            `load_file` loads a precomputed posterior
            `kron`, `diagonal` and `full` computes `Psi` with the Laplace lib
        cfg: Configurations file
        model: Pytorch model
        data: Pytorch Dataset
        path: string to point to the file loaded by `load_file`
    """
    dtype = getattr(torch, cfg.dtype)

    if method=="ggn_it":
        def compute_psi_ggn_iterator(cfg, model, data):
            dl = DataLoader(
                dataset=data,
                batch_size=cfg.projector.v.batch_size,
                shuffle=False
                )
            def create_V_it():
                return get_V_iterator(
                    model=model,
                    dl=dl,
                    is_classification=cfg.data.is_classification,
                    n_batches=cfg.projector.v.n_batches,
                    chunk_size=cfg.projector.chunk_size,
                )
            IPsi = HalfInvPsi(
                V=create_V_it,
                prior_precision=cfg.projector.sigma.prior_precision
            )
            return IPsi
        return compute_psi_ggn_iterator(cfg, model, data)
    
    elif method in ["kron", "full"]:
        likelihood = "classification" if cfg.data.is_classification \
            else "regression"
        dl = DataLoader(
            dataset=data,
            batch_size=cfg.projector.v.batch_size,
            shuffle=False
            )
        make_deterministic(cfg.seed)
        la = laplace.Laplace(
                    model=model,
                    hessian_structure=method,
                    likelihood=likelihood,
                    subset_of_weights="all",
                    prior_precision=cfg.projector.sigma.prior_precision,
                )
        la.fit(dl)
        if method=="kron":
            assert type(la) is KronLaplace
            return KronInvPsi(inv_Psi=la)

        elif method=="full":
            assert type(la) is FullLaplace
            return FullInvPsi(inv_Psi=la.posterior_precision.to(dtype))
        
    elif method=="load_file":
        hessian_name = cfg.projector.posterior_hessian.load.name
        hessian_file_name = os.path.join(
            path, cfg.projector.posterior_hessian.load.type, hessian_name
        )
        with open(hessian_file_name, "rb") as f:
            H_file = torch.load(f, map_location=cfg.device_torch)
        
        if hessian_file_name.startswith("Ihalf"):
            V = H_file["H"].to(dtype)
            return  HalfInvPsi(
                    V=V,
                    prior_precision=cfg.projector.sigma.prior_precision,
                )
        else:
            H = H_file["H"].to(dtype)
            assert H.size(0) == H.size(1), "Hessian must be squared matrix."
            inv_Psi = H \
                + cfg.projector.sigma.prior_precision \
                * torch.eye(H.size(0)).to(cfg.device_torch)
            return FullInvPsi(inv_Psi=inv_Psi)
        
    else:
        raise NotImplementedError

def get_P(        
        method: Literal["ggn_it", "load_file" "kron", "full", "swag", "magnitude", 
                        "diagonal", "custom"], 
        cfg: DictConfig, 
        model: nn.Module, 
        data_Psi: Dataset,
        data_J: Dataset, 
        path: str
    ) -> InvPsi:
    """
    Wrapper to get the linear operator `P`.

    Args:
        method: Method to compute `P`. 
            `ggn_it`, `load_file`, `kron` and `full` are methods to compute the 
            posterior to obtain the optimal linear operator
            `swag`, `magnitude`, `diagonal` and `custom` select a certain set
            of weights to get `P` using the Laplace lib
        cfg: Configurations file
        model: Pytorch model
        data_Psi: Pytorch Dataset to compute the posterior (if needed)
        data_Psi: Pytorch Dataset to compute the Jacobians (if needed)
        path: string to point to the file loaded by `load_file`
    """

    if method in ["ggn_it", "load_file", "kron", "full"]:
        def create_proj_jac_it():
            return create_jacobian_data_iterator(
                dataset=data_J,
                model=model,
                batch_size=cfg.projector.batch_size,
                number_of_batches=cfg.projector.n_batches,
                device=cfg.device_torch,
                dtype=getattr(torch, cfg.dtype),
                chunk_size=cfg.projector.chunk_size,
            )
        inv_Psi = get_Psi(method, cfg, model, data_Psi, path)
        U = inv_Psi.Sigma_svd(create_proj_jac_it)[0]
        P = compute_optimal_P(IPsi=inv_Psi, J_X=create_proj_jac_it, U=U)
        return P
    
    elif method in ["diagonal", "magnitude", "swag", "custom"]:
        subset_kwargs = dict(cfg.data.swag_kwargs)
        likelihood = "classification" if cfg.data.is_classification \
            else "regression"
        dl = DataLoader(
            dataset=data_Psi,
            batch_size=cfg.projector.v.batch_size,
            shuffle=False
            )
        make_deterministic(cfg.seed)
        Ind = subset_indices(
                model=model,
                likelihood=likelihood,
                train_loader=dl,
                method=method,
                **subset_kwargs,
            )
        P = Ind.P(cfg.projector.s_max_regularized).to(cfg.device_torch)
        return P

    else:
        raise NotImplementedError


@hydra.main(config_path="config", config_name="config")
def run_main(cfg: DictConfig) -> None:
    make_deterministic(cfg.seed)
    torch.set_default_dtype(getattr(torch, cfg.dtype))

    # store all results in this dictionary
    results = {"cfg": cfg}
    nll = {}

    print(f"Considering {cfg.data.name}")
    get_model_kwargs = dict(cfg.pred_model.param) | dict(cfg.data.param)
    get_model_kwargs["name"] = cfg.pred_model.name
    results["get_model_kwargs"] = get_model_kwargs

    # load optional arguments
    corrupt_data = getattr(cfg, "corrupt_data", False)
    s_max = cfg.projector.s.max
    s_number = cfg.projector.s.n
    s_min = cfg.projector.s.min

    # setting up kwargs for loading of model and data
    if not corrupt_data:
        get_dataset_kwargs = dict(
            name=cfg.data.name, path=cfg.data.path, dtype=cfg.dtype
        )
    else:
        print(f'Using corrupt_data {cfg.data.name_corrupt}')
        get_dataset_kwargs = dict(
            name=cfg.data.name_corrupt, path=cfg.data.path, dtype=cfg.dtype
        )
    results["get_dataset_kwargs"] = get_dataset_kwargs

    # setting up paths
    results_path = os.path.join(
        "results", cfg.data.name, cfg.pred_model.name, f"seed{cfg.seed}"
    )
    projector_path = os.path.join(results_path, "projector")
    results_name = f"SigmaP{cfg.projector.sigma.method.p}" + \
        f"Psi{cfg.projector.sigma.method.psi}{cfg.projector.name_postfix}.pt"
    nll_name = f"nll{cfg.projector.sigma.method.p}" + \
        f"Psi{cfg.projector.sigma.method.psi}{cfg.projector.name_postfix}.pt"
    results_filename = os.path.join(results_path, results_name)
    nll_filename = os.path.join(results_path, nll_name)
    print(f"Using folder {results_path}")

    # load data
    train_data = get_dataset(**get_dataset_kwargs, train=True)
    test_data = get_dataset(**get_dataset_kwargs, train=False)
    # used for fitting laplacian
    fit_dataloader = DataLoader(
        dataset=train_data,
        batch_size=cfg.projector.fit.batch_size,
        shuffle=False
    )
    # used for computation of NLL metric
    nll_dataloader = DataLoader(
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
        if type(module) in cfg.projector.layers_to_ignore:
            for par in module.parameters():
                par.requires_grad = False

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
    if s_max is None:
        number_of_parameters = number_of_parameters_with_grad(model)
        test_out = model(next(iter(fit_dataloader))[0].to(cfg.device_torch))
        if len(test_out.shape) == 1:
            n_out = 1
        else:
            n_out = test_out.size(-1)
        n_data = min(
            len(train_data), cfg.projector.n_batches * cfg.projector.batch_size
        )
        s_max = min(n_data * n_out, number_of_parameters)
    s_step = math.ceil((s_max-s_min) / (s_number-1))
    s_list = np.concatenate((
        np.arange(s_min, s_max, step=s_step),
        np.array([s_max]),
    ))
    with open_dict(cfg):
        cfg.projector.s_max_regularized = s_max

    results["s_list"] = s_list

    # load checkpoint
    ckpt_file_name = os.path.join(results_path, "ckpt", cfg.data.model.ckpt)
    results["ckpt_file_name"] = ckpt_file_name
    with open(ckpt_file_name, "rb") as f:
        state_dict = torch.load(f, map_location=cfg.device_torch)

    model.load_state_dict(state_dict=state_dict)
    # for regression problems estimate the sigma of the likelihood
    if not cfg.data.is_classification:
        print('Estimating sigma of likelihood')
        regression_likelihood_sigma = estimate_regression_likelihood_sigma(
            model=model,
            dataloader=fit_dataloader,
            device=cfg.device_torch,
        )
        results['regression_likelihood_sigma'] \
            = regression_likelihood_sigma
    else:
        regression_likelihood_sigma = None

    # TODO: Check if IPsi_ggn can be deleted.
    # IPsi_ggn = get_Psi("ggn_it", cfg, model, train_data, path=projector_path)
    IPsi = get_Psi(
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
            path=projector_path
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
    results["s_list"] = s_list
    nll["s_list"] = s_list
    for s, Sigma_P_s in zip(s_list, create_Sigma_P_s_it):
        # store Sigma_P
        name_Sigma = f"SigmaP{s}" if s is not None else "SigmaP"
        results[name_Sigma] = Sigma_P_s

        # compute nll
        predictive_s = lambda X: predictive(X=X, s=s)
        nll_value = collect_NLL(
            predictive=predictive_s,
            dataloader=nll_dataloader,
            is_classification=cfg.data.is_classification,
            reduction="mean",
            verbose=False,
            device=cfg.device_torch
        )
        name_nll = f"nll{s}" if s is not None else "nll"
        nll[name_nll] = nll_value

    # save results after each seed computation
    print(f"Seed {cfg.seed}! Save results in {results_filename}")
    with open(results_filename, "wb") as f:
        torch.save(results, f)
    with open(nll_filename, "wb") as f:
        torch.save(nll, f)


if __name__ == "__main__":
    run_main()
