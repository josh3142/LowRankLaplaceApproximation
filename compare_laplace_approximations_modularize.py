"""Compute the predictive covariance of different methods,
use them to infer a projection operator and compare the results
with `update_performance_metrics`.
"""

import os 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import os
import math
from typing import Optional, Union, Callable, Tuple

import hydra
from omegaconf import DictConfig, open_dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import laplace
from laplace import FullLaplace, KronLaplace
from tqdm import tqdm

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
from linearized_model.approximation_metrics import (
    update_performance_metrics,
    relative_error,
    trace,
    collect_NLL,
)

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
                shuffle=True
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
            shuffle=True
            )
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
                + cfg.projector.sigma.prior_precision * torch.eye(H.size(0)).to(cfg.device_torch)
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
            shuffle=True
            )
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
    # store all results in this dictionary
    results = {"cfg": cfg}

    # load non-optional arguments
    dtype_str = cfg.dtype
    dtype = getattr(torch, dtype_str)
    torch.set_default_dtype(dtype)
    print(f"Considering {cfg.data.name}")
    get_model_kwargs = dict(cfg.pred_model.param)
    likelihood = "classification" if cfg.data.is_classification else "regression"

    # load optional arguments
    corrupt_data = getattr(cfg, "corrupt_data", False)
    stored_hessians = getattr(cfg, "stored_hessians", [])
    laplace_methods = getattr(cfg, "laplace_methods", [])
    if len(stored_hessians) > 0:
        laplace_methods += ["file"]
    subset_methods = getattr(cfg, "subset_methods", [])
    reference_method = getattr(cfg, "reference_method", None)
    s_max = cfg.projector.s.max
    s_number = cfg.projector.s.n
    s_min = cfg.projector.s.min

    compute_reference_method = getattr(cfg, "compute_reference_method", True)
    store_method_sigma = getattr(cfg, "store_method_sigma", True)
    store_p = getattr(cfg, "store_p", False) # should p_s be stored?

    # Fix reference method
    if reference_method is None:
        print("No reference method specified")
        if len(stored_hessians) > 0:
            print("Choosing first stored hessian")
            reference_method = "file_" + stored_hessians[0]
        else:
            assert len(laplace_methods) > 0
            print("Choosing first laplace method")
            reference_method = laplace_methods[0]
    print(f"Taking {reference_method} as reference method")
    results["reference_method"] = reference_method

    # setting up kwargs for loading of model and data
    get_model_kwargs["name"] = cfg.pred_model.name
    get_model_kwargs |= dict(cfg.data.param)
    results["get_model_kwargs"] = get_model_kwargs
    if not corrupt_data:
        get_dataset_kwargs = dict(
            name=cfg.data.name, path=cfg.data.path, dtype=dtype_str
        )
    else:
        print(f'Using corrupt_data {cfg.data.name_corrupt}')
        get_dataset_kwargs = dict(
            name=cfg.data.name_corrupt, path=cfg.data.path, dtype=dtype_str
        )
    results["get_dataset_kwargs"] = get_dataset_kwargs

    # setting up paths
    results_path = os.path.join(
        "results", cfg.data.name, cfg.pred_model.name, f"seed{cfg.seed}"
    )
    projector_path = os.path.join(results_path, "projector")
    results_name = f"comparison_laplace_approximations{cfg.projector.name_postfix}.pt"
    results_filename = os.path.join(results_path, results_name)
    print(f"Using folder {results_path}")

    def ckpt_file(seed: Union[None, int], ckpt_name: str=cfg.data.model.ckpt) -> str:
        if seed is not None:
            return os.path.join(results_path, f"seed{seed}", "ckpt", ckpt_name)
        else:
            return os.path.join(results_path, "ckpt", ckpt_name)


    # load data
    train_data = get_dataset(**get_dataset_kwargs, train=True)
    test_data = get_dataset(**get_dataset_kwargs, train=False)
    # used for fitting laplacian
    fit_dataloader = DataLoader(
        dataset=train_data,
        batch_size=cfg.projector.fit.batch_size,
        shuffle=True
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
    def create_test_proj_jac_it():
        return create_jacobian_data_iterator(
            dataset=test_data,
            model=model,
            batch_size=cfg.projector.batch_size,
            number_of_batches=cfg.projector.n_batches,
            device=cfg.device_torch,
            dtype=dtype,
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
    make_deterministic(cfg.seed)
    # Collect for each seed results and store them in `results`
    print(f"Using seed {cfg.seed}\n............")

    # load checkpoint for seed
    ckpt_file_name = ckpt_file(seed=None, ckpt_name=cfg.data.model.ckpt)
    results["ckpt_file_name"] = ckpt_file_name
    print(f"Loading model from {ckpt_file_name}")
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
    # collecting posterior covariances for low rank methods
    print(">>>> Collecting posterior covariances")
    results["low_rank"] = {}
    if reference_method == "ggn_it":
        results["low_rank"][reference_method] = {"InvPsi": None}
        IPsi_ref = get_Psi(cfg.projector.sigma.method.psi, cfg, model, train_data, path=projector_path)
    else:
        assert (
            reference_method.startswith("file")
            or reference_method in laplace_methods
        )

    for method_name in laplace_methods:
        if method_name == "file":
            inv_Psi = get_Psi("load_file", cfg, model, train_data, path=projector_path)
            results["low_rank"]["file_" + cfg.projector.posterior_hessian.load.name] = {
                "InvPsi": inv_Psi
            }
        else:
            if method_name in ["full"]:
                inv_Psi = get_Psi("full", cfg, model, train_data, path=projector_path)
            elif method_name in ["kron"]:
                 inv_Psi = get_Psi("kron", cfg, model, train_data, path=projector_path)
            results["low_rank"][method_name] = {"InvPsi": inv_Psi}

    # Collect indices for subset methods
    print(">>>>> Collecting subset methods")
    results["subset"] = {}
    for method in subset_methods:
        print(f"Computing {method}")
        if method == "swag":
            subset_kwargs = dict(cfg.data.swag_kwargs)
        else:
            subset_kwargs = {}
        results["subset"][method] = {
            "Indices": subset_indices(
                model=model,
                likelihood=likelihood,
                train_loader=fit_dataloader,
                method=method,
                **subset_kwargs,
            )
        }

    # collect reference and baseline metrics
    results["baseline"] = {}

    def update_Sigma_metrics(
        metrics_dict: dict,
        Sigma_approx: torch.Tensor,
        Sigma_ref: Optional[torch.Tensor],
    ):
        """Summarizes the metric collection for
        computed Sigma approximation"""
        # trace
        update_performance_metrics(
            metrics_dict=metrics_dict,
            key="trace",
            value=trace(Sigma_approx=Sigma_approx),
        )

        # relative error
        if Sigma_ref is not None:
            update_performance_metrics(
                metrics_dict=metrics_dict,
                key="rel_error",
                value=relative_error(
                    Sigma_approx=Sigma_approx, Sigma=Sigma_ref
                ),
            )

    def update_NLL_metrics(
        metrics_dict: dict,
        predictive: Callable[
            [torch.Tensor,],
            Tuple[torch.Tensor, torch.Tensor]
        ],
        nll_dataloader=nll_dataloader,
        is_classification=cfg.data.is_classification,
    ):

        update_performance_metrics(
            metrics_dict=metrics_dict,
            key="NLL",
            value=collect_NLL(
                predictive=predictive,
                dataloader=nll_dataloader,
                is_classification=is_classification,
                reduction="mean",
                verbose=False,
                device=cfg.device_torch,
            ),
        )

    # do only store reference InvPsi if it isn't
    # build using a V iterator (cf. above)
    if reference_method != "ggn_it":
        IPsi_ref = results["low_rank"][reference_method]["InvPsi"]
        results["baseline"]["InvPsi"] = IPsi_ref
    else:
        results["baseline"]["InvPsi"] = None

    # obtain best optimal approximation using test data
    # and the reference method
    if compute_reference_method:
        print(">>>>> Computing baseline results for reference method")
        Sigma_ref = compute_Sigma(
            IPsi=IPsi_ref, J_X=create_test_proj_jac_it
        )

        P = get_P(
            cfg.reference_method, 
            cfg, 
            model, 
            data_Psi=train_data, 
            data_J=test_data, 
            path=projector_path
        )
        if store_p:
            results["baseline"]["P"] = P
        results["baseline"]["metrics"] = {}
        create_Sigma_P_s_it = compute_Sigma_P(
            P=P,
            IPsi=IPsi_ref,
            J_X=create_test_proj_jac_it,
            s_iterable=s_list,
        )
        predictive = IPsi_predictive(
            model=model,
            IPsi=IPsi_ref,
            P=P,
            chunk_size=cfg.projector.chunk_size,
            regression_likelihood_sigma=regression_likelihood_sigma,
        )
        assert callable(create_Sigma_P_s_it)
        for s, Sigma_P_s in zip(s_list, create_Sigma_P_s_it()):
            update_Sigma_metrics(
                metrics_dict=results["baseline"]["metrics"],
                Sigma_approx=Sigma_P_s,
                Sigma_ref=Sigma_ref,
            )
            update_NLL_metrics(
                metrics_dict=results["baseline"]["metrics"],
                predictive=lambda X: predictive(X=X, s=s)
            )

    else:
        Sigma_ref = None
        # results["baseline"]["metrics"] = None
    results["baseline"]["Sigma_test"] = Sigma_ref

    # collect metrics for low_rank methods
    print(">>>>>> Evaluating low rank methods\n............")
    for method in results["low_rank"].keys():
        if not compute_reference_method and method == reference_method:
            continue
        print(f"> Computing results for {method}")
        results["low_rank"][method]["metrics"] = {}
        if store_method_sigma:
            results["low_rank"][method]["full_cov"] = {"metrics": {}}
        else:
            results["low_rank"][method]["full_cov"] = None
        if method != "ggn_it":
            IPsi = results["low_rank"][method]["InvPsi"]
        elif reference_method == "ggn_it":
            IPsi = IPsi_ref
        else:
            raise NotImplementedError(
                "ggn_it is only available as reference method"
            )
        
        P = get_P(
            method, 
            cfg, 
            model, 
            data_Psi=train_data, 
            data_J=train_data, 
            path=projector_path
        )

        if store_p:
            results["low_rank"][method]["P"] = P
        print("Computing performance of P on test data")
        create_Sigma_P_s_it = compute_Sigma_P(
            P=P,
            IPsi=IPsi_ref,
            J_X=create_test_proj_jac_it,
            s_iterable=s_list,
        )
        predictive = IPsi_predictive(
            model=model,
            IPsi=IPsi,
            P=P,
            chunk_size=cfg.projector.chunk_size,
            regression_likelihood_sigma=regression_likelihood_sigma,
        )
        assert callable(create_Sigma_P_s_it)
        for s, Sigma_P_s in tqdm(
            zip(s_list, create_Sigma_P_s_it()),
            desc="Running through s"
        ):
            update_Sigma_metrics(
                metrics_dict=results["low_rank"][method]["metrics"],
                Sigma_approx=Sigma_P_s,
                Sigma_ref=Sigma_ref,
            )
            update_NLL_metrics(
                metrics_dict=results["low_rank"][method]["metrics"],
                predictive=lambda X: predictive(X=X, s=s),
            )
        if store_method_sigma:
            print(f"Compute full Sigma on test data for method {method}")
            Sigma_test_method = compute_Sigma(
                IPsi=IPsi, J_X=create_test_proj_jac_it
            )
            results["low_rank"][method]["full_cov"]["Sigma_test"] = (
                Sigma_test_method
            )
            update_Sigma_metrics(
                metrics_dict=results["low_rank"][method]["full_cov"][
                    "metrics"
                ],
                Sigma_approx=Sigma_test_method,
                Sigma_ref=Sigma_ref,
            )
            full_predictive = IPsi_predictive(
                model=model,
                IPsi=IPsi,
                P=None,
                chunk_size=cfg.projector.chunk_size,
                regression_likelihood_sigma=regression_likelihood_sigma,
            )
            update_NLL_metrics(
                metrics_dict=results["low_rank"][method]["full_cov"][
                    "metrics"
                ],
                predictive=full_predictive,
            )
        else:
            results["low_rank"][method]["full_cov"]["Sigma_test"] = (
                None
            )

    # collect metrics for subset methods
    print(">>>>>> Evaluating subset methods\n............")
    for method in results["subset"].keys():
        print(f"> Computing results for {method}")
        P = get_P(
            method, 
            cfg, 
            model, 
            data_Psi=train_data, 
            data_J=train_data, 
            path=projector_path
        )
        print("Computing performance of P on test data")
        results["subset"][method]["metrics"] = {}
        create_Sigma_P_s_it = compute_Sigma_P(
            P=P,
            IPsi=IPsi_ref,
            J_X=create_test_proj_jac_it,
            s_iterable=s_list,
        )
        predictive = IPsi_predictive(
            model=model,
            IPsi=IPsi_ref,
            P=P,
            regression_likelihood_sigma=regression_likelihood_sigma,
            chunk_size=cfg.projector.chunk_size,
        )
        assert callable(create_Sigma_P_s_it)
        for s, Sigma_P_s in zip(s_list, create_Sigma_P_s_it()):
            update_Sigma_metrics(
                metrics_dict=results["subset"][method]["metrics"],
                Sigma_approx=Sigma_P_s,
                Sigma_ref=Sigma_ref,
            )
            update_NLL_metrics(
                metrics_dict=results["subset"][method]["metrics"],
                predictive=lambda X: predictive(X=X, s=s),
            )

    # save results after each seed computation
    print(
        f"Done with seed {cfg.seed}! Saving results under {results_filename}"
            )
    with open(results_filename, "wb") as f:
        torch.save(results, f)


if __name__ == "__main__":
    run_main()
