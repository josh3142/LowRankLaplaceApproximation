"""Compute the predictive covariance of different methods,
use them to infer a projection operator and compute the epistemic covariances.
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import os

import hydra
from omegaconf import DictConfig, open_dict
import torch
from torch.utils.data import DataLoader
from typing import Optional

from data.dataset import get_dataset
from pred_model.model import get_model
from projector.projector import get_P, get_IPsi 
from linearized_model.low_rank_laplace import (
    IPsi_predictive,
)
from linearized_model.approximation_metrics import (
    update_performance_metrics,
    NLL,
    get_brier,
    probit_approximation,
    ECE,
    get_coverage,
    get_calibration_error,
)

from utils import make_deterministic

def update_predictive_metrics(
    metrics_dict: dict,
    cfg: DictConfig,
    predictive: callable,
    dataloader: DataLoader,
):
    is_classification = cfg.data.is_classification
    nll_collection = []
    probs_collection = []
    target_collection = []
    prediction_collection = []
    variance_collection = []
    for x,y in dataloader: 
        device = cfg.device_torch
        if device is not None:
            x, y= x.to(device), y.to(device)
        predictions, variances = predictive(x)
        if is_classification:
            probs_collection.append(probit_approximation(
                predictions=predictions,
                variances=variances,
                log=False
            ))
        target_collection.append(y)
        nll_collection.append(
            NLL(
                predictions=predictions,
                variances=variances,
                targets=y,
                is_classification=is_classification,
                sum=False
            )
        )
        if not is_classification:            
            # 95% Coverage
            # coverage_collection.append(get_coverage(
            #         predictions=predictions_collection,
            #         variances=variances_collection,
            #         targets=target_collection,
            #         regression_sigma=regression_likelihood_sigma,
            #         alpha=0.05
            # ))
            prediction_collection.append(predictions)
            variance_collection.append(variances)

    target_collection = torch.cat(target_collection, dim=0)
    nll_collection = torch.cat(nll_collection, dim=0)
    update_performance_metrics(
        metrics_dict=metrics_dict,
        key="nll",
        value=torch.mean(nll_collection).item()
    )
    if is_classification:
        probs_collection = torch.cat(probs_collection, dim=0)
        update_performance_metrics(
            metrics_dict=metrics_dict,
            key="brier",
            value=get_brier(probs=probs_collection,
                            targets=target_collection)
        )
        
        update_performance_metrics(
            metrics_dict=metrics_dict,
            key="ece",
            value=ECE(post_pred=probs_collection,
                      y=target_collection)
        )
    else:
        prediction_collection = torch.cat(prediction_collection, dim=0)
        variance_collection = torch.cat(variance_collection, dim=0)
        coverage_collection = get_coverage(
            predictions=prediction_collection,
            variances=variance_collection,
            targets=target_collection,
            alpha=0.05
        )
        update_performance_metrics(
            metrics_dict=metrics_dict,
            key="coverage",
            value=torch.mean(coverage_collection).item()
        )
        # Calibration Error
        calibration_error = get_calibration_error(
                predictions=prediction_collection,
                variances=variance_collection,
                targets=target_collection,
        )    
        update_performance_metrics(
            metrics_dict=metrics_dict,
            key="calibration",
            value=calibration_error
        )

@hydra.main(config_path="config", config_name="config")
def run_main(cfg: DictConfig) -> None:
    # only use cfg for loading of results
    make_deterministic(cfg.seed)
    torch.set_default_dtype(getattr(torch, cfg.dtype))

    
    # paths
    results_path = os.path.join(
        "results", cfg.data.folder_name, cfg.pred_model.name, f"seed{cfg.seed}"
    )
    ## projector
    projector_path = os.path.join(results_path, "projector")
    projector_filename = f"projector_{cfg.projector.sigma.method.p}" + \
        f"_Psi{cfg.projector.sigma.method.psi}{cfg.projector.name_postfix}.npz"
    projector_file = os.path.join(projector_path, projector_filename)
    
    ## results
    namestring_bit = f"{cfg.projector.sigma.method.p}" + \
        f"_Psi{cfg.projector.sigma.method.psi}{cfg.results_file.name_postfix}"
    results_name = f"SigmaP_{namestring_bit}.pt"
    results_filename = os.path.join(results_path, results_name)
    predictive_metrics_filename = f"PredictiveMetrics_{namestring_bit}.pt"
    predictive_metrics_file = os.path.join(results_path, predictive_metrics_filename)
    
    # loading results
    print(f'Loading results from {results_filename}')
    with open(results_filename, "rb") as f:
        results = torch.load(
            f,
            map_location=cfg.device_torch,
            weights_only=False
        )

    # using cfg from results from now on
    print(f'Using configuration from {results_filename}')

    # load test dataset
    get_dataset_kwargs = results["get_dataset_kwargs"]
    print('Loading test dataset')
    train_data = get_dataset(**get_dataset_kwargs, train=True)
    test_data = get_dataset(**get_dataset_kwargs, train=False)


    # load model 
    get_model_kwargs = results["get_model_kwargs"]
    print('Loading model')
    model = get_model(**get_model_kwargs)
    with open(results["ckpt_file_name"], "rb") as f:
        state_dict = torch.load(f, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(cfg.device_torch)
    # switch off layers to ignore
    for module in model.modules():
        if type(module).__name__ in cfg.projector.layers_to_ignore:
            for par in module.parameters():
                par.requires_grad = False
    model.eval()
        

    # load s list
    s_list = results["s_list"]
    s_max_regularized = s_list[-1]
    with open_dict(cfg):
        cfg.projector.s_max_regularized = s_max_regularized

    # load regression likelihood sigma
    regression_likelihood_sigma = results["regression_likelihood_sigma"]


    # load IPsi
    IPsi = get_IPsi(
        method=cfg.projector.sigma.method.psi,
        cfg=cfg,
        model=model,
        data=train_data,
        path=projector_path,
        data_std=regression_likelihood_sigma,
    )

    # load/compute P
    if cfg.projector.sigma.method.p is None:
        P = None
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
            # construct P
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


    dl_test = DataLoader(
        dataset=test_data,
        batch_size=cfg.projector.batch_size,
        shuffle=False
    )


    # compute predictive to compute nll
    predictive = IPsi_predictive(
        model=model,
        IPsi=IPsi,
        P=P,
        chunk_size=cfg.projector.chunk_size,
        regression_likelihood_sigma=regression_likelihood_sigma,
    )


    # compute predictive metrics
    predictive_metrics = {'s_list': s_list}
    for s in s_list:
        def predictive_s(X: torch.Tensor):
            return predictive(X=X, s=s)
        update_predictive_metrics(
            metrics_dict=predictive_metrics,
            cfg=cfg,
            predictive=predictive_s,
            dataloader=dl_test,
        )
        



    with open(predictive_metrics_file, "wb") as f:
            torch.save(predictive_metrics, f)

if __name__ == "__main__":
    run_main()