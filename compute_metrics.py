import os

from typing import Optional

import hydra
from omegaconf import DictConfig
import torch

from linearized_model.approximation_metrics import (
    update_performance_metrics,
    relative_error,
    trace
)

def update_Sigma_metrics(
    metrics_dict: dict,
    SigmaP: torch.Tensor,
    Sigma: Optional[torch.Tensor]=None,
):
    """ 
    Summarizes the metric collection for computed Sigma approximation. 
    
    If `Sigma` is not `None` the relative error between the reduced predictive
    covariance `Sigma_P` and the full predictive covariance `Sigma` is 
    computed.
    """
    # trace
    update_performance_metrics(
        metrics_dict=metrics_dict,
        key="trace",
        value=trace(Sigma_approx=SigmaP),
    )

    # relative error
    if Sigma is not None:
        update_performance_metrics(
            metrics_dict=metrics_dict,
            key="rel_error",
            value=relative_error(Sigma_approx=SigmaP, Sigma=Sigma),
        )


@hydra.main(config_path="config", config_name="config")
def run_main(cfg: DictConfig) -> None:
 
    # set file names
    results_path = os.path.join(
        "results", cfg.data.name, cfg.pred_model.name, f"seed{cfg.seed}"
    )
    results_name = f"MetricsSigmaP{cfg.projector.sigma.method.p}" + \
        f"Psi{cfg.projector.sigma.method.psi}{cfg.projector.name_postfix}.pt"
    results_filename = os.path.join(results_path, results_name)
    SigmaP_name =f"SigmaP{cfg.projector.sigma.method.p}" + \
        f"Psi{cfg.projector.sigma.method.psi}{cfg.projector.name_postfix}.pt"
    SigmaP_filename = os.path.join(results_path, SigmaP_name)
    Sigma_name = f"SigmaPNone" + \
        f"Psi{cfg.projector.sigma.method.psi}{cfg.projector.name_postfix}.pt"
    Sigma_filename = os.path.join(results_path, Sigma_name)

    # load predictive ccovariance matrices
    try:
        Sigma = torch.load(Sigma_filename)["SigmaP"]
    except:
        Sigma = None
        print("Non-reduced predictive covariance matrix does not exists. " + 
              "The relative error is not computed.")
    Sigma_Ps_s = torch.load(SigmaP_filename)
    s_list = Sigma_Ps_s["s_list"]

    # compute metrics
    results = {"s_list": s_list}
    for s in s_list:
        if s is None:
            continue
        Sigma_P_s = Sigma_Ps_s[f"SigmaP{s}"]
        update_Sigma_metrics(
            metrics_dict=results,
            SigmaP=Sigma_P_s,
            Sigma=Sigma,
        )
    # store metrics
    with open(results_filename, "wb") as f:
        torch.save(results, f)


if __name__ == "__main__":
    run_main()
