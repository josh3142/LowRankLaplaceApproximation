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
    Sigma_approx: torch.Tensor,
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
        value=trace(Sigma_approx=Sigma_approx, logarithmic=False),
    )
    update_performance_metrics(
        metrics_dict=metrics_dict,
        key="logtrace",
        value=trace(Sigma_approx=Sigma_approx, logarithmic=True),
    )

    # relative error
    if Sigma is not None:
        update_performance_metrics(
            metrics_dict=metrics_dict,
            key="rel_error",
            value=relative_error(Sigma_approx=Sigma_approx, Sigma=Sigma),
        )


@hydra.main(config_path="config", config_name="config")
def run_main(cfg: DictConfig) -> None:
    psi_ref = getattr(cfg, 'psi_ref', cfg.projector.sigma.method.psi)
 
    # set file names
    results_path = os.path.join(
        "results", cfg.data.folder_name, cfg.pred_model.name, f"seed{cfg.seed}"
    )
    name = f"{cfg.projector.sigma.method.p}" + \
        f"_Psi{cfg.projector.sigma.method.psi}{cfg.projector.name_postfix}"
    results_name = f"Metrics_{name}.pt"
    results_filename = os.path.join(results_path, results_name)
    SigmaP_name =f"SigmaP_{name}.pt"
    SigmaP_filename = os.path.join(results_path, SigmaP_name)
    Sigma_name = f"SigmaP_None" + \
        f"_Psi{psi_ref}{cfg.projector.name_postfix}.pt"
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
            Sigma_approx = Sigma_Ps_s['SigmaP']
        else:
            Sigma_approx = Sigma_Ps_s[f"SigmaP{s}"]
        update_Sigma_metrics(
            metrics_dict=results,
            Sigma_approx=Sigma_approx,
            Sigma=Sigma,
        )
    # store metrics
    with open(results_filename, "wb") as f:
        torch.save(results, f)


if __name__ == "__main__":
    run_main()
