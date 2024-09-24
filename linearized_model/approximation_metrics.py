from typing import Optional, Callable
import torch
import numpy as np

def rho(eigenvalues: Optional[torch.Tensor]=None,
    pred_cov: Optional[torch.Tensor]=None,
    normalize=False) -> np.ndarray:
    # assert 1D vector
    if eigenvalues is None:
        assert pred_cov is not None
        eigenvalues = torch.linalg.svdvals(pred_cov)
    assert len(eigenvalues.shape) == 1
    rho_values = [torch.sqrt(torch.sum(eigenvalues[:s])/torch.sum(eigenvalues)).item()
                for s in range(len(eigenvalues))]
    return np.array(rho_values)



def relative_error(Sigma_approx: torch.Tensor, Sigma: torch.Tensor,
                   norm: Callable=torch.linalg.norm) -> float:
    return (norm(Sigma_approx-Sigma)/norm(Sigma)).item()
    
def trace(Sigma_approx: torch.Tensor)-> float:
       return torch.trace(Sigma_approx).item()
 

def update_performance_metrics(metrics_dict: dict, Sigma_approx: torch.Tensor, 
                Sigma: Optional[torch.Tensor]) -> None:
    """
    Method to collect metrics to evaluate approximate predictive
    covariances. Upon call the dictionary `metrics_dict` is updated and
    the metrics `relative_error` (if `Sigma` is not `None` ) and `trace` are
    computed and stored in `metrics_dict`. If values are already present in
    `metrics_dict` the computed values are appended. 

    *Note*: This method does **in place** operations on `metrics_dict`.
    """
    # relative error
    if Sigma is not None:
        if 'rel_error' not in metrics_dict.keys():
            metrics_dict['rel_error'] = []
        rel_error_value = relative_error(Sigma_approx=Sigma_approx,
                                            Sigma=Sigma)
        metrics_dict['rel_error'].append(rel_error_value)
    # trace
    if 'trace' not in metrics_dict.keys():
        metrics_dict['trace'] = []
    trace_value = trace(Sigma_approx=Sigma_approx)
    metrics_dict['trace'].append(trace_value)

    # trace quotient
    if Sigma is not None:
        if 'trace_quotient' not in metrics_dict.keys():
            metrics_dict['trace_quotient'] = []
        trace_quotient_value = trace(Sigma_approx=Sigma_approx)/trace(Sigma)
        metrics_dict['trace_quotient'].append(trace_quotient_value)
