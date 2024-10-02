import math
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def rho(
        eigenvalues: Optional[torch.Tensor] = None,
        pred_cov: Optional[torch.Tensor] = None
    ) -> np.ndarray:
    # assert 1D vector
    if eigenvalues is None:
        assert pred_cov is not None
        eigenvalues = torch.linalg.svdvals(pred_cov)
    assert len(eigenvalues.shape) == 1
    rho_values = [torch.sqrt(torch.sum(eigenvalues[:s])/torch.sum(eigenvalues)).item()
                    for s in range(len(eigenvalues))]
    return np.array(rho_values)



def relative_error(
        Sigma_approx: torch.Tensor,
        Sigma: torch.Tensor,
        norm: Callable=torch.linalg.norm
    ) -> float:
    return (norm(Sigma_approx-Sigma)/norm(Sigma)).item()
  

def trace(Sigma_approx: torch.Tensor)-> float:
       return torch.trace(Sigma_approx).item()





def NLL(
        predictions: torch.Tensor,
        variances: torch.Tensor,
        targets: torch.Tensor,
        is_classification: bool = True,
        sum: bool =True
    ) -> torch.Tensor:
    """Computes the log posterior predictive, known as the "log likelihood" 
    in the literature on Bayesian Deep Learning. For classification the "probit
    approximation" of the posterior predictive is used (cf. Bishop)."""
    if is_classification:
        kappa = 1 / torch.sqrt(
            1 + math.pi/8 * torch.diagonal(variances, dim1=1, dim2=2))
        log_probs = torch.nn.LogSoftmax(dim=-1)(kappa * predictions)
        ll = log_probs[torch.arange(log_probs.size(0)), targets]
    else:
        number_of_parameters = variances.size(-1)
        log_normalization = 0.5 * (number_of_parameters * math.log(2 * math.pi) + 
                                         torch.logdet(variances))
        error = targets - predictions
        batch_inv = torch.vmap(torch.linalg.inv, in_dims=0)
        square_error = -0.5 * torch.einsum('bt,btT,Tb->b', error,
                                           batch_inv(variances), error.T)
        ll = square_error - log_normalization
    if not sum:
        return -1.0 * ll
    else:
        return -1.0 * torch.sum(ll)


def collect_NLL(
        predictive: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        dataloader: DataLoader,
        is_classification: bool,
        device: Optional[torch.device]=None,
        reduction: Literal['sum', 'mean', 'none']='mean',
        verbose: bool = True
    ) -> torch.Tensor:
    ll_collection = []
    if verbose:
        dataloader_with_description = tqdm(dataloader, desc='Collecting LL')
    else:
        dataloader_with_description = dataloader
    for x, y in dataloader_with_description:
        if device is not None:
            x, y= x.to(device), y.to(device)
        predictions, variances = predictive(x)
        ll_collection.append(
            NLL(
                predictions=predictions,
                variances=variances,
                targets=y,
                is_classification=is_classification,
                sum=False
            )
        )
    ll_collection = torch.concat(ll_collection, dim=0)
    if reduction == 'none':
        return ll_collection
    elif reduction == 'sum':
        return torch.sum(ll_collection)
    elif reduction == 'mean':
        return torch.mean(ll_collection)
    else:
        raise NotImplementedError


def update_performance_metrics(
        metrics_dict: dict, key: str, 
        value: Union[float, torch.Tensor],
        tensor_to_float: bool = True
    ) -> None:
    """
    Method to collect metrics. Checks whether `key` is contained in
    `metrics_dict`. If it is not, `[value,]` is stored under this key. If it is,
    `value` is appended to the list under this key. If `tensor_to_float` is
    `True` any tensor `value` is converted into a float if possible.
    """
    if tensor_to_float:
        if type(value) is torch.Tensor:
            if value.numel() == 1:
                value = value.item()
    if key not in metrics_dict.keys():
        metrics_dict[key] = [value,]
    else:
        assert type(metrics_dict[key]) is list
        metrics_dict[key].append(value)