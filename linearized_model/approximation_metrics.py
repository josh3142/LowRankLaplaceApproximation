import math
from typing import Callable, Literal, Optional, Tuple, Union
from torch.nn.functional import mse_loss
from torch.nn import KLDivLoss

from scipy.stats import norm, chi2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def check_condition_number(matrix: torch.Tensor, threshold: int=1e5):
    S = torch.svd(matrix)[1]
    assert S.max() / S.min() < threshold, "Condition nubmer is too large."   


def get_sqrt_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """ Compute the square root of a symmetric/ Hermitian matrix. """
    assert torch.allclose(matrix, matrix.H), "Matrix is not Hermitian."
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    eigvals = torch.clamp(eigvals, min=0.0) # make it more stable by preventing neg eigvals
    sqrt_eigvals = torch.diag(torch.sqrt(eigvals))
    return eigvecs @ sqrt_eigvals @ eigvecs.T


def relative_error(
        Sigma_approx: torch.Tensor,
        Sigma: torch.Tensor,
        norm: Callable=torch.linalg.norm
) -> float:
    return (norm(Sigma_approx.cpu()-Sigma.cpu())/norm(Sigma.cpu())).item()
  

def trace(Sigma_approx: torch.Tensor, logarithmic: bool=True) -> float:
    if logarithmic:
        return np.log(torch.trace(Sigma_approx).item())
    else:
        return torch.trace(Sigma_approx).item()

def KL_multivariate_normal(
        Sigma_approx: torch.Tensor,
        Sigma: torch.Tensor,
        epsilon: float = 1e-1,
) -> float:
    """Computes the Kullback-Leibler divergence between two multivariate normals 
    with equal mean, but different covariances `Sigma_approx` and `Sigma`.

    Args:
        Sigma_approx (torch.Tensor): covariance of approximate distribution
        Sigma (torch.Tensor): covariance of reference distribution

    Returns:
        float: KL divergence
    """
    # regularize Sigma and Sigma_approx
    regularizer = epsilon * torch.eye(Sigma.size(0)).to(Sigma.device)
    Sigma = Sigma + regularizer
    Sigma_approx = Sigma_approx + regularizer

    # check_condition_number(Sigma)

    trace_term = torch.trace(torch.linalg.inv(Sigma_approx) @ Sigma)
    logdet_term = torch.logdet(Sigma_approx) - torch.logdet(Sigma)

    return 0.5 * (trace_term + logdet_term - Sigma.size(0))


def get_kl_categorical(
        log_Y_full: torch.Tensor, log_Y_proj: torch.Tensor
    ) -> torch.Tensor:
    """ Computes the KL divergence for categorical distribution.
    
    Args:
        log_Y_full (torch.Tensor): log probabilities of the full LA
        log_Y_proj (torch.Tensor): log probabilities of the projected LA
    """
    return KLDivLoss(reduction="none", log_target=True)(log_Y_proj, log_Y_full)

def W2_multivariate_normal(
        Sigma_approx: torch.Tensor,
        Sigma: torch.Tensor,
) -> float:
    """Computes the Wasserstein-2-distance between two multivariate normals 
    with equal mean, but different covariances `Sigma_approx` and `Sigma`.

    Args:
        Sigma_approx (torch.Tensor): covariance of approximate distribution
        Sigma (torch.Tensor): covariance of reference distribution

    Returns:
        float: W2 distance
    """
    sqrt_Sigma = get_sqrt_matrix(Sigma)
    return torch.trace(Sigma + Sigma_approx - 2 * \
            get_sqrt_matrix(sqrt_Sigma @ Sigma_approx @ sqrt_Sigma)
        ).item()

def probit_approximation(
    predictions: torch.Tensor,
    variances: torch.Tensor,
    log: bool = False,
    diagonal_variances: bool = False,
) -> torch.Tensor:
        if not diagonal_variances:
            kappa = 1 / torch.sqrt(
                1 + math.pi/8 * torch.diagonal(variances, dim1=1, dim2=2))
        else:
            assert variances.shape == predictions.shape, "Shape mismatch"
            kappa = 1 / torch.sqrt(
                1 + math.pi/8 * variances)
        if log:
            return torch.nn.LogSoftmax(dim=-1)(kappa * predictions)
        else:
            return torch.nn.Softmax(dim=-1)(kappa * predictions)
    

def NLL(
        predictions: torch.Tensor,
        variances: torch.Tensor,
        targets: torch.Tensor,
        is_classification: bool = True,
        sum: bool = True,
        diagonal_variances: bool = False,
) -> torch.Tensor:
    """Computes the log posterior predictive, known as the "log likelihood" 
    in the literature on Bayesian Deep Learning. For classification the "probit
    approximation" of the posterior predictive is used (cf. Bishop)."""
    if is_classification:
        log_probs = probit_approximation(
            predictions=predictions,
            variances=variances,
            log=True,
            diagonal_variances=diagonal_variances,
        )
        ll = log_probs[torch.arange(log_probs.size(0)), targets]
    else:
        assert not diagonal_variances,\
            "Usage of diagonal variances only implemented for classification"
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
    """Collects the so-called "negative-log-likelihood" (NLL) on the data
    obtained by `dataloader`. If `reduction` is `none` the nll values for each
    datapoint are stored in a Tensor, otherwise they are summed
    (`reduction==sum`) or averaged (`reduction=mean`).

    Args:
        predictive (Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]): 
        Should return predictions and covariances (batched covariance matrices
        for each input) when called upon an input obtained from `dataloader`.
        dataloader (DataLoader): Should yield inputs and targets. 
        is_classification (bool): Specifies whether classification of regression
        problem is considered.
        device (Optional[torch.device], optional): Device for computation.
        reduction (Literal[&#39;sum&#39;, &#39;mean&#39;, &#39;none&#39;],
        optional): Specify whether NLLs for each datapoint are kept, averaged or
        summed.
        verbose (bool, optional): If `True` a status bar is printed.


    Returns:
        torch.Tensor: NLL as collection, sum or average (depending on
        `reduction`.)
    """
    nll_collection = []
    if verbose:
        dataloader_with_description = tqdm(dataloader, desc='Collecting LL')
    else:
        dataloader_with_description = dataloader
    for x, y in dataloader_with_description:
        if device is not None:
            x, y= x.to(device), y.to(device)
        predictions, variances = predictive(x)
        nll_collection.append(
            NLL(
                predictions=predictions,
                variances=variances,
                targets=y,
                is_classification=is_classification,
                sum=False
            )
        )
    nll_collection = torch.concat(nll_collection, dim=0)
    if reduction == 'none':
        return nll_collection
    elif reduction == 'sum':
        return torch.sum(nll_collection)
    elif reduction == 'mean':
        return torch.mean(nll_collection)
    else:
        raise NotImplementedError


def get_brier(probs: torch.Tensor,
              targets: torch.Tensor) -> float:
    """ Compute the Brier score. """
    def one_hot(targets, n_class):
        targets = targets
        res = torch.eye(n_class, device=targets.device)[targets.view(-1)]
        return res.reshape(list(targets.shape) + [n_class])

    return mse_loss(probs, one_hot(targets, probs.shape[-1]))

def ECE(
    post_pred: torch.Tensor,
    y: torch.Tensor,
    p_bins: np.ndarray=np.linspace(0,1,20)
) -> float:
    """
    Computes the ECE ("Expected Calibration Error") from average bin
    confidence, average bin accuracy and bin size for the bins specified by
    `p_bins`.
    """
    # check probabilities and targets
    assert len(post_pred.shape) == 2, "Too many dimensions of post predictive"
    assert torch.allclose(torch.sum(post_pred, dim=1).cpu(),
                          torch.tensor([1.0]))
    assert len(y.shape) == 1, "Too many dimensions of y"

    # turn into numpy arrays
    post_pred = post_pred.cpu().numpy()
    y = y.cpu().numpy()

    # obtain predicted classes
    pred = np.argmax(post_pred, axis=1)
    # confidence behind prediction
    p_max = np.amax(post_pred, axis=1)

    # to compute confidence, accuravy and size for a *single* bin
    def conf_acc_bsize_in_bin(pred: np.ndarray, y: np.ndarray,
        p: np.ndarray,
        p_bin: tuple[float, float],
        first_bin: bool =False) -> Union[tuple[float,float,int],None]:
        """
        Computes the average bin confidence, average bin accuracy and the
        number of elements in a bin specified by a tuple `p_bin`. When no
        elements with confidence in `p_bin` exist, `None` is returned.
        """
        if first_bin:
            idx = np.logical_and(p_bin[0] <= p, p <= p_bin[1])
        else:
            idx = np.logical_and(p_bin[0] < p, p <= p_bin[1])
        if np.sum(idx) == 0:
            return None
        else:
            return float(np.mean(p[idx])), float(np.mean(pred[idx] == y[idx])), np.sum(idx)


    # collect all confidences accuracies and bin sizes
    conf_collection = []
    acc_collection = []
    bsize_collection = []
    for i in range(len(p_bins)-1):
        if i==0:
            first_bin = True
        else:
            first_bin = False
        conf_acc_bsize = conf_acc_bsize_in_bin(pred=pred, y=y, p=p_max,
                                p_bin=(p_bins[i], p_bins[i+1]),
                                first_bin=first_bin)
        if conf_acc_bsize:
            conf_collection.append(conf_acc_bsize[0])
            acc_collection.append(conf_acc_bsize[1])
            bsize_collection.append(conf_acc_bsize[2])
    conf = np.array(conf_collection)
    acc = np.array(acc_collection)
    bsize = np.array(bsize_collection)

    return np.sum(bsize * np.abs(conf-acc))/np.sum(bsize)

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

def get_coverage(
    predictions: torch.Tensor,
    variances: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.05, 
) -> float:
    """
    Compute coverage: fraction of targets within (1-alpha) prediction ellipsoid.
    
    Uses Mahalanobis distance with full covariance matrix to determine
    if points fall within the (1-alpha) confidence ellipsoid.
    
    Args:
        predictions: (N, C) predicted means
        variances: (N, C, C) predicted covariances (full, including aleatoric)
        targets: (N, C) true targets
        alpha: float, significance level (0.05 for 95% CI)
    
    Returns:
        coverage: float in [0, 1], should be close to (1-alpha) if well-calibrated
    """   
    N, C = predictions.shape
    
    # Compute Mahalanobis distance for each datapoint
    batch_inv = torch.vmap(torch.linalg.inv, in_dims=0)
    inv_cov = batch_inv(variances)                    # (N, C, C)
    diff = (targets - predictions).unsqueeze(-1)      # (N, C, 1)
    
    mahal_sq = torch.matmul(torch.matmul(diff.transpose(1,2), inv_cov), diff)
    mahal_sq = mahal_sq.squeeze()                     # shape (N,)
    
    # Critical radius for (1-alpha) confidence ellipsoid
    # Chi-square distribution with C degrees of freedom
    r_crit_sq = chi2.ppf(1 - alpha, df=C)
    
    # Check if within ellipsoid
    in_ellipsoid = (mahal_sq <= r_crit_sq).float()
    
    return in_ellipsoid


def get_calibration_error(
    predictions: torch.Tensor,
    variances: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 5,
) -> float:
    """
    - uses the full covariance matrix Sigma
    - uses Mahalanobis distance for standardized residuals
    - computes calibration of (1 - alpha) ellipsoidal regions
    """
    N, C = predictions.shape

    # Mahalanobis distance for each datapoint
    batch_inv = torch.vmap(torch.linalg.inv, in_dims=0)
    inv_cov = batch_inv(variances)                    # (N, C, C)
    diff = (targets - predictions).unsqueeze(-1)      # (N, C, 1)

    mahal_sq = torch.matmul(torch.matmul(diff.transpose(1,2), inv_cov), diff)
    mahal_sq = mahal_sq.squeeze()                     # shape (N,)

    # Calibration over confidence levels
    confidence_levels = torch.linspace(0.1, 0.95, n_bins)
    calibration_errors = []

    for conf_level in confidence_levels:
        # Critical radius for chi-square distribution with C dimensions
        r_crit_sq = chi2.ppf(conf_level, df=C)

        empirical_coverage = torch.mean((mahal_sq <= r_crit_sq).float()).item()
        expected_coverage = conf_level.item()

        calibration_errors.append(abs(empirical_coverage - expected_coverage))

    return float(np.mean(calibration_errors))