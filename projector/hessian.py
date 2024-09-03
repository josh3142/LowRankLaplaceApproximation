import torch
from torch import nn, Tensor, log
from torch.func import functional_call, hessian
from torch.nn.functional import cross_entropy, nll_loss, gaussian_nll_loss
from torch.utils.data import DataLoader

from typing import Callable, Optional

from utils import vec_to_dict, param_to_vec

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_nll_fun(model: nn.Module, X: Tensor, Y: Tensor, fun: Callable=log
    ) -> Callable:
    """ 
    Returns a the negative fun-likelihood function of `model(X)`: `-log(model(X)_true_class`.
    The true class is obtained from `Y`.
    Args:
        model (nn.Module): `torch` model
        X (Tensor): input tensor
        Y (Tensor): label tensor (not one-hot encoded)
        fun (Callable): function which is applied on the result of `model(X)`.
            For `fun=log` (default) nll is obtained
    Return:
        Fun that computes the nll wrt to the parameters.
    """
    return lambda param_vec, param_gen: nll_loss(
        fun(functional_call(model, vec_to_dict(param_vec, param_gen), X)), 
        Y, 
        reduction="none")


def get_hessian_nll(model: nn.Module, X: Tensor, Y: Tensor, fun: Callable=log
    ) -> Tensor:
    """ 
    Compute the Hessian for negative log-likelihood loss.
    Return:
        Fun that computes the hessian of nll wrt the parameters. 
    """
    param_vec = param_to_vec(model.parameters())
    param_gen = model.named_parameters()
    H = hessian(get_nll_fun(model, X, Y, fun))(param_vec, param_gen)
    return H


def get_cross_entropy_fun(model: nn.Module, X: Tensor, Y: Tensor) -> Callable:
    """ 
    Returns a the cross entropy of `model(X)`. It expects that `model(X)` 
    yields the logits.
    Return:
        Fun that computes the cross entropy wrt to the parameters. 
    """
    return lambda param_vec, param_gen: cross_entropy(
        functional_call(model, vec_to_dict(param_vec, param_gen), X), 
        Y, 
        reduction="none")


def get_hessian_ce(model: nn.Module, X: Tensor, Y: Tensor) -> Tensor:
    """ 
    Compute the Hessian for cross entropy loss.
    Return:
        Fun that computes the hessian of cross entropy wrt the parameters. 
    """
    param_vec = param_to_vec(model.parameters())
    param_gen = model.named_parameters()
    H = hessian(get_cross_entropy_fun(model, X, Y))(param_vec, param_gen)
    return H


def get_gaussian_nll_fun(model: nn.Module, X: Tensor, Y: Tensor, var: float=1.0
    ) -> Callable:
    """ 
    Returns a the negative fun-likelihood function of 
    `(model(X) - Y)**2 / var`
    Args:
        model (nn.Module): `torch` model
        X (Tensor): Input tensor
        Y (Tensor): True observation
        var (float): Variance of Gaussian distribution
    Return:
        Fun that computes the Gaussian nll wrt to the parameters.
    """
    return lambda param_vec, param_gen: gaussian_nll_loss(
        input=functional_call(model, vec_to_dict(param_vec, param_gen), X), 
        target=Y,
        var=var,
        full=False,
        reduction="none")


def get_hessian_gaussian_nll(model: nn.Module, X: Tensor, Y: Tensor, var: float=1.0,
    ) -> Tensor:
    """ 
    Compute the Hessian for negative log-likelihood loss.
    Return:
        Fun that computes the hessian of Gaussian nll wrt the parameters.
    """
    param_vec = param_to_vec(model.parameters())
    param_gen = model.named_parameters()
    var = var * torch.ones_like(Y)

    H = hessian(get_gaussian_nll_fun(model, X, Y, var))(param_vec, param_gen)
    return H

#TODO: Should I implement chunk_size? How does this work for jacfwd?
def get_hessian(
        model: nn.Module,
        X: Tensor,
        Y: Tensor,
        is_classification: bool=False,
        var: float=1.0
    ) -> Tensor:
    """ 
    Computes the Hessian of cross entropy loss for vectorized parameters.
    Args:
        model: `torch` model.
        X: Input Tensor
        Y: Labels given as class indices if is_classification=True else true
            values for multi-output regression
        is_classification: Is it a classification task?
        var: Homoscedastic variance of the Gaussian distribution (only relevant
            if `is_classification=False`)
    """
    if is_classification:
        H = get_hessian_ce(model, X, Y)
    else:
        H = get_hessian_gaussian_nll(model, X, Y, var)
    return H

def get_H_sum(
        model: nn.Module,
        dl: DataLoader,
        is_classification: bool=False,
        n_batches: Optional[int]=None,
        var: float=1.0,
        **kwargs
    ) -> Tensor:
    """
    Computes the sum of Hessians of an entire `DataLoader`.

    Args:
        model: Parametrized model wrt to which the Hessian is computed
        dl: DataLoader on which the model is evaluated
        n_batches: Number of iterations the DataLoader should use. If n_batch is
            None the entire DataLoader is used.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    n_param = count_parameters(model)
    if n_batches is None:
        n_batches = len(dl.dataset)

    dl_iter = iter(dl)
    for _ in range(n_batches):
        try:
            X, Y = next(dl_iter)
            X, Y = X.to(device).to(dtype), Y.to(device)
            if torch.is_floating_point(Y):
                Y = Y.to(dtype)
                if len(Y.shape)==1:
                    Y = Y[:, None]
        except:
            break    
        H = get_hessian(
                model, 
                X, 
                Y,
                is_classification=is_classification,
                var=var
            ).detach().cpu()
        
        assert H.shape==Y.shape + (n_param, n_param), f"Hessian has wrong shape: {H.shape}."
        try:
            Hs = torch.sum(torch.cat([Hs, H], dim=0), dim=0, keepdim=True)
        except:
            Hs = H
    assert Hs.shape[-2:]==(n_param, n_param), "The last two dimensions of " + \
        f"the Hessian should match the dimensions of the parameters {n_param}"
    n_dim = Hs.dim()
    dims_to_keep = [n_dim-1, n_dim-2]
    dims_to_sum = [i for i in range(n_dim) if i not in dims_to_keep]
    Hs = torch.sum(Hs, dim=dims_to_sum)
    return Hs
