import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import numpy as np

from typing import Optional, Iterable

from projector.projector1d import get_jacobian


def get_I_outer_product(V: Tensor, reduction: str="sum") -> Tensor:
    """
    Computes the Fisher Information `I` if it can be written as an outer product.
    
    This works for a Gaussian distribution
    ```
        I = sum_bc V_bc V_bc 
        f_bc = softmax output, V_bc = nabla f_bc / sqrt(var)
    ```
    and a categorical distribution
    ```
        I = sum_bc V_bc V_bc
        f_bc = softmax output, V_bc = 2 nabla sqrt f_bc / sqrt(var)
    ```
    Args:
        V: Vector that factorizes I
        reduction: sum or mean of the samples
    """
    I = torch.einsum("bci, bcj -> ij", V, V)
    if reduction=="sum":
        return I
    elif reduction=="mean":
        return I / len(V)


def get_I_softmax_ln(jac_ln: Tensor, y_hat: Tensor, reduction: str="sum"
    ) -> Tensor:
    """
    Computes the Fisher Information `I` for the categorical distribution with
    probability vector `y_hat` provided by softmax classification.
    ```
        I = sum_bc jac_ln_bc jac_ln_bc y_hat_bc
        y_hat = f_bc = softmax output, jac_ln_bc = nabla ln f_bc
    ```
    Args:
        jac_ln: Jacobian of the logarithm of the probability vector
        y_hat: probability vector
        reduction: sum or mean of the samples
    """
    I = torch.einsum("bc, bci, bcj -> ij", y_hat, jac_ln, jac_ln)
    if reduction=="sum":
        return I
    elif reduction=="mean":
        return I / len(y_hat)


def get_Vs(
        model: nn.Module, 
        dl: DataLoader, 
        is_classification: bool=False,
        n_batches: Optional[int]=None,
        chunk_size: Optional[int]=None,
        var: float=1.0,
        **kwargs
    ) -> Tensor:
    """
    Computes the gradients to compute the Fisher Information I iteratively 
    and returns all of them on the cpu.
    
    Args:
        model: Model to compute the score vectors V
        dl: DataLoader whose data is used to compute V
        n_batches: Number of iterations the DataLoader should use. If n_batch is
            None the entire DataLoader is used.
        var: Homoscedastic variance of the Gaussian distribution (only relevant
            if `is_classification=False`)

    Note:
        For Gaussian distribution the Fisher Information is the outer product of
        the gradients of Y_hat.
        For categorical distribution the Fisher Information is the outer product
        of the gradient of the sqrt of the softmax classifier multiplied by two.
    """
    model.eval()
    device = next(model.parameters()).device
    if n_batches is None:
        n_batches = len(dl.dataset)

    # define function to obtain the correct gradiends
    if is_classification:
        fun = lambda x: 2 * torch.sqrt(x)
    else:
        fun = lambda x: x

    Vs = []
    dl_iter = iter(dl)
    for _ in range(n_batches):
        try:
            X = next(dl_iter)[0]
            X = X.to(device)
        except:
            break    
        V = get_jacobian(
                model, 
                X, 
                fun=fun,
                is_classification=is_classification,
                chunk_size=chunk_size
            ).detach().cpu()
        Vs.append(V)
    # Since the Fisher Information is an outer product of V. Only the squareroot
    # of the variance `var` is taken.
    Vs = torch.cat(Vs, dim=0) / np.sqrt(var)
    return Vs


def get_I_sum(
        model: nn.Module, 
        dl: DataLoader, 
        is_classification: bool=False,
        n_batches: Optional[int]=None,
        chunk_size: Optional[int]=None,
        var: float=1.0,
        **kwargs):
    device = next(model.parameters()).device

    Vs = get_Vs(model, dl, is_classification, n_batches, chunk_size, var)
    try:
        I = get_I_outer_product(Vs.to(device))
    except:
        I = get_I_outer_product(Vs)
    return I

def get_V_iterator(
        model: nn.Module, 
        dl: DataLoader, 
        is_classification: bool=False,
        n_batches: Optional[int]=None,
        chunk_size: Optional[int]=None,
        var: float=1.0,
        **kwargs
    ) -> Iterable:
    """
    Creates an iterator that computes the gradients `V` to compute the Fisher
    Information I =V^T @ V and yields them.
    
    Args:
        model: Model to compute the score vectors V
        dl: DataLoader whose data is used to compute V
        n_batches: Number of iterations the DataLoader should use. If n_batch is
            None the entire DataLoader is used.
        var: Homoscedastic variance of the Gaussian distribution (only relevant
            if `is_classification=False`)
    """
    model.eval()
    device = next(model.parameters()).device
    if n_batches is None:
        n_batches = len(dl.dataset)

    # define function to obtain the correct gradiends
    if is_classification:
        fun = lambda x: 2 * torch.sqrt(x)
    else:
        fun = lambda x: x

    Vs = []
    dl_iter = iter(dl)
    for _ in range(n_batches):
        try:
            X = next(dl_iter)[0]
            X = X.to(device)
        except:
            break    
        V = get_jacobian(
                model, 
                X, 
                fun=fun,
                is_classification=is_classification,
                chunk_size=chunk_size
            ).detach()
        # Since the Fisher Information is an outer product of V. Only the squareroot
        # of the variance `var` is taken.
        yield V/np.sqrt(var)
