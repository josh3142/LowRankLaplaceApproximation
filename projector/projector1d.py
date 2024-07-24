import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.func import jacrev

import cupy as cp
import numpy as np

from functools import partial
from typing import Optional, Callable

from utils import param_to_vec, get_softmax_model_fun, get_model_fun


def get_gradient_projector_1d(p: Tensor, j: Tensor, V: Tensor) -> Tensor:
    """
    Computes the gradient of a one-dimensional projector of the projected
    linearized covariance matrix.
    Args:
        j: Gradient of sample onto which the covariance matrix is projected.
        V: Rescaled score that are used to compute the Fisher Information.
        p: projector
    """
    jp = torch.einsum("a, a", j, p)
    pp = torch.einsum("a, a", p, p)
    pV = torch.einsum("a, ai -> i", p, V)
    pVV = torch.einsum("a, ai -> i", pV, V.T)
    pVVp = torch.einsum("a, a", pV, pV)
    fac = jp / (pp + pVVp)

    return fac * j - fac**2 * (pVV + p)


def get_sigma_projected_1d(p: Tensor, j: Tensor, V: Tensor) -> Tensor:
    """
    Compute the one-dimensional projected linearized covariance matrix.
    """
    jp = torch.einsum("a, a", j, p)
    pp = torch.einsum("a, a", p, p)
    pV = torch.einsum("a, ai -> i", p, V)
    pVVp = torch.einsum("a, a", pV, pV)

    return jp**2 / (pp + pVVp)


def get_sigma_1d(j: Tensor, V: Tensor) -> Tensor:
    """
    Compute a one-dimensional linearized covaraince matrix.
    """
    jj = torch.einsum("a, a", j, j)
    jV = torch.einsum("a, ai -> i", j, V)
    VV = torch.einsum("ia, aj -> ij", V.T, V)
    VV_inv = torch.linalg.inv(torch.eye(V.shape[-1], device=V.device) + VV)
    fac = torch.einsum("i, ij, j", jV, VV_inv, jV)
    return jj - fac

def get_jacobian(
        model: nn.Module, 
        X: Tensor, 
        fun: Callable=lambda x: x, 
        is_classification: bool=True,
        chunk_size: Optional[int]=None
    ) -> Tensor:
    """
    Returns Jacobian for a given `model` and input `X`.

    Args:
        model: pytorch model
        X: Input tensor of `model`.
        fun: function that maps the output of the model to the desired output.
        is_classification: If `True`, a softmax classifier is concatenated with 
            the output of `model` (it is expected the model returns logits). 
            Otherwise the jacobian of the output of `model` is computed.
        chunk_size: If None (default), use the maximum chunk size (equivalent 
            to doing a single vmap over vjp to compute the jacobian)
    """
    param_vec = param_to_vec(model.parameters())
    if is_classification:
        J = jacrev(
            partial(
                    get_softmax_model_fun, 
                    param_gen=model.named_parameters(), 
                    model=model, 
                    X=X,
                    fun=fun
                ),
            chunk_size=chunk_size)(param_vec)
    else:
        J = jacrev(
            partial(
                    get_model_fun, 
                    param_gen=model.named_parameters(), 
                    model=model, 
                    X=X,
                    fun=fun
                ),
            chunk_size=chunk_size)(param_vec)
    return J

# def get_lhs_linear_equ_of_1d_projector(
#         w_cp: cp.ndarray, 
#         model: nn.Module, 
#         dl: DataLoader, 
#         is_classification: bool=False,
#         n_batches: Optional[int]=None,
#         chunk_size: Optional[int]=None
#     ) -> cp.ndarray:
#     """
#     Compute the left-hand side of a system of linear equations  `A w_cp = b`.

#     The lhs of the system of linear equations is given by `A w_cp` where
#     `A = (1 + VV^T)`.
#     Args:
#         w_cp: vector multiplied with A to obtain lhs.
#         model: Model to compute the score vectors V
#         dl: DataLoader whose data is used to compute V
#         n_batch: Number of iterations the DataLoader should use. If n_batch is
#             None the entire DataLoader is used.
#     """
#     w_th = torch.from_dlpack(w_cp).squeeze()
#     w_out = 0

#     if n_batches is None:
#         n_batches = len(dl.dataset)

#     dl_iter = iter(dl)
#     for _ in range(n_batches):
#         try:
#             X = next(dl_iter)[0]
#             X = X.to(w_th.device)
#             X = X.to(torch.float64)
#         except:
#             break    
#         V = get_jacobian(
#                 model, 
#                 X, 
#                 fun=torch.log,
#                 is_classification=is_classification,
#                 chunk_size=chunk_size
#             ).detach()

#         V = V.reshape(-1, V.shape[-1]).T
#         w_out = w_out + torch.einsum("pi, qi, q -> p", V, V, w_th)  
#     w_out = w_out + w_th
#     w_out_cp = cp.from_dlpack(w_out)
#     return w_out_cp


def get_lhs_linear_equ_of_1d_projector(
        w_cp: cp.ndarray, 
        Vs: torch.Tensor,
        batch_size: int
    ) -> cp.ndarray:
    """
    Compute the left-hand side of a system of linear equations  `A w_cp = b`.

    The lhs of the system of linear equations is given by `A w_cp` where
    `A = (1 + VV^T)`.
    Args:
        w_cp: vector multiplied with A to obtain lhs.
        Vs: Model to compute the score vectors `V`
        batch_size: Number of score vectors `V` used to accumulate `w_cp`
            in one step
    """
    w_out = 0
    rounds = int(np.ceil(Vs.shape[-1] / batch_size))
    for i in range(rounds): 
        V = cp.asarray(Vs[:, i * batch_size: (i + 1 ) * batch_size])
        w_out = w_out + cp.einsum("pi, qi, q -> p", V, V, w_cp)  
    w_out = w_out + w_cp
    return w_out

def get_Vs(
        model: nn.Module, 
        dl: DataLoader, 
        is_classification: bool=False,
        n_batches: Optional[int]=None,
        chunk_size: Optional[int]=None
    ) -> Tensor:
    """
    Computes the score vectors iteratively and returns all of them on the cpu.
    
    Args:
        model: Model to compute the score vectors V
        dl: DataLoader whose data is used to compute V
        n_batches: Number of iterations the DataLoader should use. If n_batch is
            None the entire DataLoader is used.
    """
    device = next(model.parameters()).device
    if n_batches is None:
        n_batches = len(dl.dataset)

    Vs = []
    dl_iter = iter(dl)
    for _ in range(n_batches):
        try:
            X = next(dl_iter)[0]
            X = X.to(device)
            X = X.to(torch.float64)
        except:
            break    
        V = get_jacobian(
                model, 
                X, 
                fun=torch.log,
                is_classification=is_classification,
                chunk_size=chunk_size
            ).detach().cpu()
        Vs.append(V.reshape(-1, V.shape[-1]).T)
    Vs = torch.cat(Vs, dim=-1)
    return Vs

# def get_least_square_error(
#         j: Tensor, 
#         p: Tensor,
#         model: nn.Module,
#         dl: DataLoader, 
#         is_classification: bool=False
        # chunk_size: Optional[int]=None
#         ) -> float:
#     """
#     Compute the error ||j - VV^Tp - p||_2^2

#     Args:
#         j: gradient of test sample
#         p: projector from projected posterior covariance matrix
#         model: Model to compute the score vectors V
#         dl: DataLoader whose data is used to compute V
#     """
#     # compute lhs = VV^Tp + p 
#     lhs = 0
#     dl_iter = iter(dl)
#     for _ in range(len(dl.dataset)):
#         try:
#             X = next(dl_iter)[0]
#             X = X.to(p.device)
#             X = X.to(torch.float64)
#         except:
#             break    
#         V = get_jacobian(
#                 model, 
#                 X, 
#                 fun=torch.log,
#                 is_classification=is_classification,
#                 chunk_size=chunk_size
#             ).detach()
#         V = V.reshape(-1, V.shape[-1]).T
#         lhs = lhs + torch.einsum("pi, qi, q -> p", V, V, p)
#     lhs = lhs + p

#     # normalize 1st component lhs such that the j and lhs can be compared
#     # Note: Theoretically, it does not matter wrt which index the normalization
#     # is obtained, but numerically the quotient can be instable if the 
#     # coefficient is close to zero
#     idx = torch.argmax(torch.abs(j)).item()
#     normalization = j[idx] / lhs[idx] 
#     lhs = normalization * lhs

#     # compute error 
#     error = ((torch.sum(lhs - j)**2)**0.5).item()
#     return error


def get_least_square_error(
        j: Tensor, 
        p: Tensor,
        Vs: torch.Tensor,
        batch_size: int,
        device: str
        ) -> float:
    """
    Compute the error ||j - VV^Tp - p||_2^2

    Args:
        j: gradient of test sample
        p: projector from projected posterior covariance matrix
                Vs: Model to compute the score vectors `V`
        batch_size: Number of score vectors `V` used to accumulate `w_cp`
            in one step
    """
    # compute lhs = VV^Tp + p 
    lhs = 0
    rounds = int(np.ceil(Vs.shape[-1] / batch_size))
    for i in range(rounds): 
        V = Vs[: , i * batch_size: (i + 1 ) * batch_size].to(device)
        lhs = lhs + torch.einsum("pi, qi, q -> p", V, V, p)
    lhs = lhs + p

    # normalize 1st component lhs such that the j and lhs can be compared
    # Note: Theoretically, it does not matter wrt which index the normalization
    # is obtained, but numerically the quotient can be instable if the 
    # coefficient is close to zero
    idx = torch.argmax(torch.abs(j)).item()
    normalization = j[idx] / lhs[idx] 
    lhs = normalization * lhs

    # compute error 
    error = ((torch.sum(lhs - j)**2)**0.5).item()
    return error

def get_inv(V: Tensor | np.ndarray) -> Tensor | np.ndarray:
    """ Compute (1 + V V^T)^-1 """
    try:
        return torch.linalg.inv(torch.eye(V.shape[1], device=V.device) + V.T @ V)
    except:
        return np.linalg.inv(np.eye(V.shape[1]) + V.T @ V)

def get_pred_var(J: Tensor | np.ndarray, V: Tensor | np.ndarray
    ) -> Tensor | np.ndarray:
    """ Compute the predictive covariance matrix with the Woodbury identity. """
    V_inv = get_inv(V)
    try:
        t1 = torch.einsum("cp, dp -> cd", J, J)
        t2 = torch.einsum("cp, po, om, nm, dn -> cd", J, V, V_inv, V, J)
    except:    
        t1 = np.einsum("cp, dp -> cd", J, J)
        t2 = np.einsum("cp, po, om, nm, dn -> cd", J, V, V_inv, V, J)
    return t1 - t2

