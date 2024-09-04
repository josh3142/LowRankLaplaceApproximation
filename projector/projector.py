from scipy.linalg import subspace_angles
import numpy as np
import cupy as cp
from cupy.linalg import qr

from typing import Callable

from projector.hessian import get_H_sum
from projector.fisher import get_Vs, get_I_sum


def get_projector_fun(name: str) -> Callable:
    """ Return a function that computes the projector. """
    if name=="projector1d":
        raise NotImplementedError()
    else:
        raise NotImplementedError(name)
    
    return projector_fun


def get_hessian_type_fun(name: str) -> Callable:
    """ Return a function that computes the (approximated) Hessian. """
    if name=="H":
        return get_H_sum
    elif name=="Ihalf":
        return get_Vs
    elif name=="I":
        return get_I_sum
    else:
        raise NotImplementedError(name)
    
    return projector_fun


def get_angle_between_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray: 
    """
    Compute the angles between to subspace represented by matrices `A` and `B`.
    The angles are returned in degree.
    """
    assert len(A.shape) == len(B.shape) == 2, "The matrices have the wrong shape."
    (n1, _), (m1, _) = A.shape, B.shape
    assert n1==m1, "The first dimensions of both matrices have to coincide."
    return np.rad2deg(subspace_angles(A, B))


def get_residual(MV: cp.ndarray, V: cp.ndarray) -> cp.ndarray:
    """
    Given a set of orthogonal vectors `V` and the vectors `MV = M @ V` on which
    the matrix `M` is applied, the residual is computed.
    If `V` forms an invariant subspace under the action of `M`, `V == MV`.
    """
    Q = qr(MV)[0] # qr decomposition
    return Q - V @ (V.T @ Q) # residual 