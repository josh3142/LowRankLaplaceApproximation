from typing import Optional
import os

from scipy.linalg import subspace_angles
import numpy as np
import cupy as cp
from cupy.linalg import qr

import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader

from omegaconf import DictConfig
from typing import Callable, Literal

import laplace
from laplace import FullLaplace, KronLaplace

from projector.hessian import get_H_sum
from projector.fisher import get_Vs, get_I_sum
from linearized_model.low_rank_laplace import (
    InvPsi,
    FullInvPsi,
    HalfInvPsi,
    KronInvPsi,
    compute_optimal_P
)
from linearized_model.subset import subset_indices
from projector.fisher import get_V_iterator
from projector.projector1d import create_jacobian_data_iterator

from utils import make_deterministic

def get_IPsi(
        method: Literal["ggnit", "load_file", "kron", "diagonal", "full"], 
        cfg: DictConfig, 
        model: nn.Module, 
        data: Dataset, 
        path: str
    ) -> InvPsi:
    """
    Wrapper to get posterior `Psi`.

    Args:
        method: Method to compute the posterior. 
            `ggnit` computes generalized Gauss-Newton matrix as an iterator 
            `load_file` loads a precomputed posterior
            `kron`, `full` and `diagonal` compute `Psi` 
            with the Laplace lib
        cfg: Configurations file
        model: Pytorch model
        data: Pytorch Dataset
        path: string to point to the file loaded by `load_file`
    """
    dtype = getattr(torch, cfg.dtype)

    if method=="ggnit":
        def compute_psi_ggn_iterator(cfg, model, data):
            dl = DataLoader(
                dataset=data,
                batch_size=cfg.projector.v.batch_size,
                shuffle=False
                )
            def create_V_it():
                return get_V_iterator(
                    model=model,
                    dl=dl,
                    is_classification=cfg.data.is_classification,
                    n_batches=cfg.projector.v.n_batches,
                    chunk_size=cfg.projector.chunk_size,
                )
            IPsi = HalfInvPsi(
                V=create_V_it,
                prior_precision=cfg.projector.sigma.prior_precision
            )
            return IPsi
        return compute_psi_ggn_iterator(cfg, model, data)
    
    elif method in ["kron", "full"]:
        likelihood = "classification" if cfg.data.is_classification \
            else "regression"
        dl = DataLoader(
            dataset=data,
            batch_size=cfg.projector.v.batch_size,
            shuffle=False
            )
        make_deterministic(cfg.seed)
        la = laplace.Laplace(
                    model=model,
                    hessian_structure=method,
                    likelihood=likelihood,
                    subset_of_weights="all",
                    prior_precision=cfg.projector.sigma.prior_precision,
                )
        la.fit(dl)
        if method=="kron":
            assert type(la) is KronLaplace
            return KronInvPsi(inv_Psi=la)

        elif method=="full":
            assert type(la) is FullLaplace
            return FullInvPsi(inv_Psi=la.posterior_precision.to(dtype))
        
        elif method=="diagonal":
            #TODO: Implement this feature with the Laplace library
            raise NotImplementedError()
        
    elif method=="load_file":
        hessian_name = cfg.projector.posterior_hessian.load.name
        hessian_file_name = os.path.join(
            path, cfg.projector.posterior_hessian.load.type, hessian_name
        )
        with open(hessian_file_name, "rb") as f:
            H_file = torch.load(f, map_location=cfg.device_torch)
        
        if hessian_file_name.startswith("Ihalf"):
            V = H_file["H"].to(dtype)
            return  HalfInvPsi(
                    V=V,
                    prior_precision=cfg.projector.sigma.prior_precision,
                )
        else:
            H = H_file["H"].to(dtype)
            assert H.size(0) == H.size(1), "Hessian must be squared matrix."
            inv_Psi = H \
                + cfg.projector.sigma.prior_precision \
                * torch.eye(H.size(0)).to(cfg.device_torch)
            return FullInvPsi(inv_Psi=inv_Psi)
        
    else:
        raise NotImplementedError


def get_P(        
        method: Literal["lowrank-ggnit", "lowrank-load_file" "lowrank-kron", 
                        "lowrank-full", "lowrank-diagonal", "subset-swag", 
                        "subset-magnitude", "subset-diagonal", "subset-custom"], 
        cfg: DictConfig, 
        model: nn.Module, 
        data_Psi: Dataset,
        data_J: Dataset, 
        path: str,
        s: Optional[int] = None,
    ) -> InvPsi:
    """
    Wrapper to get the linear operator `P`.

    Args:
        method: Method to compute `P`. 
            `lowrank-ggnit`, `lowrank-load_file`, `lowrank-kron`, 
            `lowrank-diagonal` and `lowrank-full` are methods to compute the 
            posterior to obtain the optimal linear operator
            `subset-swag`, `subset-magnitude`, `subset-diagonal` and 
            `subset-custom` select a certain set of weights to get `P` using the 
            Laplace lib
        cfg: Configurations file
        model: Pytorch model
        data_Psi: Pytorch Dataset to compute the posterior (if needed)
        data_J: Pytorch Dataset to compute the Jacobians (if needed)
        path: string to point to the file loaded by `lowrank-load_file`
        s: If given, the eigenvectors of the SVD cut at s vectors in the
        computation.
    """

    if method in ["lowrank-ggnit", "lowrank-load_file", "lowrank-kron", 
                  "lowrank-diagonal", "lowrank-full"]:
        def create_proj_jac_it():
            return create_jacobian_data_iterator(
                dataset=data_J,
                model=model,
                batch_size=cfg.projector.batch_size,
                number_of_batches=cfg.projector.n_batches,
                device=cfg.device_torch,
                dtype=getattr(torch, cfg.dtype),
                chunk_size=cfg.projector.chunk_size,
            )
        method = method.split("-")[1] # extract name for `get_IPsi`
        inv_Psi = get_IPsi(method, cfg, model, data_Psi, path)
        U = inv_Psi.Sigma_svd(create_proj_jac_it)[0]
        P = compute_optimal_P(IPsi=inv_Psi, J_X=create_proj_jac_it, U=U, s=s)
        return P
    
    elif method in ["subset-diagonal", "subset-magnitude", "subset-swag", 
                    "subset-custom"]:
        method = method.split("-")[1] # extract name for Laplace library 
        subset_kwargs = dict(cfg.data.swag_kwargs)
        likelihood = "classification" if cfg.data.is_classification \
            else "regression"
        dl = DataLoader(
            dataset=data_Psi,
            batch_size=cfg.projector.v.batch_size,
            shuffle=False
            )
        make_deterministic(cfg.seed)
        Ind = subset_indices(
                model=model,
                likelihood=likelihood,
                train_loader=dl,
                method=method,
                **subset_kwargs,
            )
        P = Ind.P(s=cfg.projector.s_max_regularized).to(cfg.device_torch)
        return P

    else:
        raise NotImplementedError

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