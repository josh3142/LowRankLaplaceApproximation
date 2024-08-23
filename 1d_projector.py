import os 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

import cupy as cp
from cupyx.scipy.sparse.linalg import LinearOperator, gmres

from functools import partial
import hydra 
from omegaconf import DictConfig 

import pandas as pd
from pathlib import Path
from time import time
from typing import List

from pred_model.model import get_model
from data.dataset import get_dataset
from projector.projector1d import (get_jacobian, get_least_square_error,
    get_lhs_linear_equ_of_1d_projector, get_sigma_1d, get_sigma_projected_1d,
    get_Vs, get_pred_var, get_inv)

from utils_df import save_df
from utils import make_deterministic


def create_df(
        idcs: List | np.ndarray, 
        idcs_c: List | np.ndarray, 
        obj: np.ndarray, 
        obj_proj: np.ndarray, 
        error: np.ndarray, 
        time: np.ndarray
    ) -> pd.DataFrame:
    """
    Args:
        idcs: Indices of the samples
        idcs_c: Class indices of the samples
        obj: Full objective
        obj_proj: Objective computed with projector
        error: Error in ||Ax - b||_2
        time: Time to compute obj_approx
    """
    df = pd.DataFrame(data = {
            "idx": idcs,
            "idx_c": idcs_c,
            "Sigma": obj,
            "Sigma_p": obj_proj,
            "Delta": np.round(obj - obj_proj, 4),
            "Sigma_p_error": np.round(error, 4),
            "time": np.round(time, 4)
        }
    ) 
    return df


@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:
    make_deterministic(cfg.seed)
    torch.set_num_threads(16)

    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    # torch.set_float32_matmul_precision("high") 
    torch.set_default_dtype(torch.float64)

    path = f"results/{cfg.data.name}/{cfg.pred_model.name}"
    path_model = os.path.join(path, "ckpt")
    path_projector = os.path.join(path, f"projector/{cfg.projector.name}")
    path_results_i = os.path.join(path_projector)
    Path(path_results_i).mkdir(parents=True, exist_ok=True)
    Path(path_projector).mkdir(parents=True, exist_ok=True)
    print((cfg))

    # initialize dataset and dataloader
    dataset = get_dataset(cfg.data.name, cfg.data.path, train=True)   
    dl = DataLoader(
        dataset, 
        shuffle=False,
        batch_size=len(dataset)
    )
    X, Y = next(iter(dl))
    X, Y = X.to(cfg.device_torch), Y.to(cfg.device_torch)
    X = X.to(torch.float64)

    dl_projector = DataLoader(
        dataset, 
        shuffle=False,
        batch_size=cfg.projector.batch_size
    )

    datasetc = get_dataset(cfg.data.name_corrupt, cfg.data.path, train=False)
    dlc = DataLoader(
        datasetc, 
        shuffle=False,
        batch_size=len(datasetc)
    )
    Xc, Yc = next(iter(dlc))
    Xc, Yc = Xc.to(cfg.device_torch), Yc.to(cfg.device_torch)
    Xc = Xc.to(torch.float64)

    model = get_model(cfg.pred_model.name, 
        **(dict(cfg.pred_model.param) | dict(cfg.data.param)))
    state_dict = torch.load(
        os.path.join(path_model, cfg.data.model.ckpt),
        map_location="cpu"
    )
    model.load_state_dict(state_dict)
    model.to(cfg.device_torch)

    with torch.no_grad():
        # try/except statements to prevent memory overflow 
        # chunk_size argument differs only
        # compute jacobian of test sample
        try:
            J = get_jacobian(
                model, 
                Xc, 
                fun=lambda x: x, 
                is_classification=cfg.data.is_classification,
                chunk_size=None
            ).detach()
        except:
            J = get_jacobian(
                model, 
                Xc, 
                fun=lambda x: x, 
                is_classification=cfg.data.is_classification,
                chunk_size=1
            ).detach()
        torch.cuda.empty_cache()
        print("Shape of J [sample, class, parameters]: ", J.shape)
        
        # compute predictive covariance
        V = get_Vs(
            model=model,
            dl=dl_projector,
            n_batches=cfg.projector.n_batches,
            chunk_size=cfg.projector.chunk_size
        ).detach()
        print("Shape of score: ", V.shape)
        v = V.cpu()
        del V
        torch.cuda.empty_cache()
    
    # define linear operator
    p_dim = v.shape[0]
    # other algorithm that computes scores on the fly
    # op = LinearOperator(
    #     shape=(p_dim, p_dim),
    #     matvec=partial(
    #         get_lhs_linear_equ_of_1d_projector,
    #         model=model,
    #         dl=dl_projector,
    #         n_batches=cfg.projector.n_batches,
    #         chunk_size=cfg.projector.chunk_size)
    #     )
    
    op = LinearOperator(
        shape=(p_dim, p_dim),
        matvec=partial(
            get_lhs_linear_equ_of_1d_projector,
            Vs=v,
            batch_size=cfg.projector.batch_size
        )
    )
    # compute optimal projector
    if cfg.projector.rounds is None:
        rounds = len(datasetc)-1
    else:
        rounds = cfg.projector.rounds
    objs, obj_projs, errors, t_deltas, ps, = [], [], [], [], []
    idcs, idcs_c = [], []
    for idx in range(rounds):
        ps_c = []
        for idx_c in range(J.shape[1]):
            j = J[idx, idx_c].to(cfg.device_torch)
            
            # obtain optimal p
            t1 = time()
            p_sol_cp, info = gmres(
                A=op, 
                b=cp.from_dlpack(j), 
                tol=1e-5,
                maxiter=cfg.projector.maxiter)
            t_deltas.append(np.round(time() - t1, 2))
            p_sol = torch.from_dlpack(p_sol_cp)
            torch.cuda.empty_cache()

            # obtain Sigma and Sigma_p
            V = v.to(cfg.device_torch)
            obj_full = np.round(get_sigma_1d(j, V).item(), 3)
            obj_proj = np.round(get_sigma_projected_1d(p_sol, j, V).item(), 3)
            print(idx, obj_proj, obj_full)

            objs.append(obj_full)
            obj_projs.append(obj_proj)
            # errors.append(np.round(get_least_square_error(j, p_sol, model, dl_projector, chunk_size), 4))
            errors.append(
                np.round(
                    get_least_square_error(
                        j, p_sol,V, cfg.projector.batch_size, cfg.device_torch
                    ), 
                4)
            )
            ps_c.append(p_sol.cpu())
            idcs.append(idx)
            idcs_c.append(idx_c)
            del V
            torch.cuda.empty_cache()
        ps.append(torch.stack(ps_c))

    objs = np.array(objs)
    obj_projs = np.array(obj_projs)
    t_deltas = np.array(t_deltas)
    errors = np.array(errors)

    # save df
    df = create_df(idcs, idcs_c, objs, obj_projs, errors, t_deltas)
    save_df(df, os.path.join(path_results_i, f"Sigma.csv"))

    # save ps
    ps = torch.stack(ps, dim=0)
    print(ps.shape)
    torch.save(
        {"P": ps}, 
        os.path.join(path_results_i, cfg.projector.name + ".pt")
    )


if __name__=="__main__":
    run_main()