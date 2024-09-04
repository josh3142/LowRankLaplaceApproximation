import os 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

import hydra 
from omegaconf import DictConfig 
from pathlib import Path

from pred_model.model import get_model
from data.dataset import get_dataset
from projector.hessian import get_H_sum
from projector.projector import get_hessian_type_fun

from utils import make_deterministic


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
    print(cfg)

    # initialize dataset and dataloader
    dataset = get_dataset(cfg.data.name, cfg.data.path, train=True) 
    # dataset = Subset(dataset, indices=np.arange(100)) 
    dl = DataLoader(
        dataset, 
        shuffle=False,
        batch_size=cfg.projector.batch_size
    )

    model = get_model(cfg.pred_model.name, 
        **(dict(cfg.pred_model.param) | dict(cfg.data.param)))
    state_dict = torch.load(
        os.path.join(path_model, cfg.data.model.ckpt),
        map_location="cpu"
    )
    model.load_state_dict(state_dict)
    model.to(cfg.device_torch)

    with torch.no_grad():
        # compute Hessian/FI
        H = get_hessian_type_fun(cfg.projector.name)(
            model=model,
            dl=dl,
            is_classification=cfg.data.is_classification,
            n_batches=cfg.projector.n_batches,
            chunk_size=cfg.projector.chunk_size
        )
        if cfg.projector.n_batches is None:
            n_sample = len(dataset)
        else:
            n_sample = min(len(dataset), 
                           cfg.projector.batch_size * cfg.projector.n_batches)
        print("Shape of (approximated) Hessian: ", H.shape)
        print("Samples to compute Hessian: ", n_sample )
        
        # sanity check
        print("\nDisplay few predictions as a sanity check.")
        X, Y = next(iter(dl))
        dtype = next(model.parameters()).dtype
        print("Predictions: ", model(X.to(cfg.device_torch).to(dtype)))
        print("True values: ", Y)

    # save Hessian
    torch.save(
        {"H": H,
         "n_samples": n_sample, 
         "hessian_type": cfg.projector.name},
        os.path.join(path_results_i, f"{cfg.projector.name}{n_sample}.pt")
    )


if __name__=="__main__":
    run_main()