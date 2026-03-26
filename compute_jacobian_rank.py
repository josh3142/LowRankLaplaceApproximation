"""Prints out some basic information on the specified dataset.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import os

import hydra
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from projector.projector1d import create_jacobian_data_iterator
from data.dataset import get_dataset
from pred_model.model import get_model

from utils import make_deterministic


@hydra.main(config_path="config", config_name="config")
def run_main(cfg: DictConfig) -> None:
    torch.set_default_dtype(getattr(torch, cfg.dtype))
    seed_range = getattr(cfg, 'seed_range', [1,2,3,4,5])
    rank_collection = []

    # loading data
    data_name = cfg.data.name_corrupt if cfg.data.use_corrupt else cfg.data.name
    get_dataset_kwargs = dict(
        name=data_name, path=cfg.data.path, dtype=cfg.dtype
        )

    # load data and construct DataLoader
    train_data = get_dataset(**get_dataset_kwargs, train=True)

    # load network
    get_model_kwargs = dict(cfg.pred_model.param) | dict(cfg.data.param)
    get_model_kwargs["name"] = cfg.pred_model.name
    model = get_model(**get_model_kwargs)
    model.eval()
    model.to(cfg.device_torch)
    # switch off layers to ignore
    for module in model.modules():
        if type(module).__name__ in cfg.projector.layers_to_ignore:
            for par in module.parameters():
                par.requires_grad = False

    for seed in tqdm(seed_range):

        make_deterministic(seed)

        # setting up paths
        results_path = os.path.join(
            "results", cfg.data.folder_name, cfg.pred_model.name, f"seed{seed}"
        )

        # load checkpoint
        ckpt_file_name = os.path.join(
            results_path,
            "ckpt",
            cfg.data.model.ckpt
        )
        with open(ckpt_file_name, "rb") as f:
            state_dict = torch.load(f, map_location=cfg.device_torch,
                                    weights_only=True)
        model.load_state_dict(state_dict=state_dict)

        #  The following objects create upon call an iterator over the jacobian
        def create_train_jac_it():
            return create_jacobian_data_iterator(
                dataset=train_data,
                model=model,
                batch_size=cfg.projector.batch_size,
                number_of_batches=cfg.projector.n_batches,
                device=cfg.device_torch,
                dtype=getattr(torch, cfg.dtype),
                jacobian_order_seed=cfg.projector.jacobian_seed,
                chunk_size=cfg.projector.chunk_size,
            )

        # rank J_X
        J_X = torch.concat([j for j in create_train_jac_it()], dim=0)
        rank_collection.append(torch.linalg.matrix_rank(J_X).item())
    # turn list into a tensor
    rank_collection = torch.tensor(rank_collection)
    print(f'Range of ranks: [{rank_collection.min().item()},{rank_collection.max().item()}]')


if __name__ == "__main__":
    run_main()
