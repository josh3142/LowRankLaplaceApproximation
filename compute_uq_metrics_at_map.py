
"""Compute the uncertainty quantification metrics at the MAP 
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import os
import math

import hydra
from omegaconf import DictConfig, open_dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm

from utils import estimate_regression_likelihood_sigma
from data.dataset import get_dataset
from pred_model.model import get_model
from linearized_model.approximation_metrics import ECE

from utils import make_deterministic



def get_regression_likelihood_sigma(model, dl, classification, device):
    """ Compute the sigma for non-classification problems. """
    if not classification:
        print('Estimating sigma of likelihood')
        regression_likelihood_sigma = estimate_regression_likelihood_sigma(
            model=model, dataloader=dl, device=device,
        )
    else:
        regression_likelihood_sigma = None
    return regression_likelihood_sigma

@hydra.main(config_path="config", config_name="config")
def run_main(cfg: DictConfig) -> None:
    make_deterministic(cfg.seed)
    torch.set_default_dtype(getattr(torch, cfg.dtype))


    # setting up paths
    print(f"Considering {cfg.data.name}")

    results_path = os.path.join(
        "results", cfg.data.folder_name, cfg.pred_model.name, f"seed{cfg.seed}"
    )

    ## for the results
    results_filename = f"MAP_UQ{cfg.results_file.name_postfix}.pt"
    results_file = os.path.join(results_path, results_filename)

    
    # setting up kwargs for loading of model and data
    get_model_kwargs = dict(cfg.pred_model.param) | dict(cfg.data.param)
    get_model_kwargs["name"] = cfg.pred_model.name

    data_name = cfg.data.name_corrupt if cfg.data.use_corrupt else cfg.data.name
    get_dataset_kwargs = dict(
        name=data_name, path=cfg.data.path, dtype=cfg.dtype
        )
    print(f'Using data {data_name}')

    # load data and construct DataLoader
    train_data = get_dataset(**get_dataset_kwargs, train=True)
    test_data = get_dataset(**get_dataset_kwargs, train=False)
    dl_train = DataLoader(
        dataset=train_data,
        batch_size=cfg.projector.fit.batch_size,
        shuffle=False
    )
    dl_test = DataLoader(
        dataset=test_data,
        batch_size=cfg.projector.fit.batch_size,
        shuffle=False
    )

    # load network
    model = get_model(**get_model_kwargs)
    model.eval()
    model.to(cfg.device_torch)

    # load checkpoint
    ckpt_file_name = os.path.join(results_path, "ckpt", cfg.data.model.ckpt)
    with open(ckpt_file_name, "rb") as f:
        state_dict = torch.load(f, map_location=cfg.device_torch, 
                                weights_only=True)
    model.load_state_dict(state_dict=state_dict)



    # get regression likelihood sigma
    if cfg.projector.data_std is None:
        regression_likelihood_sigma = get_regression_likelihood_sigma(
                model, dl_train, cfg.data.is_classification, cfg.device_torch
            )
    elif cfg.projector.data_std > 0:
        regression_likelihood_sigma = cfg.projector.data_std
    else:
        raise ValueError("Insert a None or a positive number for `data_std`.")

    # compute NLL at MAP
    logit_collection = []
    target_collection = []
    for x,y in tqdm(dl_test):
        x, y = x.to(cfg.device_torch), y.to(cfg.device_torch)
        with torch.no_grad():
            outputs = model(x)
            logit_collection.append(outputs.cpu())
            target_collection.append(y.cpu())

    logits = torch.cat(logit_collection, dim=0)
    targets = torch.cat(target_collection, dim=0)

    # compute nll
    if cfg.data.is_classification:
        nll = nn.CrossEntropyLoss(reduction='mean')(logits, targets)
        brier_score = torch.mean(
            torch.mean(
                (torch.nn.functional.one_hot(targets, num_classes=logits.shape[1]) - 
                torch.nn.functional.softmax(logits, dim=1))**2,
                dim=1
            )
        ).item()
        ece = ECE(post_pred = torch.nn.functional.softmax(logits, dim=1), y=targets)
    else:
        nll = (
            0.5 * math.log(2 * math.pi)
            + torch.log(regression_likelihood_sigma)
            + 0.5 * torch.mean((targets - logits.squeeze())**2) 
              / (regression_likelihood_sigma**2)
        )
        brier_score = None
        ece = None

    uq_results = {
        'nll': nll.item(),
        'brier': brier_score,
        'ece': ece,
    }

    print(
        f"NLL at MAP: {uq_results['nll']}, \n "
        f"Brier Score at MAP: {uq_results['brier']},  \n"
        f"ECE at MAP: {uq_results['ece']}"
    )
    with open(results_file, "wb") as f:
        torch.save(uq_results, f)





if __name__ == "__main__":
    run_main()
