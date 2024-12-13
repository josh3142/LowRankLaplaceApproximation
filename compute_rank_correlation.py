import os
import glob
import hydra
from omegaconf import DictConfig
import torch
import pandas
from scipy.stats import kendalltau


@hydra.main(config_path="config", config_name="config")
def run_main(cfg: DictConfig) -> None:

    # setting up paths
    print(f"Considering {cfg.data.name}")
    path = f"results/{cfg.data.name}/{cfg.pred_model.name}"
    file_names = glob(os.path.join(path,  "*.csv"))

    for file_name in file_names:
        name = os.path.split(file_name)[-1]
        name = name.split(".")[0]
        name = name.split("_")[-1]

    



if __name__ == "__main__":
    run_main()