#%%
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from hydra import compose, initialize

from data.dataset import get_dataset
from pred_model.model import get_model
from utils_df import plot

with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name="config", 
                  overrides=[
                      "data=mnist",
                      "device_torch=cuda:0",
                      "projector.model.ckpt=epoch29-step7050_dict.ckpt",
                      "pred_model=mlp_small",
                      "pred_model.param.n_hidden=256", 
                      "pred_model.param.n_layer=2"
                      ])
    
path = "results/mnist/mlp_small"
path_model = os.path.join(path, "ckpt")

#%%
# initialize dataset and dataloader
dataset = get_dataset(cfg.data.name, cfg.data.path, train=True)
dl = DataLoader(
    dataset, 
    shuffle=False,
    batch_size=len(dataset)
)
X = next(iter(dl))[0].numpy()

data_c = f"{cfg.data.name}_corrupt"
datasetc = get_dataset(data_c, cfg.data.path, train=False)
dlc = DataLoader(
    datasetc, 
    shuffle=False,
    batch_size=len(dataset)
)
Xc = next(iter(dlc))[0].numpy()

#%%
idx = 1395
img_n, img_c = np.transpose(X[idx], (1, 2, 0)) , np.transpose(Xc[idx], (1, 2, 0))

plot(
    imgs=[img_n, img_c],
    col_title=["normal image", "corrupted image"],
    #save=os.path.join(path_results_i, f"Sample{idx}.png"), 
    close=False
)
# %%
# initialize pyro model and update MLE parameters
model = get_model(cfg.pred_model.name, 
    **(dict(cfg.pred_model.param) | dict(cfg.data.param)))
state_dict = torch.load(
    os.path.join(path_model, cfg.projector.model.ckpt),
    map_location="cpu"
)
model.load_state_dict(state_dict)
model.to(cfg.device_torch)

# %%
with torch.no_grad():
    print("Non-corrupted image: ", 
        torch.round(nn.functional.softmax(
            model(torch.tensor(X[idx], device=cfg.device_torch)), dim=-1),
            decimals=4))
    print("Corrupted image: ", 
        torch.round(nn.functional.softmax(
            model(torch.tensor(Xc[idx], device=cfg.device_torch)), dim=-1),
            decimals=4))
# %%
