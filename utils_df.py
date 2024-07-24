import numpy as np
import random
import torch

import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional
import PIL.Image


def make_deterministic(seed) -> None:
    random.seed(seed)   	    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only = True)

# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
def plot(imgs: PIL.Image,  title: Optional[str] = None, 
    row_title: Optional[List] = None, col_title: Optional[List] = None,
    cmap: str = "viridis",
    save: Optional[str] = None, close: bool = True, **imshow_kwargs) -> None:
    
    plt.clf()
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(figsize = (10, 10), 
        nrows = num_rows, ncols = num_cols, squeeze = False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), cmap = cmap, **imshow_kwargs)
            ax.set(xticklabels = [], yticklabels = [], xticks = [], yticks = [])

    if title is not None:
        fig.suptitle(title, fontsize = 40, y = 1)

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel = row_title[row_idx])

    if col_title is not None:
        for col_idx in range(num_cols):
            axs[0, col_idx].set(title = col_title[col_idx])

    plt.tight_layout()

    if save is not None:
        fig.savefig(save)

    if close:
        plt.close("all")


def create_df(objective: List[float], lr: List[float], k: int = 100 
    ) -> pd.DataFrame:
    """
    Args:
        k: store every kth step of optimization
    """
    df = pd.DataFrame(data =
            {"step": [int(i) * k for i in range(len(objective))],
            "objective": objective,
            "lr": lr
            }
        )
    
    return df
    
def save_df(df: pd.DataFrame, filename: str, 
    is_file_overwritten: bool=True) -> None:
    
    if not is_file_overwritten:
        try:
            df_loaded = pd.read_csv(filename)
            df = pd.concat([df_loaded, df], axis = 0)
        except FileNotFoundError:
            print("File does not exists.")
            print("A new file is created")
    df.to_csv(filename, index = False)

def df_covariance(sigma: np.ndarray) -> pd.DataFrame:
    assert len(sigma.shape) == 2, "sigma has to be a matrix"
    df = pd.DataFrame(
        data=sigma, 
        index=range(sigma.shape[0]), 
        columns=range(sigma.shape[1])
    )
    return df 
