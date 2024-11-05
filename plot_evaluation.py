import os

import pandas as pd
import matplotlib.pyplot as plt

from glob import glob

import hydra 
from omegaconf import DictConfig 

def get_plt(df: pd.DataFrame, fmt: str) -> None:
    """ 
    Args:
        df: `DataFrame`
        fmt: Set marker specification, e.g. fmt = [marker][line][color]. See
          https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
        label: label of plotted data.
    """
    df_mean = df.groupby(["dim"], as_index = False).mean()
    df_std = df.groupby(["dim"], as_index = False).std(ddof = 1)

    methods = set(df.columns)
    methods.remove("seed")
    methods.remove("dim")
    for method in methods:
        plt.plot(df_mean["dim"], df_mean[f"{method}"], label=method)
        plt.fill_between(df_mean["dim"], 
                        df_mean[f"{method}"] - df_std[f"{method}"],
                        df_mean[f"{method}"] + df_std[f"{method}"],
                        #color = fmt[0],
                        alpha = 0.15)


def save_fig_acc(df: pd.DataFrame, img_name: str, title: str, **kwags) -> None:
    color="black"

    plt.clf()
    plt.tight_layout()
    get_plt(df, fmt = "m+:")
      
    if title is not None:
        plt.title(title)
    plt.xlabel(r"$s$", color=color)
    plt.ylabel(f"{title}", color=color)
    plt.yticks(color=color)
    plt.xticks(color=color)
    plt.gca().spines[["top", "bottom", "right", "left"]].set_color(color)
    plt.legend()

    plt.savefig(img_name)
    # plt.savefig(img_name, transparent = True, bbox_inches = "tight") # for presentation 


@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:

    path = f"results/{cfg.data.name}/{cfg.pred_model.name}"
    file_names = glob(os.path.join(path, "*.csv"))

    for file_name in file_names:
        name = os.path.split(file_name)[-1]
        name = name.split(".")[0]
        name = name.split("_")[-1]
        df = pd.read_csv(file_name, index_col=False)
    
        # Create plot
        img_name = os.path.join(path, name)
        title = name
        save_fig_acc(df, img_name, title)


if __name__ == "__main__":
    run_main()
