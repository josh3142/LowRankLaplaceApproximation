import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob

import hydra 
from omegaconf import DictConfig 

from typing import Tuple

from color_map import get_color, get_color2

p_names = {
    'kron': "KFAC",
    'magnitude': "Magnitude",
    'swag': "SWAG",
    'diagonal': "Diagonal",
    'None': "Full",
}

psi_names ={
    "Psiggnit": "GGN",
    "Psikron": "KFAC",
    "Psifull": "GGN",
    "Psiload_file": "loaded"
}

def get_p_name(method: str) -> str:
    if not method in p_names.keys():
        raise NotImplementedError
    return p_names[method]

def get_psi_name(method: str) -> str:
    if not method in psi_names.keys():
        raise NotImplementedError
    return psi_names[method]

def get_linestyle(group: str) -> str:
    if group=="Plowrank":
        return "-"
    elif group=="Psubset":
        return "--"
    elif group=="PNone":
        return ":"
    else:
        raise NotImplementedError
    
def get_p_line_formating(method: str) -> Tuple[str, str]:
    # The splits are such that from the df column name the appropriate 
    # legend and linestyle is extracted.
    if not method=="PNone": 
        group, name = method.split("-")
    else:
        group, name = "PNone", "None"
    return group, name

def get_title(name: str) -> str:
    if name=="error":
        return r"Relative Error between $P_{full}$ and $P_{method}$"
    elif name=="nll":
        return "Negative Log-Likelihood"
    elif name=="trace":
        return "Trace of Covariance Matrix $\Sigma$"
    elif name=="logtrace":
        return "Logarithm of Trace of Covariance Matrix $\Sigma$"
    else:
        raise NotImplementedError
    
def get_ylabel(name: str) -> str:
    if name=="error":
        return r"relative error"
    elif name=="nll":
        return "NLL"
    elif name=="trace":
        return "Trace"
    elif name=="logtrace":
        return "Log-Trace"
    else:
        raise NotImplementedError



def get_plt(df: pd.DataFrame) -> None:
    n_seed = len(df.seed.unique())
    df_mean = df.groupby(["dim"], as_index = False).mean()
    df_std = df.groupby(["dim"], as_index = False).std(ddof=1) / np.sqrt(n_seed)

    methods = set(df.columns)
    methods.remove("seed")
    methods.remove("dim")
    for method in methods:
        p_setting, psi_setting = method.split("_")
        psi_name_plot = get_psi_name(psi_setting)
        p_group, p_name = get_p_line_formating(p_setting)
        p_name_plot = get_p_name(p_name)

        plt.plot(df_mean["dim"], 
                 df_mean[f"{method}"], 
                 marker="x",
                 color=get_color(p_name) if psi_name_plot=="GGN" else get_color2(p_name), 
                 label=fr"$P_{{{p_name_plot}}}-\Psi_{{{psi_name_plot}}}$", 
                 alpha= 1.,
                 linestyle=get_linestyle(p_group))
        plt.fill_between(df_mean["dim"], 
                        df_mean[f"{method}"] - df_std[f"{method}"],
                        df_mean[f"{method}"] + df_std[f"{method}"],
                        color = get_color(p_name) if psi_name_plot=="GGN" else get_color2(p_name),
                        alpha = 0.15)


def save_fig_acc(df: pd.DataFrame, img_name: str, title: str, y_label: str
    ) -> None:
    color="black"

    plt.clf()
    plt.tight_layout()
    get_plt(df)
      
    if title is not None:
        plt.title(title)
    plt.xlabel(r"$s$", color=color)
    plt.ylabel(f"{y_label}", color=color)
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
        save_fig_acc(
            df, img_name, 
            title=get_title(name), 
            y_label=get_ylabel(name)
        )


if __name__ == "__main__":
    run_main()
