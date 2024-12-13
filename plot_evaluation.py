import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from glob import glob

import hydra 
from omegaconf import DictConfig 

from typing import Tuple, Optional, List

from color_map import get_color, get_color2


p_names = {
    'kron': "KFAC",
    'magnitude': "Magnitude",
    'swag': "SWAG",
    'diag': "Diagonal",
    'None': "Full",
    'ggnit': "GGN"
}

psi_names ={
    "Psiggnit": "GGN",
    "Psikron": "KFAC",
    "Psidiag": "Diagonal",
    "Psifull": "GGN",
    "Psiloadfile": "loaded"
}

p_method = {
    "Plowrank": "lowrank",
    "Psubset": "subset",
    "PNone": "None",
    "Plowrankoptimal": "lowrankopt"
}

def get_p_name(method: str) -> str:
    if not method in p_names.keys():
        raise NotImplementedError
    return p_names[method]

def get_psi_name(method: str) -> str:
    if not method in psi_names.keys():
        raise NotImplementedError
    return psi_names[method]

def get_p_method(method: str) -> str:
    if not method in p_method.keys():
        raise NotImplementedError
    return p_method[method]

def get_linestyle(group: str) -> str:
    if group=="Plowrank":
        return "-"
    elif group=="Psubset":
        return "--"
    elif group=="PNone":
        return ":"
    elif group=="Plowrankoptimal":
        return "-."
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

def get_ylabel(name: str) -> str:
    if name=="error":
        return "Relative Error"
    elif name=="nll":
        return "NLL"
    elif name=="trace":
        return "Trace"
    elif name=="logtrace":
        return "Log-Trace"
    else:
        raise NotImplementedError
    
def get_plot_settings(name: str) -> str:
    if name=="error":
        loc, ylim = "upper right", (-0.05, 1.05)
    elif name=="nll":
        loc, ylim = "lower right", None
    elif name=="trace":
        loc, ylim = "lower right", (-2, None)
    elif name=="logtrace":
        loc, ylim = "lower right", (-1, None)
    else:
        raise NotImplementedError
    return loc, ylim


def get_plt(
        df: pd.DataFrame, 
        methods: Optional[List[str]]=None,
        is_label_method: bool=True, 
        is_label_psi: bool=False
    ) -> None:
    n_seed = len(df.seed.unique())
    df_mean = df.groupby(["dim"], as_index = False).mean()
    df_std = df.groupby(["dim"], as_index = False).std(ddof=1) / np.sqrt(n_seed)

    if methods is None:
        methods = set(df.columns)
        methods.remove("seed")
        methods.remove("dim")
    for method in methods:
        p_setting, psi_setting = method.split("_")
        psi_name_plot = get_psi_name(psi_setting)
        p_group, p_name = get_p_line_formating(p_setting)
        p_name_plot = get_p_name(p_name)

        # generate label
        label_method = f"{get_p_method(p_group)}-" if is_label_method else ""
        label_psi = fr"$-\Psi_{{{psi_name_plot}}}$" if is_label_psi else ""
        label = fr"$P_{{{label_method}{p_name_plot}}}${label_psi}"
        
        plt.plot(df_mean["dim"], 
                 df_mean[f"{method}"], 
                 marker="x",
                 color=get_color(p_name) if psi_name_plot=="GGN" else get_color2(p_name), 
                 label=label, 
                 alpha= 1.,
                 linestyle=get_linestyle(p_group))
        plt.fill_between(df_mean["dim"], 
                        df_mean[f"{method}"] - df_std[f"{method}"],
                        df_mean[f"{method}"] + df_std[f"{method}"],
                        color = get_color(p_name) if psi_name_plot=="GGN" else get_color2(p_name),
                        alpha = 0.15)


def save_fig_acc(
        df: pd.DataFrame, 
        img_name: str, 
        is_label_method: bool,
        is_label_psi: bool,
        title: str, 
        y_label: str,
        ylim: Tuple, 
        loc: str, 
        cfg: DictConfig,
        methods=Optional[List[str]]
    ) -> None:
    
    color="black"

    plt.clf()
    plt.tight_layout()
    get_plt(df, methods, is_label_method, is_label_psi)
      
    plt.xlabel(r"$s$", color=color)
    plt.ylabel(f"{y_label}", color=color)
    plt.yticks(color=color)
    plt.xticks(color=color)
    plt.gca().spines[["top", "bottom", "right", "left"]].set_color(color)
    if cfg.plot.show_legend:
        plt.legend(loc=loc, fontsize=cfg.plot.fontsize.legend)
    plt.ylim(ylim)
    plt.savefig(img_name, bbox_inches='tight')
    # plt.savefig(img_name, transparent = True, bbox_inches = "tight") # for presentation 


@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:
    mpl.rcParams['font.size'] = cfg.plot.fontsize.axes  # Set global font size

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
            df, 
            img_name, 
            is_label_method=cfg.plot.evaluation.legend.reduction_type,
            is_label_psi=cfg.plot.evaluation.legend.psi,
            title=get_title(name), 
            y_label=get_ylabel(name),
            loc=get_plot_settings(name)[0],
            ylim=get_plot_settings(name)[1],
            methods=cfg.plot.evaluation.methods,
            cfg=cfg
        )


if __name__ == "__main__":
    run_main()
