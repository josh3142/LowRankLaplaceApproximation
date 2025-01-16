#%%
import os

import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hydra 
from omegaconf import DictConfig 

from typing import Literal

def plot_log_value_histogram(
        path: str,
        values: np.ndarray,
        hessian: Literal["H", "I"]="I",
        eigtype: Literal["SingularValues", "EigenValues"]="SingularValues",
        bins: int=30
    ) -> None:
    """
    Plot a log-log histogram with transformed values `lambda`.

    The values are transformed as 
        ```
            sign(\lambda) * log_{10}(1 + |\lambda|)
        ```
    Args:
        path: Where to store the plot
        values: The values to by plotted
        eigtype: Which type of values should be plotted
        bins: number of bins to group the eigenvalues
    """
    if eigtype=="SingularValues":
        name="singular values"
    elif eigtype=="EigenValues":
        name="eigenvalues"
    
    log_values = np.sign(values) * np.log10(1 + np.abs(values))
    
    plt.figure(figsize=(8, 6))
    plt.title(f'Histogram of {name} of {hessian} on logarithmic scale')
    counts, bins, patches = plt.hist(
        log_values, bins=bins, density=False, alpha=0.5, color='b', label=name)
    
    # make a count for each bin above the bar
    for count, patch in zip(counts, patches):
        # Positioning text in the center of the bar
        plt.text(patch.get_x() + patch.get_width() / 2, count, str(int(count)), 
                 ha='center', va='bottom', fontsize=10, color='black', 
                 rotation=35)
    
    plt.yscale('log')
    plt.ylim(0.5, max(counts) * 2)
    plt.xlabel(r'$sign(\lambda) * \log_{10}(1 + |\lambda|)$ with ' +
               r'eigenvalues $\lambda$')
    plt.ylabel("Log-Scaled counts")
    plt.savefig(path)    
    plt.close()


@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    # torch.set_float32_matmul_precision("high") 
    if cfg.dtype=="float64":
        torch.set_default_dtype(torch.float64)
    elif cfg.dtype=="float32":
        torch.set_default_dtype(torch.float64)

    print(cfg)

    path = f"results/{cfg.data.folder_name}/{cfg.pred_model.name}/seed{cfg.seed}"
    path_projector = os.path.join(path, f"projector/{cfg.projector.name}")
  
    Hdict = torch.load(os.path.join(
        path_projector, 
        f"{cfg.projector.name}{cfg.data.projector.n_sample}.pt"),
        map_location=cfg.device_torch,
        weights_only=False
    )
    H, n_sample = Hdict["H"], Hdict["n_samples"]
    assert n_sample==cfg.data.projector.n_sample, \
        "The projector is computed with the wrong number of samples."

    # Obtain the spectrum
    try:
        # Load spectrum
        spectrum = torch.load(os.path.join(
            path_projector,
            f"{cfg.projector.name}{n_sample}Spectrum.pt"),
            map_location="cpu",
            weights_only=False
        )
        sing_values, eig_values = spectrum["SingValues"], spectrum["EigValues"]
        print("spectrum is loaded")
    except:
        # compute spectrum
        sing_values = torch.svd(H, compute_uv=False).S.cpu()
        eig_values = torch.linalg.eigvalsh(H).sort(descending=True).values.cpu()
        torch.save(
            {"SingValues": sing_values,
            "EigValues": eig_values,
            "n_samples": n_sample, 
            "hessian_type": cfg.projector.name},
            os.path.join(path_projector, f"{cfg.projector.name}{n_sample}Spectrum.pt")
        )

    # store values
    df = pd.DataFrame(data={
            "SingularValues": sing_values,
            "EigenValues": eig_values
        },
        index=None
    )
    df.to_csv(os.path.join(
        path_projector, f"{cfg.projector.name}{n_sample}Spectrum.csv")
    )

    # plot spectrum
    plot_log_value_histogram(
        path=os.path.join(path_projector, f"{cfg.projector.name}{n_sample}SingValues.png"),
        values=sing_values,
        hessian=cfg.projector.name,
        eigtype="SingularValues",
        bins=30
    )
    plot_log_value_histogram(
        path=os.path.join(path_projector, f"{cfg.projector.name}{n_sample}EigValues.png"),
        values=eig_values,
        hessian=cfg.projector.name,
        eigtype="EigenValues",
        bins=30
    )

if __name__=="__main__":
    run_main()
