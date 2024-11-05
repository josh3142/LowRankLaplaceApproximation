import os
import sys

import torch
import pandas as pd
from glob import glob
import re
from collections import defaultdict

import hydra 
from omegaconf import DictConfig 

from typing import List

from utils_df import save_df


def create_evaluation_df(
        seed: int,
        s: List[int]
    ) -> pd.DataFrame:

    df = pd.DataFrame(data = {"seed": [seed] * len(s), "dim": s})
    
    return df

def are_all_s_list_the_same(file_names: List[str]) -> bool:
    "Check if all s_list in the dicitonaries are the same."
    for file_name in file_names:
        file=torch.load(file_name)
        try:
            if not s_list==file["s_list"]:            
                return False
            
        except NameError:
            s_list = file_name
        
        return True

def filter_sort_file_names_by_seed(file_names: List[str]) -> List[List[str]]:
    """
    Filters and sorts the `file_names` by **/seed{number}/**  

    Return: List of List where each sublist contains the files of one seed 
        subfolder
    """
    def extract_section(file_names):
        match = re.search(r'/s(\d+)/', file_names)
        # Convert the section number to an integer for numerical sorting
        return int(match.group(1)) if match else None 
    
    # Filter out unmatched filenames, then sort by seed
    filt_file_names = [f for f in file_names if extract_section(f) is not None]
    sorted_file_names = sorted(filt_file_names, key=extract_section)

    # Group by seed
    grouped_filenames = defaultdict(list)
    for filename in sorted_file_names:
        section = extract_section(filename)
        grouped_filenames[section].append(filename)
    
    # Convert to a list of lists and sort by seed
    grouped_and_sorted_lists = [grouped_filenames[section] 
                                for section in sorted(grouped_filenames)]

    return grouped_and_sorted_lists

@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:

    # open all files with suffix ".pt" in one-level subfolder
    # (this subfolders should be the different seeds, e.g. "seed1")
    path = f"results/{cfg.data.name}/{cfg.pred_model.name}"
    file_names = glob(os.path.join(path, "*/", "*.pt"))
    nll, rel_error, trace = {"name": "nll"}, {"name": "rel_error"}, {"name": "trace"}
    seeds = set()
    if not are_all_s_list_the_same(file_names):
        sys.exit("Not all s_list are the same. Please have only " + \
                "dictionaries where all s_list are the same in the folder.")


    # for file_names in file_names_sorted:
    for file_name in file_names:
        path_seed, file_name_without_path = os.path.split(file_name)
        seed = int(path_seed.split("/")[-1][-1])

        # works only for specific file name set by compute_metric.py or 
        # compare compare_laplace_approximations_modularize.py
        if len(file_name_without_path.split("_"))==3:
            dic_name, p_approx, psi_approx = file_name_without_path.split("_")
            psi_approx, _ = psi_approx.split(".")
            file = torch.load(file_name, weights_only=False)
            name = f"P{p_approx}_{psi_approx}"

            if dic_name == "Metrics":
                rel_error.setdefault(seed, {}).setdefault(name, file["rel_error"])
                trace.setdefault(seed, {}).setdefault(name, file["trace"])
            elif dic_name == "nll":
                nll.setdefault(seed, {}).setdefault(name, file["nll"])
            else:
                continue
        seeds.add(seed)
        
    for metric in [rel_error, trace, nll]:
        for seed in seeds:
            df = create_evaluation_df(seed, file["s_list"]) 
            for method in metric[seed]:        
                df = pd.concat([df, pd.DataFrame({method: metric[seed][method]})], 
                               axis=1)        
            try:
                df_save = pd.concat([df_save, df], axis=0)
            except:
                df_save = df
        save_df(df_save, os.path.join(path, f'df_{metric["name"]}.csv'))
        del df_save


if __name__ == "__main__":
    run_main()
