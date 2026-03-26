import os
import sys
from copy import deepcopy

import torch
import pandas as pd
from glob import glob
import re
from collections import defaultdict

import hydra 
from omegaconf import DictConfig 

from typing import List, Union
from tqdm import tqdm

from utils_df import save_df

# files that do not contain this pattern in their name are ignored
file_patterns = ["Metrics", "PredictiveMetrics", "kl"] 

def create_evaluation_df(
        seed: int,
        s: List[int]
    ) -> pd.DataFrame:

    df = pd.DataFrame(data = {"seed": [seed] * len(s), "dim": s})
    
    return df

def extract_non_trivial_s_list(file_names: List[str]) -> List:
    """
    Loops through the `s_list` items in the files from `file_names`
    and checks whether all s_list are either the same list of integers or
    [None,]. Returns the list of integers or throws an error if no such list
    was found.
    """
    s_list = None
    print('Checking consistency of s lists')
    for file_name in tqdm(file_names):
        file = torch.load(file_name, weights_only=False, map_location="cpu")
        if file['s_list'] != [None,]:
            if s_list is not None:
                if not s_list == file["s_list"]:           
                    raise ValueError('Different non-trivial s lists found')
            else:
                s_list = file['s_list']
    assert s_list is not None, "No non trivial s list found"
    return s_list
    
            
        

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

def avoid_nan_for_PNone(method: str, values: List[int], s_list: List[int]
    ) -> List[int]:
    """ 
    For `PNone` only one value is stored in the dictionary. This leads to NaN 
    values in the dataframe. Hence the value is duplicated for all `s` values.
    """
    if "PNone" in method and len(values)==1:
        return values * len(s_list)
    else:
        return values


@hydra.main(config_path="config", config_name="config")
def run_main(cfg: DictConfig) -> None:

    # open all files with suffix ".pt" in one-level subfolder
    # (this subfolders should be the different seeds, e.g. "seed1")
    path = f"results/{cfg.data.folder_name}/{cfg.pred_model.name}"
    file_names = glob(os.path.join(path, "*/", "*.pt"))
    # filter file names by file_patterns
    filtered_file_names = [f for f in file_names 
                           if any(pat in f for pat in file_patterns)]
    file_names = filtered_file_names
    nll, rel_error = {"name": "nll"}, {"name": "rel_error"} 
    trace, log_trace = {"name": "trace"}, {"name": "logtrace"}
    kl, w2 = {"name": "kl"}, {"name": "w2"}
    brier, ece = {"name": "brier"}, {"name": "ece"}
    coverage = {"name": "coverage"}
    calibration_error = {"name": "calibration"}
    metric_list = [nll, rel_error, trace, log_trace, kl, w2, brier, ece,
    coverage, calibration_error]
    seeds = set()
    s_list = extract_non_trivial_s_list(file_names)

    # for file_names in file_names_sorted:
    print('Collecting metrics from files')
    for file_name in tqdm(file_names):
        path_seed, file_name_without_path = os.path.split(file_name)
        seed = int(path_seed.split("/")[-1][-1])

        # works only for specific file name set by compute_metric.py or 
        # compare get_epistemic_covariance.py
        # I.e. the underscore "_" has semantic meaning. The file name has to 
        # contain two underscores
        if len(file_name_without_path.split("_"))==3:
            dic_name, p_approx, psi_approx = file_name_without_path.split("_")
            psi_approx, _ = psi_approx.split(".")
            file = torch.load(file_name, weights_only=False, map_location="cpu")
            name = f"P{p_approx}_{psi_approx}"

            if dic_name == "Metrics":
                try:
                    rel_error.setdefault(seed, {}).setdefault(
                        name, file["rel_error"]
                    )
                except KeyError:
                    pass
                try:
                    kl.setdefault(seed, {}).setdefault(
                        name, file["kl"]
                    )
                except KeyError:
                    pass
                try:
                    w2.setdefault(seed, {}).setdefault(
                        name, file["w2"]
                    )
                except KeyError:
                    pass
                trace.setdefault(seed, {}).setdefault(name, file["trace"])
                log_trace.setdefault(seed, {}).setdefault(name, file["logtrace"])
            elif dic_name == "PredictiveMetrics":
                nll.setdefault(seed, {}).setdefault(name, file["nll"])
                try:
                    brier.setdefault(seed, {}).setdefault(
                        name, file["brier"]
                    )
                except KeyError:
                    pass
                try:
                    ece.setdefault(seed, {}).setdefault(
                        name, file["ece"]
                    )
                except KeyError:
                    pass
                try:
                    coverage.setdefault(seed, {}).setdefault(name, file["coverage"])
                except KeyError:
                    pass
                try:
                    calibration_error.setdefault(seed, {}).setdefault(name, file["calibration"])
                except KeyError:
                    pass
            elif dic_name == "kl":
                try:
                    kl.setdefault(seed, {}).setdefault(
                        name, file["kl"]
                    )
                except KeyError:
                    pass
        seeds.add(seed)
        
    # clean up metric list
    non_trivial_metric_list = []
    for metric in metric_list:
        # only keep metrics with non-trivial values for all seeds
        try:
            if not any(metric[seed] == {} for seed in seeds):
                non_trivial_metric_list.append(metric)
        except KeyError:
            pass
    metric_list = non_trivial_metric_list

    # create dataframes for each metric
    for metric in metric_list:
        for seed in seeds:
            df = create_evaluation_df(seed, s_list)
            for method in metric[seed]: 
                values = avoid_nan_for_PNone(
                    method, metric[seed][method], s_list
                )
                df = pd.concat([df, pd.DataFrame({method: values})],
                               axis=1) 
            try:
                df_save = pd.concat([df_save, df], axis=0)
            except NameError:
                df_save = df
        df_name = f'df_{metric["name"]}.csv'
        print(f'Saving {df_name}')
        save_df(df_save, os.path.join(path,df_name ))
        del df_save


if __name__ == "__main__":
    run_main()
