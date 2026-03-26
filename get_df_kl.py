import os
import math
from glob import glob
import re
from collections import defaultdict
import numpy as np
import pandas as pd

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


@hydra.main(config_path="config", config_name="config")
def run_main(cfg: DictConfig) -> None:

    # setting up paths
    path = os.path.join("results", cfg.data.folder_name, cfg.pred_model.name)
    file_names = glob(os.path.join(path, "*/", "df_kl_*"))
    name = "df_kl.csv"

    dfs = []
    for file_name in file_names:
        dfs.append(pd.read_csv(file_name))

    groups = defaultdict(list)
    for df in dfs:
        # Use (dim, seed) as a grouping key
        key = tuple(df[['dim', 'seed']].itertuples(index=False, name=None))
        groups[key].append(df)

    # Concatenate DataFrames horizontally within each group
    grouped_dfs = []
    for _, dfs_grouped in groups.items():
        # Prevent duplicate shared columns
        merged_df = pd.concat(dfs_grouped, axis=1).T.drop_duplicates().T
        grouped_dfs.append(merged_df)

    # Step 3: Concatenate the grouped DataFrames vertically
    final_df = pd.concat(grouped_dfs, ignore_index=True)
    final_df.seed = final_df.seed.astype(int)
    final_df.dim = final_df.dim.astype(int)

    save_df(final_df, os.path.join(path, name))
    
if __name__ == "__main__":
    run_main()
