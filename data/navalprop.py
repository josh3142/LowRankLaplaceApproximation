import numpy as np
import pandas as pd
import torch
from torchvision.transforms import Compose
from torch import Tensor
from typing import Tuple, Callable, List

from functools import partial

from data.utils_data import shuffle_and_split_data


def get_df(path: str) -> pd.DataFrame:
    "Load the csv-file into a pandas dataframe."
    df = pd.read_csv(path, delim_whitespace=True, header=None)
    df.columns = [
        'lever_position', 'ship_speed', 'gt_shaft', 'gt_rate', 'gg_rate', 
        'sp_torque', 'pp_torque', 'hpt_temp', 'gt_c_i_temp', 'gt_c_o_temp', 
        'hpt_pressure', 'gt_c_i_pressure', 'gt_c_o_pressure', 
        'gt_exhaust_pressure', 'turbine_inj_control', 'fuel_flow', 'gt_c_decay', 
        'gt_t_decay'
    ]
    # drop two columns which only contain one unique element.
    df.drop(['gt_c_i_pressure', 'gt_c_i_temp'], axis=1, inplace=True)

    return df


def get_navalpro(
        path: str="navalplantmaintenance.csv", 
        train_size: float=0.8, 
        seed: int=42,
        **kwargs, 
    ) -> Tuple[Tensor]:
    """
    Load dataset from csv-file, shuffle the dataset and divide it in a training 
    and test set.
    """
    target=["gt_c_decay", "gt_t_decay"]
    df = get_df(path)
    X = df.drop(target, axis=1).to_numpy().astype(np.float32)
    Y = df[target].to_numpy().astype(np.float32)
    X_tr, Y_tr, X_te, Y_te = shuffle_and_split_data(X, Y, train_size, seed) 
    Y_tr, Y_te = torch.FloatTensor(Y_tr), torch.FloatTensor(Y_te)

    return X_tr, Y_tr, X_te, Y_te


def normalize(x: Tensor, mu: List, std: List) -> Tensor:
    mu  = torch.tensor(mu)
    std = torch.tensor(std)

    return (x - mu) / std


def get_navalpro_trafo(train: bool=True) -> Callable:
    """
    Data transformation for protein data.
    These values are taken from the X1 with 
    X1, _, _, _ = shuffle_and_split_data(X, Y, 0.8, 42)
    """
    mu  = [5.1554, 14.9654, 27183.146, 2133.397, 8194.811, 226.8203,
        226.8203, 734.8378, 645.923, 2.3496, 12.2792, 1.0294, 33.5883, 0.6613]
    std =[2.6314, 7.7609, 22217.77, 775.8903, 1093.4258, 201.0936, 201.0936,
        174.2754, 72.8701, 1.0879, 5.3529, 0.0104, 25.9269, 0.5089]
    trafo = [
        lambda x: torch.FloatTensor(x),
        partial(normalize, mu=mu, std=std)
    ]
    return Compose(trafo)


if __name__=="__main__":
    path = "../../../SharedData/AI/datasets/naval_propulsion/navalplantmaintenance.csv"
    seed = 42
    split_size = 0.8
    target=["gt_c_decay", "gt_t_decay"]

    X_tr = get_navalpro(path, split_size, seed)[0]
    mu  = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) 

    np.set_printoptions(precision=4, suppress=True)
    print("mean:, ", mu)
    print("std: ", std)

