import openml
import numpy as np
import torch

from torch import Tensor
from typing import Tuple, List

from functools import partial

from data.utils_data import shuffle_and_split_data


def get_protein(id = 42903, train_size: float = 0.8, seed: int = 42, **kwargs, 
    ) -> Tuple[Tensor]:
    """
    Load dataset from openml, shuffle the dataset and divide it in a training 
    and test set.
    """
    dataset = openml.datasets.get_dataset(id)
    data, _, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format = "dataframe")
    X, Y  = data.drop(columns = ["RMSD"]).to_numpy(), data["RMSD"].to_numpy()
    X_tr, Y_tr, X_te, Y_te = shuffle_and_split_data(X, Y, train_size, seed)
    Y_tr, Y_te = torch.FloatTensor(Y_tr), torch.FloatTensor(Y_te)

    return X_tr, Y_tr, X_te, Y_te


def normalize(x: Tensor, mu: List, std: List) -> Tensor:
    mu  = torch.tensor(mu)
    std = torch.tensor(std)

    return (x - mu) / std


def get_protein_trafo(train: bool = True) -> List:
    """
    Data transformation for protein data.
    These values are taken from the X1 with 
    X1, _, _, _ = shuffle_and_split_data(X, Y, 0.8, 42)
    """
    mu  = [9887.95, 3022.32, 0.30, 103.77, 1370553.21, 145.94, 3989.74, 70.32, 
           34.49]
    std = [4071.43, 1466.28, 0.06, 55.69, 565994.52, 70.27, 1912.42, 56.84,
           6.01]
    trafo = [
        lambda x: torch.FloatTensor(x),
        partial(normalize, mu = mu, std = std)
    ]
    return trafo

if __name__ == "__main__":
    id   = 42903
    seed = 42
    dataset = openml.datasets.get_dataset(id)
    data, _, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format = "dataframe")
    X, _  = data.drop(columns = ["RMSD"]).to_numpy(), data["RMSD"].to_numpy()
    
    # shuffle data    
    rng  = np.random.default_rng(seed)
    split_size = 0.8
    idcs = rng.permutation(len(X))
    X    = torch.tensor(X[idcs, ...])

    mu  = X[: int(len(X) * split_size), ...].mean(dim = 0)
    std = X[: int(len(X) * split_size), ...].std(dim = 0) 

    torch.set_printoptions(precision = 2, sci_mode = False)
    print(mu)
    print(std)


