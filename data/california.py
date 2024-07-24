import openml
import numpy as np
import torch 
from torchvision.transforms import Compose

from torch import Tensor
from typing import Tuple, Callable, List

from functools import partial

from data.utils_data import shuffle_and_split_data

def get_california(id=44025, train_size: float=0.8, seed: int=42, **kwargs, 
    ) -> Tuple[Tensor]:
    """
    Load dataset from openml, shuffle the dataset and divide it in a training 
    and test set.
    """
    dataset = openml.datasets.get_dataset(id)
    X, Y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format = "dataframe")
    X, Y  = X.to_numpy().astype(np.float32), Y.to_numpy().astype(np.float32)
    X_tr, Y_tr, X_te, Y_te = shuffle_and_split_data(X, Y, train_size, seed)
    Y_tr, Y_te = torch.FloatTensor(Y_tr), torch.FloatTensor(Y_te)

    return X_tr, Y_tr, X_te, Y_te


def normalize(x: Tensor, mu: List, std: List) -> Tensor:
    mu  = torch.tensor(mu)
    std = torch.tensor(std)

    return (x - mu) / std


def get_california_trafo(train: bool=True) -> Callable:
    """
    Data transformation for protein data.
    These values are taken from the X1 with 
    X1, _, _, _ = shuffle_and_split_data(X, Y, 0.8, 42)
    """
    mu  = [3.87, 28.71, 5.43, 1.10, 1424.57, 3.11, 35.63, -119.57]
    std = [1.90, 12.61, 2.54, 0.49, 1139.63, 11.61, 2.13, 2.00]
    trafo = [
        lambda x: torch.FloatTensor(x),
        partial(normalize, mu = mu, std = std)
    ]
    return Compose(trafo)

if __name__ == "__main__":
    id   = 44025
    seed = 42
    dataset = openml.datasets.get_dataset(id)
    X, Y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format = "dataframe")
    X, Y  = X.to_numpy().astype(np.float32), Y.to_numpy().astype(np.float32)
    
    # shuffle data    
    rng  = np.random.default_rng(seed)
    split_size = 0.8
    idcs = rng.permutation(len(X))
    X    = torch.tensor(X[idcs, ...])

    mu  = X[: int(len(X) * split_size), ...].mean(dim = 0)
    std = X[: int(len(X) * split_size), ...].std(dim = 0) 

    torch.set_printoptions(precision=2, sci_mode=False)
    print(mu)
    print(std)


