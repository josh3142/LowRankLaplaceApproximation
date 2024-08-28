import openml
import numpy as np
import torch 
from torchvision.transforms import Compose

from torch import Tensor
from typing import Tuple, Callable, List

from functools import partial

from data.utils_data import shuffle_and_split_data

def get_enb(id=41478, train_size: float=0.8, seed: int=42, **kwargs, 
    ) -> Tuple[Tensor]:
    """
    Load dataset from openml, shuffle the dataset and divide it in a training 
    and test set.
    """
    dataset = openml.datasets.get_dataset(id)
    X, _, categorical_indicator, attribute_names = dataset.get_data(
        target=None, dataset_format="dataframe")
    target_columns = ["Y1", "Y2"]
    Y = X[target_columns]
    X = X.drop(columns=target_columns)
    X, Y  = X.to_numpy().astype(np.float32), Y.to_numpy().astype(np.float32)
    X_tr, Y_tr, X_te, Y_te = shuffle_and_split_data(X, Y, train_size, seed)
    Y_tr, Y_te = torch.FloatTensor(Y_tr), torch.FloatTensor(Y_te)

    return X_tr, Y_tr, X_te, Y_te


def normalize(x: Tensor, mu: List, std: List) -> Tensor:
    mu  = torch.tensor(mu)
    std = torch.tensor(std)

    return (x - mu) / std


def get_enb_trafo(train: bool=True) -> Callable:
    """
    Data transformation for protein data.
    These values are taken from the X1 with 
    X1, _, _, _ = shuffle_and_split_data(X, Y, 0.8, 42)
    """
    mu  = [0.76, 672.95, 318.66, 177.15, 5.23, 3.47, 0.23, 2.84]
    std = [0.11, 88.16, 42.81, 45.04,  1.75, 1.13, 0.13, 1.55]
    trafo = [
        lambda x: torch.FloatTensor(x),
        partial(normalize, mu = mu, std = std)
    ]
    return Compose(trafo)

if __name__ == "__main__":
    id   = 41478
    seed = 42
    dataset = openml.datasets.get_dataset(id)
    X, _, cc, aa = dataset.get_data(target=None, 
                                    dataset_format="dataframe")
    target_columns = ["Y1", "Y2"]
    Y = X[target_columns]
    X = X.drop(columns=target_columns)
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


