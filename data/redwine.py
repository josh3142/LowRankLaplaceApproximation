import openml
import numpy as np
import torch 
from torchvision.transforms import Compose

from torch import Tensor
from typing import Tuple, Callable, List

from data.utils_data import shuffle_and_split_data


def get_redwine(id: int=40691, train_size: float=0.8, seed: int=42, **kwargs, 
    ) -> Tuple[Tensor]:
    """
    Load dataset from openml, shuffle the dataset and divide it in a training 
    and test set.
    """
    dataset = openml.datasets.get_dataset(id)
    X, Y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format = "dataframe")
    X, Y  = X.to_numpy().astype(np.float32), Y.to_numpy().astype(np.float32)
    X_tr, Y_tr, X_te, Y_te = shuffle_and_split_data(X, Y, train_size, seed)
    Y_tr, Y_te = torch.FloatTensor(Y_tr), torch.FloatTensor(Y_te)
    
    return X_tr, Y_tr, X_te, Y_te


def normalize(x: Tensor, mu: List, std: List) -> Tensor:
    mu  = torch.tensor(mu)
    std = torch.tensor(std)

    return (x - mu) / std


def get_redwine_trafo(train: bool = True) -> Callable:
    """
    Data transformation for protein data.
    These values are taken from the X1 with 
    X1, _, _, _ = shuffle_and_split_data(X, Y, 0.8, 42)
    """
    mu  = [8.2845, 0.5274, 0.2689, 2.5570, 0.0867, 15.8425, 46.4429, 0.9967,
        3.3149, 0.6593, 10.4398] 
    std = [1.7439, 0.1816, 0.1949, 1.4591, 0.0453, 10.5184, 32.6342, 0.0019,
        0.1568, 0.1693, 1.0790]
    trafo = [
        lambda x: torch.FloatTensor(x)#,
        # partial(normalize, mu = mu, std = std)
    ]
    return Compose(trafo)

if __name__ == "__main__":
    id   = 40691
    seed = 42
    dataset = openml.datasets.get_dataset(id)
    X, Y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format = "dataframe")
    X, Y  = X.to_numpy(), Y.to_numpy().astype(np.float32)
    # shuffle data    
    rng  = np.random.default_rng(seed)
    split_size = 0.8
    idcs = rng.permutation(len(X))
    X    = torch.tensor(X[idcs, ...])

    mu  = X[: int(len(X) * split_size), ...].mean(dim = 0)
    std = X[: int(len(X) * split_size), ...].std(dim = 0) 

    torch.set_printoptions(precision = 4, sci_mode = False)
    print(mu)
    print(std)


