import numpy as np

from typing import Tuple

def shuffle_and_split_data(X: np.ndarray, Y: np.ndarray, split_size: float, 
    seed: int) -> Tuple[np.ndarray]:
    """
    Shuffles the data wrt seed. Splits data into two subsets wrt split_size 
    """
    n_ele = len(Y) 

    # shuffle data    
    rng  = np.random.default_rng(seed)
    idcs = rng.permutation(n_ele)
    X, Y = X[idcs, ...], Y[idcs]

    # split test training data
    split = int(n_ele * split_size)
    X1, X2 = X[: split, ...], X[split: , ...]
    Y1, Y2 = Y[: split], Y[split: ]

    return X1, Y1, X2, Y2 
