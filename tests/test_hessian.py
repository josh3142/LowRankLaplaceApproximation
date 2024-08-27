import torch
from torch import nn, Tensor
from torch.func import jacrev, hessian, functional_call
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader, TensorDataset
from functools import partial
import pytest
from typing import Tuple, Callable

from projector.hessian import (get_hessian_nll, get_hessian_ce, 
    get_hessian_gaussian_nll, get_H_sum)
from utils import param_to_vec, vec_to_dict


class ToyModel(nn.Module):

    def __init__(self, rand_init: bool=True, **kwargs):
        super().__init__()
        if not rand_init:
            self.p = nn.Parameter(
                torch.ones(4, requires_grad=True, dtype=torch.float32)
            )
        else: 
            self.p = nn.Parameter(
                torch.rand(4, requires_grad=True, dtype=torch.float32)
            )

    def __call__(self, x):
        return self.get_y_hat(x)
        
    def get_y_hat(self, x: Tensor) -> Tensor:
        """
        Generates multiclass polynomial of the form 
        ```
            p_2 * (x * p_0 + x**2 * p_1)
            p_3 * (x * p_0 + x**2 * p_1)
        ```
        """
        poly = lambda x: torch.cat([
            self.p[2] * (x * self.p[0] + x**2 * self.p[1]),
            self.p[3] * (x * self.p[0] + x**2 * self.p[1])
        ], axis=-1)
        
        return poly(x)

    def get_y_hat_true_class(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.take_along_dim(self.get_y_hat(x), y, dim=-1)
    
    def get_dy_hat_true_class(self, x: Tensor, y: Tensor) -> Tensor:
        dy_hat_true_class = lambda x, y: torch.cat([
            x * (self.p[2] * (1 - y) + y * self.p[3]), 
            x**2 * (self.p[2] * (1 - y) + y * self.p[3]), 
            (1 - y) * (self.p[0] * x + self.p[1] * x**2),
            y * (self.p[0] * x + self.p[1] * x**2), 
        ], axis=-1)

        return dy_hat_true_class(x, y)

    def get_ddy_hat_true_class(self, x: Tensor, y: Tensor) -> Tensor:
        zero = torch.zeros(x.shape[0], 1)
        ddy_hat_true_class = lambda x, y: torch.stack([
                torch.cat([zero, zero, (1 - y) * x, y * x], axis=-1),
                torch.cat([zero, zero, (1 - y) * x**2, y * x**2], axis=-1),
                torch.cat([(1 - y) * x, (1 - y) * x**2, zero, zero], axis=-1),
                torch.cat([y * x, y * x**2, zero, zero], axis=-1)
            ], axis=-1)

        return ddy_hat_true_class(x, y)

    def get_hessian_nll(self, x, y):
        y_hat_true_class = self.get_y_hat_true_class(x, y).unsqueeze(-1)
        
        return - (
            - y_hat_true_class**(-2) * torch.einsum("bi, bj -> bij", 
                                                    self.get_dy_hat_true_class(x, y), 
                                                    self.get_dy_hat_true_class(x, y))
            + y_hat_true_class**(-1) * self.get_ddy_hat_true_class(x, y)
       )

    def forward(self, x: Tensor) -> Tensor:
        return self.get_y_hat(x)


def model_fun(model: nn.Module, param_vec: Tensor, param_dict: dict, X: Tensor
    ) -> Callable:
    "Helper fun that takes as input the model parameters to take derivates."
    return lambda param_vec: functional_call(model, vec_to_dict(param_vec, param_dict), X)


@pytest.fixture
def init_data() -> Tuple:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    # generate data
    n_data, n_input, n_class = 20, 1, 2
    X = torch.randn(n_data, n_input, dtype=torch.float32)
    Y_ce = torch.randint(low=0, high=(n_class), size=(n_data, 1))
    Y_nll = torch.rand((n_data, n_class))
    return X, Y_ce, Y_nll

def test_nll(init_data: Tuple):
    X, Y, _ = init_data
    # model 
    model = ToyModel()

    H_true = model.get_hessian_nll(X, Y)
    H = get_hessian_nll(model, X, Y.squeeze())

    assert torch.allclose(H_true, H, atol=1e-4)


def test_ce(init_data: Tuple):
    X, Y, _ = init_data
    Y = Y.squeeze()
    # model 
    model = ToyModel()

    H_true = get_hessian_nll(model, X, Y, 
                             fun=partial(log_softmax, dim=-1))
    H = get_hessian_ce(model, X, Y)

    assert torch.allclose(H_true, H, atol=1e-7)


def test_gaussian_nll(init_data: Tuple):
    """ Test Hessian for Gaussian negative log-likelihood. 
    The true Hessian is computed with `hessian` and `jacrev` from `torch`.
    ```
        H_true = 1 / var * (dY_hat * dY_hat - (Y - Y_hat) * ddY_hat)
    ```
    """
    X, _, Y = init_data
    var = 2.3 # set homoscedastic variance to some value
    # model 
    model = ToyModel()
    param_vec = model.p

    # compute true Hessian
    Y_hat = model(X)
    dY_hat = jacrev(model_fun(model, param_vec, model.named_parameters(), X))(param_vec)
    dY_hat2 = torch.einsum("bij, bil -> bijl", dY_hat, dY_hat)
    ddY_hat = hessian(model_fun(model, param_vec, model.named_parameters(), X))(param_vec)
    ddY_hat2 = torch.einsum("bi, bijl -> bijl", Y - Y_hat, ddY_hat)
    H_true = 1 / var * (dY_hat2 - ddY_hat2)

    # compute hessian
    H = get_hessian_gaussian_nll(model, X, Y, var)

    assert torch.allclose(H_true, H, atol=1e-5)

def test_H_sum_se(init_data: Tuple):
    X, Y, _ = init_data
    Y = Y.squeeze()
    dl = DataLoader(TensorDataset(X, Y), batch_size=8, shuffle=False)
    
    # model 
    model = ToyModel()

    H_true = get_hessian_ce(model, X, Y).sum(0)
    H = get_H_sum(model, dl, is_classification=True, n_batches=None)

    assert torch.allclose(H_true, H, atol=1e-5)

def test_H_sum_gaussian_nll(init_data: Tuple):
    """ 
    Test Hessian for Gaussian negative log-likelihood summed over `DataLoader`. 
    The true Hessian is computed with `hessian` and `jacrev` from `torch`.
    ```
        H_true = 1 / var * (dY_hat * dY_hat - (Y - Y_hat) * ddY_hat)
    ```
    """   
    X, _, Y = init_data
    dl = DataLoader(TensorDataset(X, Y), batch_size=8, shuffle=False)

    var = 2.03214 # set homoscedastic variance to some value
    
    # model 
    model = ToyModel()
    param_vec = model.p

    # compute true Hessian
    Y_hat = model(X)
    dY_hat = jacrev(model_fun(model, param_vec, model.named_parameters(), X))(param_vec)
    dY_hat2 = torch.einsum("bij, bil -> bijl", dY_hat, dY_hat)
    ddY_hat = hessian(model_fun(model, param_vec, model.named_parameters(), X))(param_vec)
    ddY_hat2 = torch.einsum("bi, bijl -> bijl", Y - Y_hat, ddY_hat)
    H_true = 1 / var * (dY_hat2 - ddY_hat2).sum(dim=(0,1))

    H = get_H_sum(model, dl, is_classification=False, n_batches=None, var=var)

    assert torch.allclose(H_true, H, atol=1e-5)

