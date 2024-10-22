from typing import Callable, Tuple

import numpy as np

import pytest
import torch
from torch import nn, Tensor


from linearized_model.approximation_metrics import NLL


def get_gaussian_nll_fun(y_hat: Tensor, covariance: Tensor) -> Callable:
    """ Compute NLL for Gaussian distribution """
    dim = y_hat.shape[-1]
    sigma_inv = torch.linalg.inv(covariance)
    def nll_fun(y: Tensor) -> float:
        return  1 / 2 * (dim * np.log(2 * np.pi) + torch.logdet(covariance)
            + torch.einsum("bj, bji, bi -> b", y - y_hat, sigma_inv, y - y_hat))
    return nll_fun


def get_categorical_nll_fun(y_hat: Tensor, covariance: Tensor) -> Callable:
    """Compute NLL for categorical distribution with probit approximation."""
    rescale_y_hat = y_hat / \
        torch.sqrt(1 + np.pi / 8 * torch.diagonal(covariance, dim1=-1, dim2=-2))
    log_phi = nn.functional.log_softmax(rescale_y_hat, dim=-1)
    def nll_fun(y: Tensor) -> float:
        return - log_phi[np.arange(len(y)), y]
    return nll_fun


@pytest.fixture
def init_data() -> Tuple:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    # generate data
    n_data, n_input, n_class = 200, 2, 3 
    X = torch.randn(n_data, n_input)
    Y = torch.rand(n_data, n_class)
    Y_class = torch.randint(low=0, high=n_class, size=(n_data,))
    # create positive definite symmetric matrix
    M = torch.rand(n_data, n_class, n_class)
    M = M @ M.transpose(-1, -2)
    covariance = M + M.transpose(-1, -2)
    # model 
    model = nn.Sequential(
                nn.Linear(n_input, 7),
                nn.Linear(7, n_class)
        )
    
    return model, X, Y, Y_class, covariance


def test_gaussian_nll(init_data: Tuple):
    model, X, Y, _, Sigma = init_data
    y_hat = model(X)
    nll_val = get_gaussian_nll_fun(y_hat, Sigma)(Y)
    nll_test = NLL(y_hat, Sigma, Y, is_classification=False, sum=False)

    assert torch.allclose(nll_test, nll_val)


def test_categorical_nll(init_data: Tuple):
    model, X, _, Y, Sigma = init_data
    y_hat = model(X)
    nll_val = get_categorical_nll_fun(y_hat, Sigma)(Y)
    nll_test = NLL(y_hat, Sigma, Y, is_classification=True, sum=False)

    assert torch.allclose(nll_test, nll_val)
