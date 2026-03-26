import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import log_softmax
from torch.distributions import MultivariateNormal, kl_divergence

from linearized_model.approximation_metrics import (
    get_kl_categorical, KL_multivariate_normal
)


def get_own_kl_categorical(log_Y_full: Tensor, log_Y: Tensor) -> Tensor:
    return log_Y_full.exp() * (log_Y_full - log_Y)

def random_positive_definite(dim, eps, seed=None):
    if seed is not None:
        torch.manual_seed(seed)  

    A = torch.rand(dim, dim)
    matrix = A @ A.T + eps * torch.eye(dim)

    eigvals = torch.linalg.eigvalsh(matrix)
    assert torch.all(eigvals > 0), "Not all eigenvalues are positive."

    return matrix


def test_categorical_kl():
    n_data, n_class = 100, 20
    log_Y_full = log_softmax(torch.randn(n_data, n_class), dim=-1)
    log_Y_proj = log_softmax(torch.randn(n_data, n_class), dim=-1)
    
    kl_val = get_own_kl_categorical(log_Y_full, log_Y_proj)
    kl_test = get_kl_categorical(log_Y_full, log_Y_proj)

    assert torch.allclose(kl_test, kl_val)


def test_gaussian_kl():
    dim, eps = 100, 1
    Sigma_approx = random_positive_definite(dim, eps, 0)
    Sigma = random_positive_definite(dim, eps, 42)
    kl_test = KL_multivariate_normal(Sigma_approx, Sigma, epsilon=0).item()
    dist_approx =  MultivariateNormal(torch.zeros(dim), Sigma_approx)
    dist = MultivariateNormal(torch.zeros(dim), Sigma)
    kl_val = kl_divergence(dist, dist_approx).item()
    
    assert np.isclose(kl_test, kl_val)