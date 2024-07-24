import torch
from torch import nn
from torch.func import jacrev


from functools import partial

import pytest
from typing import Tuple

from projector.projector1d import (get_gradient_projector_1d, 
    get_sigma_projected_1d, get_sigma_1d, get_pred_var)
from utils import param_to_vec, get_softmax_model_fun


@pytest.fixture
def init_data() -> Tuple:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    # generate data
    n_data, n_input, n_class = 200, 2, 3 
    X = torch.randn(n_data, n_input, dtype=torch.float64)
    Y = torch.randint(low=0, high=n_class, size=(n_data,))
    # model 
    model = nn.Sequential(
                nn.Linear(n_input, 5),
                nn.Linear(5, n_class)
        ).to(dtype=torch.float64)
    
    return model, X, Y

def test_get_gradient_projector_1d(init_data: Tuple):
    """
    Test if the gradient of get_sigma_projected_1d is computed correctly.
    """
    model, X, _ = init_data
    # index and class of the component of Sigma_p of which the gradient is computed
    idx, cc = 0, 0  

    # compute V
    param_vec = param_to_vec(model.parameters())
    V = jacrev(partial(
            get_softmax_model_fun, 
            param_gen=model.named_parameters(), 
            model=model, 
            X=X, 
            fun=torch.sqrt)
        )(param_vec)
    V = V.reshape(-1, V.shape[-1]).T.detach()

    # compute J
    param_vec = param_to_vec(model.parameters())
    j = jacrev(partial(
            get_softmax_model_fun, 
            param_gen=model.named_parameters(), 
            model=model, 
            X=X[idx], 
            fun=lambda x: x)
        )(param_vec)
    j = j[cc].detach()
    
    # compute gradient
    p = torch.randn_like(j, requires_grad=True)
    grad_sigma_p = get_gradient_projector_1d(p, j, V)

    # compute gradient numerically with pytorch
    grad_sigma_p_true = jacrev(partial(
            get_sigma_projected_1d, 
            j=j, 
            V=V)     
        )(p)
    
    # grad_sigma_p computes gradient times 2        
    torch.allclose(grad_sigma_p_true, grad_sigma_p / 2)


def test_get_sigma_1d():
    """
    Test if get_sigma_1d gives the correct output
    """
    n, m = 20, 10
    j = torch.rand(n)
    V = torch.rand((n, m))
    V_inv = torch.linalg.inv(torch.eye(n) + V @ V.T)
    Sigma_true = torch.einsum("a, ab, b", j, V_inv, j)

    Sigma = get_sigma_1d(j, V)

    torch.allclose(Sigma_true, Sigma)
    
def test_get_sigma_projected_1d():
    """
    Test if get_sigma_projected_1d yields same result as get_sigma_1d.
    """
    n, m = 20, 10
    j = torch.rand(n)
    V = torch.rand((n, m))
    V_inv = torch.linalg.inv(torch.eye(n) + V @ V.T)
    Sigma_true = torch.einsum("a, ab, b", j, V_inv, j)

    # optimal p
    p = torch.linalg.inv(V @ V.T  + torch.eye(n)) @ j
    Sigma_p = get_sigma_projected_1d(p, j, V)

    torch.allclose(Sigma_true, Sigma_p)


def test_get_pred_var():
    """
    Compare get_pred_var with its Woodbury identity representation.
    """
    n, m, c = 20, 10, 4
    j = torch.rand(c, n)
    V = torch.rand((n, m))
    V_inv = torch.linalg.inv(torch.eye(n) + V @ V.T)
    Sigma_true = torch.einsum("ka, ab, lb", j, V_inv, j)

    # optimal p
    Sigma = get_pred_var(j, V)

    torch.allclose(Sigma_true, Sigma)