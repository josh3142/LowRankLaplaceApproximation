from copy import deepcopy

import torch
from torch import nn
from torch.func import jacrev


from functools import partial

import pytest
from typing import Tuple

from projector.projector1d import (
    get_gradient_projector_1d, 
    get_sigma_projected_1d,
    get_sigma_1d,
    get_pred_var,
    get_jacobian,
    where_parameters_with_grad,
    number_of_parameters_with_grad,
    )
from pred_model.resnet9 import ResNet9
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

@pytest.fixture
def resnet_without_batchnorm_grad() -> Tuple:
    # parameters for generating data and models
    n_data = 2
    C = 3
    n_class = 3
    image_dim = 32
    seed = 0
    generator = torch.Generator().manual_seed(seed)
    # generate random data
    X = torch.randn((n_data,C,image_dim,image_dim,), generator=generator)
    # generate models
    model_batchnorm_grad = ResNet9(C=C, n_class=10)
    model_batchnorm_grad.eval()
    model_batchnorm_no_grad = deepcopy(model_batchnorm_grad)
    model_batchnorm_no_grad.eval()
    # switch off gradients
    for module in model_batchnorm_no_grad.modules():
        if type(module) is nn.BatchNorm2d:
            for par in module.parameters():
                par.requires_grad = False
    return X, model_batchnorm_grad, model_batchnorm_no_grad

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

    
def test_get_jacobian_for_switched_off_batchnorm_layers(
    resnet_without_batchnorm_grad: Tuple
    ):
    # load model with BatchNorm2d layers
    X, model_batchnorm_grad, model_batchnorm_no_grad = \
        resnet_without_batchnorm_grad
    full_J_x = get_jacobian(model=model_batchnorm_grad, X=X)
    sub_J_X = get_jacobian(model=model_batchnorm_no_grad, X=X)

    # indices for switched off gradients
    where_requires_grad_False = torch.logical_not(
       where_parameters_with_grad(model_batchnorm_no_grad) 
    )
    # check whether there really parameters with no requires_grad 
    assert torch.sum(where_requires_grad_False) > 0

    # test consistency
    assert torch.allclose(
        full_J_x[...,torch.logical_not(where_requires_grad_False)],
        sub_J_X
    )
    

def test_where_parameters_with_grad(resnet_without_batchnorm_grad: Tuple):
    _, model_batchnorm_grad, model_batchnorm_no_grad = \
        resnet_without_batchnorm_grad
    assert sum(where_parameters_with_grad(model_batchnorm_grad)) \
        != sum(where_parameters_with_grad(model_batchnorm_no_grad))
    assert sum(where_parameters_with_grad(model_batchnorm_grad)) \
        == number_of_parameters_with_grad(model_batchnorm_grad)
    assert sum(where_parameters_with_grad(model_batchnorm_no_grad)) \
        == number_of_parameters_with_grad(model_batchnorm_no_grad)
     