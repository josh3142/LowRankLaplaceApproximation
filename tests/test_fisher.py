import torch
from torch import nn, Tensor
from torch.func import jacrev
from torch.utils.data import DataLoader, TensorDataset


from functools import partial

import pytest
from typing import Tuple

from projector.fisher import (get_Vs, get_I_softmax_ln, get_I_outer_product,
    get_I_sum, get_V_iterator)
from projector.projector1d import get_jacobian
from utils import param_to_vec, get_softmax_model_fun, get_model_fun


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
    
    def get_softmax_y_hat(self, x: Tensor) -> Tensor:
        return nn.functional.softmax(self.get_y_hat(x), dim=-1)

    def get_dy_hat(self, x: Tensor) -> Tensor:
        zero = torch.zeros_like(x)
        return torch.stack([
            torch.cat([self.p[2] * x, self.p[2] * x**2, self.p[0] * x + self.p[1] * x**2, zero], axis=-1),
            torch.cat([self.p[3] * x, self.p[3] * x**2, torch.zeros_like(x), self.p[0] * x + self.p[1] * x**2], axis=-1)
        ], axis=-1)

    def get_fisher_categorical(self, x: Tensor) -> Tensor:
        """
        Computes 
        ```
           f_0 \nabla_p ln(f_0) \nabla_p ln(f_0) + 
           f_1 \nabla_p ln(f_1) \nabla_p ln(f_1)
           with f_i = softmax(y_hat_i)
        ```
        """
        dy_hat = self.get_dy_hat(x)
        y_hat = self(x)
        f = nn.functional.softmax(y_hat, dim=-1)
        exp_y_hat = torch.exp(y_hat)
        inv_sum_exp_y_hat = torch.sum(exp_y_hat, dim=-1)**(-1)
        dsum_exp_y_hat = torch.sum(exp_y_hat[:, None, :] * dy_hat, dim=-1) 
        d_ln_f = (dy_hat - inv_sum_exp_y_hat[:, None, None] * 
                dsum_exp_y_hat[:, :, None])

        fisher_prod = lambda x, dx: torch.einsum("bc, bic, bjc -> ij", x, dx, dx)

        return fisher_prod(f, d_ln_f)


    def get_fisher_gaussian(self, x: Tensor) -> Tensor:
        """
        Computes 
        ```
            \nabla_p y_hat_0 \nabla_p y_hat_0 + \nabla_p y_hat_1 \nabla_p y_hat_1
        ```
        """
        dyadic_prod = lambda x: torch.einsum("bi, bj -> ij", x, x)

        return (dyadic_prod(self.get_dy_hat(x)[..., 0]) + 
                dyadic_prod(self.get_dy_hat(x)[..., 1]))
    

    def forward(self, x: Tensor) -> Tensor:
        return self.get_y_hat(x)


@pytest.fixture
def init_data() -> Tuple:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    # generate data
    n_data, n_input, n_class = 83, 3, 4 
    X = torch.randn(n_data, n_input, dtype=torch.float64)
    Y_ce = torch.randint(low=0, high=n_class, size=(n_data,))
    Y_nll = torch.rand((n_data, n_class))
    # model 
    model = nn.Sequential(
                nn.Linear(n_input, 5),
                nn.ReLU(),
                nn.Linear(5, n_class)
        ).to(dtype=torch.float64)
    
    return model, X, Y_ce, Y_nll


def test_I_ln_and_I_sqrt_coincide(init_data: Tuple):
    """
    Test if get_I_softmax_ln and get_I_softmax_sqrt yield the same Information 
    matrix.
    """
    model, X, _, _ = init_data
    param_vec = param_to_vec(model.parameters())

    jac_sqrt = jacrev(partial(
            get_softmax_model_fun, 
            param_gen=model.named_parameters(), 
            model=model, 
            X=X, 
            fun=lambda x: 2 * torch.sqrt(x))
        )(param_vec)
    I_sqrt = get_I_outer_product(jac_sqrt)

    Y_hat = nn.Softmax(dim=-1)(model(X))
    jac_ln = jacrev(partial(
            get_softmax_model_fun, 
            param_gen=model.named_parameters(), 
            model=model, 
            X=X, 
            fun=torch.log)
        )(param_vec)
    I_ln = get_I_softmax_ln(jac_ln, Y_hat)

    assert torch.allclose(I_ln, I_sqrt), \
        "Both methods to compute the Fisher Information coincide."


def test_V_ce(init_data: Tuple):
    """ Test the sum of V with the Fisher information. 
    For a categorical distribution the Fisher Information is
    ```
        4 \sum_ic \nabla_p sqrt(Y_hat_ic) \nabla_p sqrt(Y_hat_ic)
    ```
    """
    model, X, Y, _ = init_data
    dl = DataLoader(TensorDataset(X, Y), batch_size=8, shuffle=False)

    fun = lambda x: torch.sqrt(x)
    V_true = 2 * get_jacobian(model, X, fun=fun, is_classification=True)

    Vs = get_Vs(model, dl, is_classification=True)

    assert torch.allclose(V_true, Vs, atol=1e-5)


def test_V_gaussian(init_data: Tuple):
    """ Test the sum of V with the Fisher information. 
    For a Gaussian distribution the Fisher Information is
    ```
        \sum_i \sum_c nabla_p Y_hat_ic nabla_p Y_hat_ic
    ```
    """
    model, X, Y, _ = init_data
    dl = DataLoader(TensorDataset(X, Y), batch_size=8, shuffle=False)

    V_true = get_jacobian(model, X, fun=lambda x: x, is_classification=False)
    Vs = get_Vs(model, dl, is_classification=False)

    assert torch.allclose(V_true, Vs, atol=1e-5)


def test_V_gaussian_explicit_with_I(init_data: Tuple):
    _, _, _, _ = init_data
    n_data, n_input, n_class = 20, 1, 2
    X = torch.randn(n_data, n_input, dtype=torch.float32)
    Y = torch.rand((n_data, n_class))
    dl = DataLoader(TensorDataset(X, Y), batch_size=7, shuffle=False)
    model = ToyModel()

    I_true = model.get_fisher_gaussian(X)

    Vs = get_Vs(model, dl, is_classification=False)
    I = get_I_outer_product(Vs)

    assert torch.allclose(I_true, I, atol=1e-5)


def test_V_categorical_explicit_with_I(init_data: Tuple):
    _, _, _, _ = init_data
    n_data, n_input, n_class = 23, 1, 2
    X = torch.randn(n_data, n_input, dtype=torch.float32)
    Y = torch.randint(low=0, high=(n_class), size=(n_data, ))
    dl = DataLoader(TensorDataset(X, Y), batch_size=7, shuffle=True)
    model = ToyModel()

    I_true = model.get_fisher_categorical(X)

    Vs = get_Vs(model, dl, is_classification=True)
    I = get_I_outer_product(Vs, reduction="sum")

    assert torch.allclose(I_true, I, atol=1e-5)

def test_I_sum_categorical(init_data):
    model, X, Y, _ = init_data
    dl = DataLoader(TensorDataset(X, Y), batch_size=7, shuffle=True)
    param_vec = param_to_vec(model.parameters())
    var = 3.21

    # true I
    jac_sqrt = jacrev(partial(
        get_softmax_model_fun, 
        param_gen=model.named_parameters(), 
        model=model, 
        X=X, 
        fun=lambda x: 2 * torch.sqrt(x))
    )(param_vec)
    I_true = get_I_outer_product(jac_sqrt) / var

    I = get_I_sum(model, dl, is_classification=True, var=var)

    assert torch.allclose(I, I_true)


def test_I_sum_gaussian(init_data):
    model, X, _, Y = init_data
    var = 1243.312
    dl = DataLoader(TensorDataset(X, Y), batch_size=7, shuffle=False)
    param_vec = param_to_vec(model.parameters())

    # true I
    jac = jacrev(partial(
        get_model_fun, 
        param_gen=model.named_parameters(), 
        model=model, 
        X=X, 
        fun=lambda x: x)
    )(param_vec)
    I_true = get_I_outer_product(jac) / var

    I = get_I_sum(model, dl, is_classification=False, var=var)

    assert torch.allclose(I, I_true)


@pytest.mark.parametrize("is_classification", [True, False])
def test_get_V_iterator_gaussian(init_data: Tuple, is_classification: bool):
    """ Test whether get_V_iterator yields the full V
    """
    model, X, Y, _ = init_data
    dl = DataLoader(TensorDataset(X, Y), batch_size=8, shuffle=False)

    Vs = get_Vs(model, dl, is_classification=is_classification)
    V_it = get_V_iterator(model=model, dl=dl, is_classification=is_classification)
    concat_V_it = torch.concat([v for v in V_it], dim=0)
    assert torch.allclose(Vs, concat_V_it.cpu(), atol=1e-5)