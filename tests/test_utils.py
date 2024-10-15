from typing import Literal, Optional, Callable, Tuple
import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils import (
    iterator_wise_quadratic_form,
    iterator_wise_matmul,
    flatten_batch_and_target_dimension,
    estimate_regression_likelihood_sigma
)

@pytest.fixture
def iterative_framework() -> Callable:
    def _create_iterative_framework(
            seed: int=0, 
            n: int=101, 
            p: int=5, 
            batch_size: int=10, 
            class_dim_entry: Optional[int]=None
        ):
        generator = torch.Generator().manual_seed(seed)
        if class_dim_entry is None:
            J = torch.randn(n, p, generator=generator)
        else:
            J = torch.randn(n, class_dim_entry, p, generator=generator)
        number_of_batches = math.ceil(n/batch_size)
        def create_J_iterator():
            for i in range(number_of_batches):
                yield J[i*batch_size:(i+1)*batch_size]
        def quadratic_form(J_i,J_j, dim: Literal[0,1]=0) -> torch.Tensor:
            if dim==0:
                # transpose for dimensional match
                return J_i
            else:
                return J_j.T

        return J, create_J_iterator, quadratic_form, number_of_batches, generator
    return _create_iterative_framework

@pytest.fixture
def random_regression_data(request) -> Tuple:
    target_dim = request.param
    seed = 0
    n_data = 1000
    std = 1
    batch_size = 100
    generator = torch.Generator().manual_seed(seed)
    X = torch.randn((n_data, target_dim), generator=generator)
    deviation = std * torch.randn((n_data, target_dim), generator=generator)
    Y = X + deviation
    emp_std = torch.std(deviation).item()
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    class ToyModel(nn.Module):
        def __init__(self, X=X):
            super().__init__()
            self.batch_counter = 0
            self._X = X

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self._X[self.batch_counter:self.batch_counter+x.size(0)]
            self.batch_counter += x.size(0)
            return out
    model = ToyModel(X=X)
    return dataloader, model, emp_std
    

def test_iterator_wise_quadratic_form(iterative_framework: Callable):
    J, create_J_iterator, quadratic_form, number_of_batches, _ = \
        iterative_framework()
    # Test dim = 0
    computed_value = iterator_wise_quadratic_form(
        quadratic_form=quadratic_form,
        create_iterator=create_J_iterator)
    theoretical_value = J.repeat(1,number_of_batches)
    assert torch.equal(computed_value, theoretical_value)
    # Test dim = 1
    quadratic_form_2 = lambda x,y: quadratic_form(x,y,dim=1)
    computed_value = iterator_wise_quadratic_form(
        quadratic_form=quadratic_form_2,
        create_iterator=create_J_iterator)
    theoretical_value = J.T.repeat(number_of_batches, 1)
    assert torch.equal(computed_value, theoretical_value)


def test_iterator_wise_matmul(iterative_framework: Callable):
    J, create_J_iterator, _, _, generator = iterative_framework()

    # test dim=0
    W = torch.randn(J.size(-1), 10, generator=generator)

    theoretical_value = J @ W
    computed_value = iterator_wise_matmul(
        create_a_iterator=create_J_iterator, 
        b=W
    )
    assert torch.all(torch.isclose(theoretical_value, computed_value))
    # test dim=1
    W = torch.randn(J.size(0), 10, generator=generator)

    theoretical_value = J.T @ W
    computed_value = iterator_wise_matmul(
        create_a_iterator=create_J_iterator, 
        b=W, 
        transpose_a=True, 
        iteration_dim=1
    )
    assert torch.all(torch.isclose(theoretical_value, computed_value))


@pytest.mark.parametrize('class_dim_entry', [None, 10])
def test_flatten_iterator(iterative_framework: Callable, class_dim_entry):
    J, get_J_iterator, _, _, _ = \
        iterative_framework(class_dim_entry=class_dim_entry)
    flat_J = flatten_batch_and_target_dimension(J)
    create_flat_J_iterator = flatten_batch_and_target_dimension(get_J_iterator)
    assert torch.equal(
        flat_J, 
        torch.concat([j for j in create_flat_J_iterator()])
    )
    

    


@pytest.mark.parametrize('random_regression_data', [1,3,10], indirect=True)
def test_estimate_regression_likelihood_sigma(random_regression_data: Tuple):
    dataloader, model, emp_std = random_regression_data
    estimated_std = estimate_regression_likelihood_sigma(
        model=model,
        dataloader=dataloader,
        )
    assert torch.isclose(torch.tensor(estimated_std), torch.tensor(emp_std))
    

