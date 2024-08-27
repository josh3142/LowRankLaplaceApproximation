from typing import Tuple

from laplace import Laplace
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

import laplace

from linearized_model.low_rank_laplace import FullInvPsi
from projector.projector1d import get_jacobian
from linearized_model.low_rank_laplace import compute_Sigma, compute_Sigma_P
from linearized_model.submodel import submodel_indices

@pytest.fixture
def init_data() -> Tuple:
    device = torch.device('cpu')
    torch.manual_seed(42)
    dtype = torch.float32
    torch.use_deterministic_algorithms(True)
    # generate data
    likelihood = 'classification'
    n_train_data, n_test_data, n_input, n_class, batch_size = 100, 50, 2, 3, 50

    X_train = torch.randn(n_train_data, n_input).to(device).to(dtype)
    Y_train = torch.randint(low=0, high=n_class, size=(n_train_data,), dtype=torch.int64).to(device)
    X_test = torch.randn(n_train_data, n_input).to(device).to(dtype)
    Y_test = torch.randint(low=0, high=n_class, size=(n_train_data,), dtype=torch.int64).to(device)
    
    # create dataloaders
    train_data = TensorDataset(X_train, Y_train)
    test_data = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_data, batch_size=batch_size, 
                                  shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, 
                                  shuffle=False)
    # create model 
    model = nn.Sequential(
                nn.Linear(n_input, 5),
                nn.Linear(5, n_class)
        ).to(device).to(dtype)
    
    return model, train_loader, test_loader, X_test, device, likelihood


def test_submodel_projector(init_data: Tuple):
    model, train_loader, test_loader, X_test, device, likelihood = init_data
    submodel_method = 'magnitude'
    s = 10
    a_tol = 1e-5
    subind = submodel_indices(model=model, likelihood=likelihood,
                            train_loader=train_loader,
                            method=submodel_method)
    P = subind.P(s).to(device)
    la = Laplace(model,
                likelihood=likelihood,
                subset_of_weights='all',
                hessian_structure='full')
    la.fit(train_loader=train_loader)
    subla = Laplace(model,
        likelihood=likelihood, 
        subset_of_weights="subnetwork",
        hessian_structure='full',
        subnetwork_indices=subind(s))
    subla.fit(train_loader=train_loader)
    IPsi = FullInvPsi(inv_Psi=la.posterior_precision)
    J_X = get_jacobian(X=X_test, model=model, is_classification=False).to(device)
    sub_J_X = J_X.index_select(dim=-1, index=subind(s))
    Sigma_P_analytical = compute_Sigma_P(P=P, IPsi=IPsi, J_X=J_X)
    Sigma_P_library = subla.functional_covariance(Js=sub_J_X)
    assert torch.all(torch.isclose(Sigma_P_analytical, Sigma_P_library, atol=a_tol))




def test_submodel_methods(init_data: Tuple):
    model, train_loader, test_loader, X_test, device, likelihood = init_data
    # diagonal
    subind = submodel_indices(model=model, likelihood=likelihood,
                     train_loader=train_loader, method='diagonal')
    diag_laplace_model = laplace.Laplace(model=model,
                                    likelihood=likelihood,
                                    subset_of_weights='all',
                                    hessian_structure='diag')
    diag_laplace_model.fit(train_loader)
    assert torch.all(torch.isclose(diag_laplace_model.posterior_variance,
                                   subind.metric))

    # magnitude
    subind = submodel_indices(model=model, likelihood=likelihood,
                     train_loader=train_loader, method='magnitude')
    parameter_vector = parameters_to_vector(model.parameters())
    assert torch.all(torch.isclose(torch.abs(parameter_vector), subind.metric))