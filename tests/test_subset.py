from typing import Tuple

from laplace import Laplace
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

import laplace
from laplace.utils import  LargestVarianceSWAGSubnetMask

from linearized_model.low_rank_laplace import FullInvPsi
from projector.projector1d import get_jacobian
from linearized_model.low_rank_laplace import compute_Sigma_P
from linearized_model.subset import subset_indices

@pytest.fixture
def init_data(request) -> Tuple:
    likelihood = request.param # classification or regression
    device = torch.device('cpu')
    torch.manual_seed(42)
    dtype = torch.float32
    torch.use_deterministic_algorithms(True)
    # generate data
    n_train_data, n_test_data, n_input, n_class, batch_size = 100, 50, 2, 3, 50

    X_train = torch.randn(n_train_data, n_input).to(device).to(dtype)
    X_test = torch.randn(n_train_data, n_input).to(device).to(dtype)
    if likelihood == 'classification':
        Y_train = torch.randint(
            low=0, high=n_class, size=(n_train_data,), dtype=torch.int64
        ).to(device)
        Y_test = torch.randint(
            low=0, high=n_class, size=(n_train_data,), dtype=torch.int64
        ).to(device)
    else:
        Y_train = torch.randn(size=(n_train_data,n_class), 
                              dtype=dtype).to(device)
        Y_test = torch.randn(size=(n_train_data,n_class), 
                             dtype=dtype).to(device)
    
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


@pytest.mark.parametrize('init_data', ('classification','regression'), indirect=True)
def test_subset_projector(init_data: Tuple):
    model, train_loader, test_loader, X_test, device, likelihood = init_data
    submodel_method = 'magnitude'
    s = 10
    a_tol = 1e-5
    Ind = subset_indices(model=model, likelihood=likelihood,
                            train_loader=train_loader,
                            method=submodel_method)
    # test projector with theoretical value
    P = Ind.P(s).to(device)
    la = Laplace(model,
                likelihood=likelihood,
                subset_of_weights='all',
                hessian_structure='full')
    la.fit(train_loader=train_loader)
    subla = Laplace(model,
        likelihood=likelihood, 
        subset_of_weights="subnetwork",
        hessian_structure='full',
        subnetwork_indices=Ind(s))
    subla.fit(train_loader=train_loader)
    IPsi = FullInvPsi(inv_Psi=la.posterior_precision)
    J_X = get_jacobian(X=X_test, model=model, is_classification=False).to(device)
    sub_J_X = J_X.index_select(dim=-1, index=Ind(s))
    Sigma_P_analytical = compute_Sigma_P(P=P, IPsi=IPsi, J_X=J_X)
    Sigma_P_library = subla.functional_covariance(Js=sub_J_X)
    assert torch.allclose(Sigma_P_analytical, Sigma_P_library, atol=a_tol)

    # test subset property
    s_max = Ind.metric.size(0)
    s_test = torch.randint(low=1, high=s_max-1,size=(1,)).item()
    P_from_s_max = Ind.P(s_max)[:,:s_test]
    P_s_test = Ind.P(s_test)
    assert torch.equal(P_from_s_max, P_s_test)


@pytest.mark.parametrize('init_data', ['classification','regression'], indirect=True)
def test_subset_methods(init_data: Tuple):
    model, train_loader, test_loader, X_test, device, likelihood = init_data
    # diagonal
    subind = subset_indices(model=model, likelihood=likelihood,
                     train_loader=train_loader, method='diagonal')
    diag_laplace_model = laplace.Laplace(model=model,
                                    likelihood=likelihood,
                                    subset_of_weights='all',
                                    hessian_structure='diag')
    diag_laplace_model.fit(train_loader)
    assert torch.all(torch.isclose(diag_laplace_model.posterior_variance,
                                   subind.metric))

    # magnitude
    subind = subset_indices(model=model, likelihood=likelihood,
                     train_loader=train_loader, method='magnitude')
    parameter_vector = parameters_to_vector(model.parameters())
    assert torch.allclose(torch.abs(parameter_vector), subind.metric)


@pytest.mark.parametrize('init_data', ['regression', 'classification'], indirect=True)
def test_SWAG_subset_index_selection_and_posterior_precision(init_data: Tuple):
    """ 
    Test if for SWAG the index selection and the posterior precision of the
    laplace library and own method coincide. 
    """
    model, train_loader, test_loader, X_test, _, likelihood = init_data
    n_select = 10

    # index selection
    # laplace library
    subnetmask_kwargs = dict(
        model=model, 
        likelihood=likelihood,
        n_params_subnet=n_select)
    subnetmask = LargestVarianceSWAGSubnetMask(**subnetmask_kwargs)
    subnetmask.select(train_loader)
    idcs_select = subnetmask.indices

    # method to test
    idcs_fun = subset_indices(
        model=model, 
        likelihood=likelihood, 
        train_loader=train_loader, 
        method='swag', 
        swag_lr=1e-2, 
        swag_n_snapshots=40
    )
    idcs_select_to_test = idcs_fun(n_select, sort=True)
    print(idcs_select)
    print(idcs_select_to_test)
    assert torch.equal(idcs_select, idcs_select_to_test)

    # posterior precision
    # laplace library
    lasub = Laplace(model,
        likelihood=likelihood, 
        subset_of_weights="subnetwork",
        hessian_structure='full',
        subnetwork_indices=idcs_select.cpu()
    )
    lasub.fit(test_loader)

    # own method
    J = get_jacobian(
        model, 
        X_test,
        fun= (lambda x: x) if likelihood=="regression" else \
            (lambda x: 2 * torch.sqrt(x)),
        is_classification=likelihood=="classification",
    )
    I = torch.einsum("bck, bcl -> kl", J, J)
    inv_psi = torch.eye(I.shape[0]) + I

    P = idcs_fun.P(s=n_select, sort=True) 
    posterior_precision = P.T @ inv_psi @ P
    assert torch.allclose(posterior_precision, lasub.posterior_precision)




