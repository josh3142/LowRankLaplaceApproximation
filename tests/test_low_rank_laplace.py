from math import ceil
from typing import Tuple, Iterable, Literal, Callable

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

import laplace
from laplace import KronLaplace, FullLaplace, DiagLaplace, Laplace
from laplace.utils import (
    LargestVarianceDiagLaplaceSubnetMask,
    LargestMagnitudeSubnetMask,
    LargestVarianceSWAGSubnetMask
)

from utils import flatten_batch_and_target_dimension
from projector.projector1d import get_jacobian, number_of_parameters_with_grad
from linearized_model.low_rank_laplace import (
    FullInvPsi,
    HalfInvPsi,
    DiagInvPsi,
    KronInvPsi,
    compute_Sigma,
    compute_optimal_P,
    compute_Sigma_P,
    compute_Sigma_s,
    IPsi_predictive,
)
from linearized_model.subset import subset_indices


@pytest.fixture
def init_data(seed=0, device=torch.device("cpu"), dtype=torch.float64) -> Callable:
    def create_data(
        likelihood: Literal["classification", "regression"] = "classification",
    ) -> Tuple:
        generator = torch.Generator().manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        # generate data
        n_train_data, n_test_data, n_input, n_target, batch_size = 100, 50, 2, 3, 50

        X_train = torch.randn(
            n_train_data, n_input, generator=generator, dtype=dtype
        ).to(device)
        X_test = torch.randn(n_test_data, n_input, generator=generator, dtype=dtype).to(
            device
        )
        if likelihood == "classification":
            Y_test = torch.randint(
                low=0,
                high=n_target,
                size=(n_test_data,),
                dtype=torch.int64,
                generator=generator,
            ).to(device)
            Y_train = torch.randint(
                low=0,
                high=n_target,
                size=(n_train_data,),
                dtype=torch.int64,
                generator=generator,
            ).to(device)
            model = (
                nn.Sequential(
                    nn.Linear(n_input, 5), nn.LeakyReLU(), nn.Linear(5, n_target)
                )
                .to(device)
                .to(dtype)
            )
        else:
            Y_test = torch.randn(
                size=(n_test_data, n_target), dtype=dtype, generator=generator
            ).to(device)
            Y_train = torch.randn(
                size=(n_train_data, n_target), dtype=dtype, generator=generator
            ).to(device)
            model = (
                nn.Sequential(
                    nn.Linear(n_input, 5), nn.LeakyReLU(), nn.Linear(5, n_target)
                )
                .to(device)
                .to(dtype)
            )

        prior_precision = 1.0

        # create dataloaders
        train_data = TensorDataset(X_train, Y_train)
        test_data = TensorDataset(X_test, Y_test)
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, generator=generator
        )
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=False, generator=generator
        )
        # create model

        return (
            model,
            train_loader,
            test_loader,
            X_test,
            X_train,
            device,
            generator,
            dtype,
            prior_precision,
        )

    return create_data


@pytest.fixture
def random_inv_psi(n_data=50, n_class=3, n_par=4, seed=1, dtype=torch.float64) -> Tuple:
    generator = torch.Generator().manual_seed(seed)
    J_X = torch.randn(n_data, n_class, n_par, generator=generator, dtype=dtype)
    V = torch.randn(n_data * n_class, n_par, generator=generator, dtype=dtype)
    prior_precision = 1.0
    inv_Psi = V.T @ V + prior_precision * torch.eye(n_par)
    return J_X, V, inv_Psi, prior_precision, generator, dtype


def test_FullInvPsi(random_inv_psi):
    J_X, V, inv_Psi, prior_precision, generator, dtype = random_inv_psi
    IPsi = FullInvPsi(inv_Psi=inv_Psi)
    Psi = torch.linalg.inv(inv_Psi)
    W = flatten_batch_and_target_dimension(
        torch.randn(J_X.shape, generator=generator, dtype=dtype)
    ).T

    # test Psi @ W
    theoretical_value = Psi @ W
    assert torch.all(torch.isclose(IPsi.Psi_times_W(W=W), theoretical_value))

    # test Sigma = J_X @ Psi @ J_X.T
    flat_J_X = flatten_batch_and_target_dimension(J_X)
    theoretical_value = flat_J_X @ Psi @ flat_J_X.T
    assert torch.allclose(IPsi.Sigma(J_X), theoretical_value)

    # test Sigma_batchwise
    # shorten for computational speed
    short_length = 3
    short_J_X = J_X[:short_length]
    # batch x target x parameters
    assert len(short_J_X.shape) == 3
    theoretical_value = []
    for b in range(short_length):
        theoretical_value.append(IPsi.Sigma(short_J_X[b]))
    theoretical_value = torch.stack(theoretical_value, dim=0)
    computed_value = IPsi.Sigma_batchwise(short_J_X)
    assert torch.allclose(theoretical_value, computed_value)

    # test quadratic form W @ inv_Psi @ W
    theoretical_value = W.T @ inv_Psi @ W
    assert torch.all(torch.isclose(IPsi.quadratic_form(W=W), theoretical_value))

    # test SVD of sigma
    theoretical_value = flat_J_X @ Psi @ flat_J_X.T
    U, Lamb = IPsi.Sigma_svd(J_X)
    assert torch.allclose(U @ torch.diag(Lamb) @ U.T, theoretical_value, atol=1e-5)


def test_HalfInvPsi(random_inv_psi):
    J_X, V, inv_Psi, prior_precision, generator, dtype = random_inv_psi
    IPsi = HalfInvPsi(V=V, prior_precision=1.0)
    Psi = torch.linalg.inv(inv_Psi)
    W = flatten_batch_and_target_dimension(
        torch.randn(J_X.shape, generator=generator, dtype=dtype)
    ).T

    # test Psi @ W
    theoretical_value = Psi @ W
    assert torch.all(torch.isclose(IPsi.Psi_times_W(W=W), theoretical_value))

    # test Sigma = J_X @ Psi @ J_X.T
    flat_J_X = flatten_batch_and_target_dimension(J_X)
    theoretical_value = flat_J_X @ Psi @ flat_J_X.T
    assert torch.all(torch.isclose(IPsi.Sigma(J_X), theoretical_value, atol=1e-5))

    # test Sigma_batchwise
    # shorten for computational speed
    short_length = 3
    short_J_X = J_X[:short_length]
    # batch x target x parameters
    assert len(short_J_X.shape) == 3
    theoretical_value = []
    for b in range(short_length):
        theoretical_value.append(IPsi.Sigma(short_J_X[b]))
    theoretical_value = torch.stack(theoretical_value, dim=0)
    computed_value = IPsi.Sigma_batchwise(short_J_X)
    assert torch.allclose(theoretical_value, computed_value)
    # test quadratic form W @ inv_Psi @ W
    theoretical_value = W.T @ inv_Psi @ W
    assert torch.all(
        torch.isclose(IPsi.quadratic_form(W=W), theoretical_value, atol=1e-5)
    )

    # test SVD of sigma
    theoretical_value = flat_J_X @ Psi @ flat_J_X.T
    U, Lamb = IPsi.Sigma_svd(J_X)
    assert torch.all(
        torch.isclose(U @ torch.diag(Lamb) @ U.T, theoretical_value, atol=1e-5)
    )

    # test iterator version
    def create_V_iterator(batch_size: int = int(V.size(0) / 10) + 1):
        number_of_batches = ceil(V.size(0) / batch_size)
        for i in range(number_of_batches):
            yield V[i * batch_size : (i + 1) * batch_size]

    it_IPsi = HalfInvPsi(V=create_V_iterator, prior_precision=prior_precision)
    # test full V
    assert torch.all(torch.isclose(it_IPsi.full_V, IPsi.full_V))
    # test quadratic form
    assert torch.all(
        torch.isclose(it_IPsi.quadratic_form(W=W), IPsi.quadratic_form(W=W))
    )

@pytest.mark.parametrize("likelihood", ["classification", "regression"])
def test_DiagInvPsi(init_data, likelihood):
    (
        model,
        train_loader,
        test_loader,
        X_test,
        X_train,
        device,
        generator,
        dtype,
        prior_precision,
    ) = init_data(likelihood)
    J_X = get_jacobian(X=X_test, model=model, is_classification=False).to(device)
    # fit diagonal Laplace approximation
    la = laplace.Laplace(
        model=model,
        likelihood=likelihood,
        subset_of_weights="all",
        hessian_structure="diag",
        prior_precision=prior_precision,
    )
    assert type(la) is DiagLaplace
    la.fit(train_loader)

    IPsi = DiagInvPsi(inv_Psi=la)
    inv_Psi = torch.diag(la.posterior_precision)
    Psi = torch.linalg.inv(inv_Psi)

    # test wether precision and variance behave as expected
    assert torch.allclose(1/IPsi.posterior_precision,IPsi.posterior_variance)

    # define vectors W1, W2 for testing with other vectors than J_X
    W1 = torch.randn(la.n_params, 3, 9, generator=generator, dtype=dtype)
    W2 = torch.randn(la.n_params, 12, generator=generator, dtype=dtype)

    # test Psi @ W
    theoretical_value = torch.tensordot(Psi, W1, dims=([-1], [0]))
    computed_value = IPsi.Psi_times_W(W=W1)
    assert torch.all(torch.isclose(theoretical_value, computed_value))

    # test Sigma
    # theoretical_value
    theoretical_value = W2.T @ inv_Psi @ W2
    computed_value = IPsi.quadratic_form(W=W2)
    assert torch.all(torch.isclose(theoretical_value, computed_value))

    # test Sigma_batchwise
    # shorten for computational speed
    short_length = 3
    short_J_X = J_X[:short_length]
    # batch x target x parameters
    assert len(short_J_X.shape) == 3
    theoretical_value = []
    for b in range(short_length):
        theoretical_value.append(IPsi.Sigma(short_J_X[b]))
    theoretical_value = torch.stack(theoretical_value, dim=0)
    computed_value = IPsi.Sigma_batchwise(short_J_X)
    assert torch.allclose(theoretical_value, computed_value)
    
    

@pytest.mark.parametrize("likelihood", ["classification", "regression"])
def test_KronInvPsi(init_data, likelihood):
    (
        model,
        train_loader,
        test_loader,
        X_test,
        X_train,
        device,
        generator,
        dtype,
        prior_precision,
    ) = init_data(likelihood)
    J_X = get_jacobian(X=X_test, model=model, is_classification=False).to(device)
    # fit KFAC Laplace approximation
    la = laplace.Laplace(
        model=model,
        likelihood=likelihood,
        subset_of_weights="all",
        hessian_structure="kron",
        prior_precision=prior_precision,
    )
    assert type(la) is KronLaplace
    la.fit(train_loader)

    IPsi = KronInvPsi(inv_Psi=la)
    inv_Psi = la.posterior_precision.to_matrix()
    Psi = torch.linalg.inv(inv_Psi)
    # define vectors W1, W2 for testing with other vectors than J_X
    W1 = torch.randn(la.n_params, 3, 9, generator=generator, dtype=dtype)
    W2 = torch.randn(la.n_params, 12, generator=generator, dtype=dtype)

    # test Psi @ W
    theoretical_value = torch.tensordot(Psi, W1, dims=([-1], [0]))
    computed_value = IPsi.Psi_times_W(W=W1)
    assert torch.all(torch.isclose(theoretical_value, computed_value))

    # test Sigma
    # theoretical_value
    theoretical_value = W2.T @ inv_Psi @ W2
    computed_value = IPsi.quadratic_form(W=W2)
    assert torch.all(torch.isclose(theoretical_value, computed_value))

    # test Sigma_batchwise
    # shorten for computational speed
    short_length = 3
    short_J_X = J_X[:short_length]
    # batch x target x parameters
    assert len(short_J_X.shape) == 3
    theoretical_value = []
    for b in range(short_length):
        theoretical_value.append(IPsi.Sigma(short_J_X[b]))
    theoretical_value = torch.stack(theoretical_value, dim=0)
    computed_value = IPsi.Sigma_batchwise(short_J_X)
    assert torch.allclose(theoretical_value, computed_value)


@pytest.mark.parametrize("likelihood", ["classification", "regression"])
def test_Sigma_iterative_method(
    init_data: Callable,
    random_inv_psi: Tuple,
    likelihood: Literal["classification", "regression"],
):
    (
        model,
        train_loader,
        test_loader,
        X_test,
        X_train,
        device,
        generator,
        dtype,
        prior_precision,
    ) = init_data(likelihood)
    J_X = get_jacobian(X=X_test, model=model).to(device)

    def create_J_X_iterator(J_X=J_X, batch_size=11) -> Iterable:
        number_of_batches = ceil(J_X.size(0) / batch_size)
        for i in range(number_of_batches):
            yield J_X[i * batch_size : (i + 1) * batch_size]

    # test using KFAC approach
    la = laplace.Laplace(
        model=model,
        likelihood=likelihood,
        subset_of_weights="all",
        hessian_structure="kron",
        prior_precision=prior_precision,
    )
    assert type(la) is KronLaplace
    la.fit(train_loader)
    IPsi = KronInvPsi(inv_Psi=la)

    computed_value = IPsi.Sigma_iterative(create_J_X_iterator=create_J_X_iterator)
    theoretical_value = IPsi.Sigma(J_X=J_X)
    assert torch.all(torch.isclose(computed_value, theoretical_value))

    # test using Full approach
    J_X, V, inv_Psi, prior_precision, generator, dtype = random_inv_psi
    IPsi = FullInvPsi(inv_Psi=inv_Psi)
    computed_value = IPsi.Sigma_iterative(
        create_J_X_iterator=lambda: create_J_X_iterator(J_X=J_X)
    )
    theoretical_value = IPsi.Sigma(J_X=J_X)
    assert torch.all(torch.isclose(computed_value, theoretical_value))
    # test using half approach
    IPsi = HalfInvPsi(V=V, prior_precision=prior_precision)
    computed_value = IPsi.Sigma_iterative(
        create_J_X_iterator=lambda: create_J_X_iterator(J_X=J_X)
    )
    theoretical_value = IPsi.Sigma(J_X=J_X)
    assert torch.all(torch.isclose(computed_value, theoretical_value))


def test_compute_Sigma(random_inv_psi: Tuple):
    J_X, V, inv_Psi, prior_precision, generator, dtype = random_inv_psi
    J_X = flatten_batch_and_target_dimension(J_X)
    IPsi = FullInvPsi(inv_Psi)
    theoretical_value = J_X @ torch.linalg.inv(inv_Psi) @ J_X.T
    computed_value = compute_Sigma(IPsi=IPsi, J_X=J_X)
    assert torch.all(torch.isclose(theoretical_value, computed_value))


@pytest.mark.parametrize("likelihood", ["regression", "classification"])
def test_optimal_P(
    init_data: Callable, likelihood: Literal["regression", "classification"]
):
    (
        model,
        train_loader,
        test_loader,
        X_test,
        X_train,
        device,
        generator,
        dtype,
        prior_precision,
    ) = init_data(likelihood)
    J_X = flatten_batch_and_target_dimension(
        get_jacobian(X=X_test, model=model).to(device)
    )

    def create_J_X_iterator(J_X=J_X, batch_size=11) -> Iterable:
        number_of_batches = ceil(J_X.size(0) / batch_size)
        for i in range(number_of_batches):
            yield J_X[i * batch_size : (i + 1) * batch_size]

    # test using KFAC approach
    la = laplace.Laplace(
        model=model,
        likelihood=likelihood,
        subset_of_weights="all",
        hessian_structure="kron",
        prior_precision=prior_precision,
    )
    assert type(la) is KronLaplace
    la.fit(train_loader)
    IPsi = KronInvPsi(inv_Psi=la)
    inv_Psi = la.posterior_precision.to_matrix()
    s_max = min(J_X.size(0), J_X.size(1))
    s_list = [1, int(s_max / 2), s_max]
    U, Lamb = IPsi.Sigma_svd(J_X=J_X)
    P = compute_optimal_P(IPsi=IPsi, J_X=J_X, U=U, s=s_max)
    for s, Sigma_P_s in zip(
        s_list, compute_Sigma_P(P=P, IPsi=IPsi, J_X=J_X, s_iterable=s_list)()
    ):
        P_s = P[:, :s]
        theoretical_Sigma_P_s = (
            J_X @ P_s @ torch.linalg.inv(P_s.T @ inv_Psi @ P_s) @ P_s.T @ J_X.T
        )
        # assert torch.all(torch.isclose(Sigma_P_s,theoretical_Sigma_P_s))
        assert torch.all(
            torch.isclose(
                theoretical_Sigma_P_s, compute_Sigma_P(P=P_s, IPsi=IPsi, J_X=J_X)
            )
        )
        assert torch.all(torch.isclose(Sigma_P_s, compute_Sigma_s(U=U, Lamb=Lamb, s=s)))

    full_Sigma = compute_Sigma(IPsi=IPsi, J_X=create_J_X_iterator)
    assert torch.all(torch.isclose(full_Sigma, Sigma_P_s))


@pytest.mark.parametrize("hessian_structure", ["kron", "full"])
@pytest.mark.parametrize("likelihood", ["regression", "classification"])
def test_glm_predictive(
    init_data: Callable,
    likelihood: Literal["regression", "classification"],
    hessian_structure: Literal["kron", "full"],
):
    (
        model,
        train_loader,
        test_loader,
        X_test,
        X_train,
        device,
        generator,
        dtype,
        prior_precision,
    ) = init_data(likelihood)

    # test using KFAC approach
    la = laplace.Laplace(
        model=model,
        likelihood=likelihood,
        subset_of_weights="all",
        hessian_structure=hessian_structure,
        prior_precision=prior_precision,
    )
    la.fit(train_loader)
    if hessian_structure == "kron":
        assert type(la) is KronLaplace
        IPsi = KronInvPsi(inv_Psi=la)
        IPsi.inv_Psi.posterior_precision.to_matrix()
    elif hessian_structure == "full":
        assert type(la) is FullLaplace
        IPsi = FullInvPsi(inv_Psi=la.posterior_precision.to(dtype))
    else:
        raise NotImplementedError
    J_X_train = get_jacobian(model=model, X=X_train, is_classification=False)
    U, _ = IPsi.Sigma_svd(J_X=J_X_train)
    s = int(min(U.size(0), J_X_train.size(-1)) / 2)
    P = compute_optimal_P(
        IPsi=IPsi, J_X=flatten_batch_and_target_dimension(J_X_train), U=U, s=s
    )
    predictions, variances = IPsi_predictive(model=model, IPsi=IPsi, P=P)(
        X=X_test
    )
    assert len(predictions) == len(variances)
    J_X_test = get_jacobian(model=model, X=X_test, is_classification=False)
    theoretical_variances = []
    for j in J_X_test:
        theoretical_variances.append(compute_Sigma_P(P=P, IPsi=IPsi, J_X=j[None, ...]))
    theoretical_variances = torch.stack(theoretical_variances, dim=0)
    assert torch.allclose(theoretical_variances, variances)

@pytest.mark.parametrize("subset_method", ["diag", "magnitude", "swag"])
@pytest.mark.parametrize("likelihood", ["regression"])
def test_subset_predictive_variances_via_laplace_library(
    init_data: Callable,
    likelihood: Literal["regression", "classification"],
    subset_method: Literal["diag", "magnitude", "swag"],
):
    (
        model,
        train_loader,
        test_loader,
        X_test,
        X_train,
        device,
        generator,
        dtype,
        prior_precision,
    ) = init_data(likelihood)
    sigma_noise = 1.0
    ggn_la = FullLaplace(
        model=model,
        likelihood=likelihood,
        sigma_noise=sigma_noise,
    )
    ggn_la.fit(train_loader)
    IPsi = FullInvPsi(inv_Psi=ggn_la.posterior_precision.to(dtype))
    Ind = subset_indices(
        model=model,
        likelihood=likelihood,
        train_loader=train_loader,
        method=subset_method,
    )
    s_max = number_of_parameters_with_grad(model)
    s_list = [int(s_max/3), int(s_max/2)]
    P = Ind.P(s=s_max).to(dtype)
    predictive = IPsi_predictive(
        model=model,
        IPsi=IPsi,
        P=P,
        regression_likelihood_sigma=sigma_noise,
    )
    for s in s_list:
        predictions, variances = predictive(X=X_test, s=s, eps=0.0)
        if subset_method == 'diag':
            diag_laplace_model = laplace.Laplace(
                model=model,
                likelihood=likelihood,
                subset_of_weights='all',
                hessian_structure='diag',
                sigma_noise=sigma_noise,
            )
            subnetwork_mask = LargestVarianceDiagLaplaceSubnetMask(
                model=model,
                n_params_subnet=s,
                diag_laplace_model=diag_laplace_model,
            )
            subnetwork_mask.select(train_loader=train_loader)
            subnetwork_indices = subnetwork_mask.indices
        elif subset_method == 'magnitude':
            subnetwork_mask = LargestMagnitudeSubnetMask(
                model=model,
                n_params_subnet=s
            )
            subnetwork_mask.select(train_loader=train_loader)
            subnetwork_indices = subnetwork_mask.indices
        elif subset_method == 'swag':
            # use the Ind object from above to ignore
            # random influences in the swag index selection
            subnetwork_indices = Ind(s)
        else:
            raise NotImplementedError
        la = Laplace(
            model=model,
            likelihood=likelihood,
            subset_of_weights="subnetwork",
            hessian_structure="full",
            subnetwork_indices=subnetwork_indices,
            sigma_noise=sigma_noise,
        )
        la.fit(train_loader)
        theoretical_predictions, theoretical_variances = la.__call__(X_test)
        # correct with sigma noise
        theoretical_variances += sigma_noise**2 \
            * torch.eye(theoretical_variances.size(1)).unsqueeze(0).repeat(
                theoretical_variances.size(0), 1, 1
            )

        assert torch.allclose(predictions, theoretical_predictions)
        assert torch.allclose(variances, theoretical_variances)

        


