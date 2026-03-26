import numpy as np
import scipy
import torch
from torch import Tensor

from linearized_model.approximation_metrics import W2_multivariate_normal

def random_positive_definite(dim, eps, seed=None):
    if seed is not None:
        torch.manual_seed(seed)  

    A = torch.rand(dim, dim)
    matrix = A @ A.T + eps * torch.eye(dim)

    eigvals = torch.linalg.eigvalsh(matrix)
    assert torch.all(eigvals > 0), "Not all eigenvalues are positive."

    return matrix


def W2_with_scipy(
        Sigma_approx: Tensor,
        Sigma: Tensor,
) -> float:
    """Computes the Wasserstein-2-distance between two multivariate normals 
    with equal mean, but different covariances `Sigma_approx` and `Sigma`.

    Args:
        Sigma_approx (torch.Tensor): covariance of approximate distribution
        Sigma (torch.Tensor): covariance of reference distribution

    Returns:
        float: W2 distance
    """
    Sigma = Sigma.cpu().numpy()
    Sigma_approx = Sigma_approx.cpu().numpy()
    sqrt_Sigma = scipy.linalg.sqrtm(Sigma).real
    return np.trace(
        Sigma + Sigma_approx - 2 * scipy.linalg.sqrtm(
            sqrt_Sigma @ Sigma_approx @ sqrt_Sigma
        ).real
    ).item()

def test_W2():
    dim, eps = 10, 1
    Sigma_approx = random_positive_definite(dim, eps, 0)
    Sigma = random_positive_definite(dim, eps, 42)
    w2_test = W2_multivariate_normal(Sigma_approx, Sigma)
    w2_val = W2_with_scipy(Sigma_approx, Sigma)
    
    assert np.isclose(w2_test, w2_val, atol=1e-5)
