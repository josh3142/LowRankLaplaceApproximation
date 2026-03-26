import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from scipy.stats import chi2

from linearized_model.approximation_metrics import get_coverage, get_calibration_error


@pytest.fixture
def init_regression_data_1d() -> Tuple:
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    
    N = 500
    n_input = 2
    n_output = 2
    
    X = torch.randn(N, n_input)
    
    model = nn.Sequential(
        nn.Linear(n_input, 10),
        nn.ReLU(),
        nn.Linear(10, n_output)  # Changed to n_output
    )
    
    with torch.no_grad():
        predictions = model(X)  # (N, 2)
    
    # Random diagonal covariances
    variances = torch.zeros(N, n_output, n_output)
    for i in range(N):
        # Create diagonal matrix with random positive values
        diag_vals = torch.rand(n_output) * 2.0 + 0.5  # [0.5, 2.5]
        variances[i] = torch.diag(diag_vals)
    # Generate targets from predictive distribution
    # Sample from multivariate normal for each sample
    targets = torch.zeros(N, n_output)
    for i in range(N):
        dist = torch.distributions.MultivariateNormal(
            predictions[i], variances[i]
        )
        targets[i] = dist.sample()
    
    return model, predictions, variances, targets, N


@pytest.fixture
def init_regression_data_1d_perfect() -> Tuple:
    """
    Create perfectly calibrated data where we know the true distribution.
    
    For a standard normal, we know exactly what the coverage should be.
    """
    torch.manual_seed(123)
    torch.use_deterministic_algorithms(True)
    
    N = 1000
    n_output = 2
    
    # Predictions at zero with unit variance
    predictions = torch.zeros(N, n_output)
    variances = torch.eye(n_output).unsqueeze(0).repeat(N, 1, 1)
    
    # Targets from standard normal (perfectly calibrated)
    targets = torch.randn(N, n_output)
    
    return predictions, variances, targets, N


def test_coverage_basic_1d(init_regression_data_1d):
    """
    Test that get_coverage runs without errors and returns reasonable value.
    
    For randomly generated data where targets come from the predicted distribution,
    coverage should be close to (1 - alpha).
    """
    _, predictions, variances, targets, N = init_regression_data_1d
    
    alpha = 0.05  # 95% confidence
    coverage = get_coverage(predictions, variances, targets, alpha=alpha).mean().item()
        
    # Since targets are generated from the predictive distribution,
    # coverage should be close to 0.95
    # Allow generous margin due to finite sample size
    assert 0.85 <= coverage <= 1.0, f"Coverage {coverage} outside expected range"


def test_coverage_perfect_calibration_1d(init_regression_data_1d_perfect):
    """
    Test coverage on perfectly calibrated data (standard normal).
    
    For N(0, 1) distribution, we know the exact coverage at different alphas.
    """
    predictions, variances, targets, N = init_regression_data_1d_perfect
    
    # Test different confidence levels
    test_cases = [
        (0.05, 0.95),
        (0.10, 0.90),
        (0.32, 0.68),
    ]
    
    for alpha, expected_coverage in test_cases:
        coverage = get_coverage(predictions, variances, targets, alpha=alpha)
        coverage = coverage.mean().item()
        
        # With N=1000, standard error ≈ sqrt(p(1-p)/N) ≈ 0.015
        # Allow 3 standard errors (99.7% confidence)
        margin = 3 * np.sqrt(expected_coverage * (1 - expected_coverage) / N)
        
        assert abs(coverage - expected_coverage) < margin, \
            f"Coverage {coverage} differs from expected {expected_coverage} by more than {margin}"


def test_coverage_extreme_cases_1d():
    """Test coverage in extreme cases."""
    torch.manual_seed(999)
    N = 100
    
    # Case 1: Very tight variance (almost deterministic)
    # Coverage should be very low because almost no points fall in tiny ellipsoid
    predictions = torch.zeros(N, 1)
    variances = torch.ones(N, 1, 1) * 1e-6  # Very small variance
    targets = torch.randn(N, 1) * 2.0  # Large spread in targets
    
    coverage_tight = get_coverage(predictions, variances, targets, alpha=0.05)
    coverage_tight = coverage_tight.mean().item()
    assert coverage_tight < 0.2, f"Coverage {coverage_tight} too high for tight variance"
    
    # Case 2: Very large variance (very uncertain)
    # Coverage should be very high because almost all points fall in huge ellipsoid
    predictions = torch.zeros(N, 1)
    variances = torch.ones(N, 1, 1) * 1e6  # Very large variance
    targets = torch.randn(N, 1)  # Normal spread
    
    coverage_loose = get_coverage(predictions, variances, targets, alpha=0.05)
    coverage_loose = coverage_loose.mean().item()
    assert coverage_loose > 0.99, f"Coverage {coverage_loose} too low for large variance"


def test_calibration_error_basic_1d(init_regression_data_1d):
    """
    Test that get_calibration_error runs without errors and returns reasonable value.
    """
    _, predictions, variances, targets, N = init_regression_data_1d
    
    calib_error = get_calibration_error(predictions, variances, targets, n_bins=10)
    
    # Basic checks
    assert isinstance(calib_error, float)
    assert calib_error >= 0.0  # Calibration error is always non-negative
    
    # Should be reasonable (not too large)
    # Since data is generated from the model, calibration should be decent
    assert calib_error < 0.5, f"Calibration error {calib_error} unreasonably large"


def test_calibration_error_perfect_1d(init_regression_data_1d_perfect):
    """
    Test calibration error on perfectly calibrated data.
    
    For standard normal data, calibration error should be very small.
    """
    predictions, variances, targets, N = init_regression_data_1d_perfect
    
    calib_error = get_calibration_error(predictions, variances, targets, n_bins=20)
    
    # For perfectly calibrated data with N=1000, 
    # calibration error should be small (< 0.05)
    # Some error expected due to finite sample
    assert calib_error < 0.05, \
        f"Calibration error {calib_error} too large for perfectly calibrated data"


def test_calibration_error_miscalibrated_1d():
    """
    Test calibration error detects miscalibration.
    
    Create intentionally miscalibrated data and verify high calibration error.
    """
    torch.manual_seed(456)
    N = 500
    
    # Case 1: Overconfident (variance too small)
    predictions = torch.zeros(N, 1)
    variances = torch.ones(N, 1, 1) * 0.1  # Variance = 0.1, true std = sqrt(0.1) ≈ 0.32
    targets = torch.randn(N, 1) * 1.0  # True std = 1.0 (much larger!)
    
    calib_error_overconf = get_calibration_error(
        predictions, variances, targets, n_bins=10
    )
    
    # Should detect that model is overconfident
    assert calib_error_overconf > 0.1, \
        f"Failed to detect overconfidence: calibration error {calib_error_overconf}"
    
    # Case 2: Underconfident (variance too large)
    predictions = torch.zeros(N, 1)
    variances = torch.ones(N, 1, 1) * 10.0  # Variance = 10.0, true std ≈ 3.16
    targets = torch.randn(N, 1) * 0.5  # True std = 0.5 (much smaller!)
    
    calib_error_underconf = get_calibration_error(
        predictions, variances, targets, n_bins=10
    )
    
    # Should detect that model is underconfident
    assert calib_error_underconf > 0.1, \
        f"Failed to detect underconfidence: calibration error {calib_error_underconf}"


def test_calibration_error_consistency_1d(init_regression_data_1d_perfect):
    """
    Test that calibration error is consistent with coverage.
    
    If calibration error is small, coverage at 95% should be close to 0.95.
    """
    predictions, variances, targets, N = init_regression_data_1d_perfect
    
    calib_error = get_calibration_error(predictions, variances, targets, n_bins=20)
    coverage_95 = get_coverage(predictions, variances, targets, alpha=0.05).mean().item()
    
    # If calibration error is small, coverage should be close to expected
    if calib_error < 0.03:
        assert abs(coverage_95 - 0.95) < 0.05, \
            f"Small calibration error {calib_error} but coverage {coverage_95} far from 0.95"



def test_n_bins_effect_1d(init_regression_data_1d_perfect):
    """Test that calibration error is not overly sensitive to n_bins."""
    predictions, variances, targets, N = init_regression_data_1d_perfect
    
    calib_errors = []
    for n_bins in [5, 10, 20, 30]:
        calib_error = get_calibration_error(
            predictions, variances, targets, n_bins=n_bins
        )
        calib_errors.append(calib_error)
    
    # All should be similarly small for well-calibrated data
    assert all(ce < 0.05 for ce in calib_errors), \
        f"Calibration errors vary too much with n_bins: {calib_errors}"
    
    # Should not vary wildly
    assert np.std(calib_errors) < 0.02, \
        f"Calibration error too sensitive to n_bins: std={np.std(calib_errors)}"