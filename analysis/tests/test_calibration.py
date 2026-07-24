"""Unit tests for Hawkes process MLE calibration module."""

import numpy as np

from sentinel_analysis.hawkes.calibration import (
    calibrate_hawkes,
    fast_hawkes_neg_log_likelihood,
)
from sentinel_analysis.hawkes.kernels import ExponentialKernel
from sentinel_analysis.hawkes.model import HawkesProcess


def test_fast_neg_log_likelihood():
    kernel = ExponentialKernel(alpha=0.4, beta=1.2)
    model = HawkesProcess(mu=1.0, kernels=kernel)

    events = model.simulate(T_end=30.0, rng=42)
    timestamps = np.array([e[0] for e in events])

    nll_fast = fast_hawkes_neg_log_likelihood(
        params=np.array([1.0, 0.4, 1.2]),
        event_times=timestamps,
        T_end=30.0,
    )
    ll_slow = model.log_likelihood(events=events, T_end=30.0)

    # Verify that fast O(N) recursive log-likelihood matches model log-likelihood
    assert np.isclose(-nll_fast, ll_slow, rtol=1e-5)


def test_calibration_on_synthetic_data():
    true_mu = 0.8
    true_alpha = 0.35
    true_beta = 1.1

    kernel = ExponentialKernel(alpha=true_alpha, beta=true_beta)
    model = HawkesProcess(mu=true_mu, kernels=kernel)

    events = model.simulate(T_end=300.0, rng=100)
    timestamps = np.array([e[0] for e in events])

    result = calibrate_hawkes(events=timestamps, T_end=300.0, seed=100)

    assert result.convergence_success is True
    assert result.n_events == len(timestamps)
    assert result.branching_ratio < 1.0

    # Fitted parameters should be within reasonable proximity to true values
    assert np.isclose(result.fitted_mu, true_mu, atol=0.3)
    assert np.isclose(result.fitted_alpha, true_alpha, atol=0.2)
    assert np.isclose(result.fitted_beta, true_beta, atol=0.4)

    # Standard errors and CIs should be positive/valid
    assert result.mu_se > 0
    assert result.alpha_se > 0
    assert result.beta_se > 0
    assert result.mu_ci_95[0] <= result.fitted_mu <= result.mu_ci_95[1]


def test_calibration_seeded_determinism():
    """Verify that running calibration twice with identical seed and data yields identical results."""
    kernel = ExponentialKernel(alpha=0.4, beta=1.5)
    model = HawkesProcess(mu=0.5, kernels=kernel)

    events = model.simulate(T_end=150.0, rng=77)
    timestamps = np.array([e[0] for e in events])

    res1 = calibrate_hawkes(events=timestamps, T_end=150.0, seed=42)
    res2 = calibrate_hawkes(events=timestamps, T_end=150.0, seed=42)

    assert res1.fitted_mu == res2.fitted_mu
    assert res1.fitted_alpha == res2.fitted_alpha
    assert res1.fitted_beta == res2.fitted_beta
    assert res1.log_likelihood == res2.log_likelihood
