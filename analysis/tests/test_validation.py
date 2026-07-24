"""Unit tests for Hawkes process goodness-of-fit validation module."""

import numpy as np

from sentinel_analysis.hawkes.kernels import ExponentialKernel
from sentinel_analysis.hawkes.model import HawkesProcess
from sentinel_analysis.hawkes.validation import (
    compute_transformed_residuals,
    evaluate_goodness_of_fit,
    plot_qq_residuals,
)


def test_transformed_residuals_computation():
    kernel = ExponentialKernel(alpha=0.3, beta=1.0)
    model = HawkesProcess(mu=0.5, kernels=kernel)

    events = model.simulate(T_end=100.0, rng=123)
    timestamps = np.array([e[0] for e in events])

    residuals = compute_transformed_residuals(model, timestamps)

    assert len(residuals) == len(timestamps) - 1
    assert np.all(residuals > 0)
    assert np.isfinite(np.sum(residuals))


def test_goodness_of_fit_flags_known_bad_fit():
    """Verify that validation correctly accepts good fit and rejects known-bad synthetic fit."""
    true_mu = 0.5
    true_alpha = 0.3
    true_beta = 1.2

    kernel_true = ExponentialKernel(alpha=true_alpha, beta=true_beta)
    model_true = HawkesProcess(mu=true_mu, kernels=kernel_true)

    events = model_true.simulate(T_end=200.0, rng=999)
    timestamps = np.array([e[0] for e in events])

    # 1. Good fit (true model)
    residuals_true = compute_transformed_residuals(model_true, timestamps)
    gof_good = evaluate_goodness_of_fit(residuals_true, alpha_threshold=0.01)

    assert gof_good.ks_pvalue > 0.001
    assert gof_good.is_valid_fit is True

    # 2. Known-bad fit (wildly wrong mu and alpha)
    kernel_bad = ExponentialKernel(alpha=5.0, beta=0.1)
    model_bad = HawkesProcess(mu=10.0, kernels=kernel_bad)

    residuals_bad = compute_transformed_residuals(model_bad, timestamps)
    gof_bad = evaluate_goodness_of_fit(residuals_bad, alpha_threshold=0.01)

    # Bad fit must be flagged as invalid with extremely low p-value
    assert gof_bad.ks_pvalue < 0.001
    assert gof_bad.is_valid_fit is False


def test_plot_qq_residuals_artifact_creation(tmp_path):
    residuals = np.random.default_rng(42).exponential(scale=1.0, size=100)
    output_png = tmp_path / "test_qq_plot.png"

    res_path = plot_qq_residuals(residuals, output_path=output_png)

    assert res_path.exists()
    assert res_path.stat().st_size > 0
