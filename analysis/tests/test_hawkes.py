"""Unit tests for sentinel_analysis.hawkes process modules."""

import numpy as np
import pandas as pd
import pytest

from sentinel_analysis.data.fetch_binance_l2 import generate_mock_l2_data
from sentinel_analysis.data.preprocess import preprocess_l2_data
from sentinel_analysis.hawkes.kernels import ExponentialKernel, PowerLawKernel
from sentinel_analysis.hawkes.model import HawkesProcess


def test_exponential_kernel_math():
    kernel = ExponentialKernel(alpha=0.5, beta=1.5)
    # phi(0) = 0.5 * exp(0) = 0.5
    assert np.isclose(kernel.eval(0.0), 0.5)
    # phi(1) = 0.5 * exp(-1.5)
    assert np.isclose(kernel.eval(1.0), 0.5 * np.exp(-1.5))
    # integral_0^1 phi(s) ds = (0.5/1.5) * (1 - exp(-1.5))
    expected_int = (0.5 / 1.5) * (1.0 - np.exp(-1.5))
    assert np.isclose(kernel.integrate(1.0), expected_int)


def test_power_law_kernel_math():
    kernel = PowerLawKernel(alpha=0.8, gamma=2.5, delta=1.0)
    # phi(0) = 0.8 * (1.0)^(-2.5) = 0.8
    assert np.isclose(kernel.eval(0.0), 0.8)
    # integral_0^1 phi(s) ds = (0.8 / 1.5) * (1 - (2)^(-1.5))
    expected_int = (0.8 / 1.5) * (1.0 - np.power(2.0, -1.5))
    assert np.isclose(kernel.integrate(1.0), expected_int)


def test_invalid_kernel_params():
    with pytest.raises(ValueError):
        ExponentialKernel(alpha=-0.1, beta=1.0)
    with pytest.raises(ValueError):
        ExponentialKernel(alpha=0.5, beta=0.0)
    with pytest.raises(ValueError):
        PowerLawKernel(alpha=0.5, gamma=0.5)


def test_hawkes_intensity_scaling():
    k1 = ExponentialKernel(alpha=0.2, beta=1.0)
    hp_low = HawkesProcess(mu=0.5, kernels=k1)

    k2 = ExponentialKernel(alpha=0.2, beta=1.0)
    hp_high = HawkesProcess(mu=2.0, kernels=k2)

    events = [(0.1, 0), (0.5, 0)]
    assert hp_high.intensity(1.0, events) > hp_low.intensity(1.0, events)


def test_hawkes_simulation_event_counts_scaling():
    k = ExponentialKernel(alpha=0.3, beta=2.0)

    model_low = HawkesProcess(mu=0.5, kernels=k)
    events_low = model_low.simulate(T_end=100.0, rng=42)

    model_high = HawkesProcess(mu=2.5, kernels=k)
    events_high = model_high.simulate(T_end=100.0, rng=42)

    assert len(events_high) > len(events_low)


def test_log_likelihood_numerical_stability():
    kernel = ExponentialKernel(alpha=0.4, beta=1.2)
    model = HawkesProcess(mu=1.0, kernels=kernel)

    # 1. Synthetic events
    events = model.simulate(T_end=50.0, rng=123)
    ll_synth = model.log_likelihood(events, T_end=50.0)

    assert np.isfinite(ll_synth)

    # 2. Preprocessed real/mock market data
    depth_data, trades_data = generate_mock_l2_data("BTCUSDT")
    df, _, _ = preprocess_l2_data(depth_data, trades_data, dry_run=True)

    # Extract event timestamps as Hawkes input
    market_events = [(t, 0) for t in df["timestamp"].values]
    T_end = max(df["timestamp"].max(), 1.0)

    ll_market = model.log_likelihood(market_events, T_end=T_end)
    assert np.isfinite(ll_market)


def test_multivariate_hawkes_simulation_and_log_likelihood():
    # 2-dimension multivariate Hawkes process
    mu = [0.5, 0.8]
    k00 = ExponentialKernel(0.2, 1.0)
    k01 = ExponentialKernel(0.1, 1.5)
    k10 = ExponentialKernel(0.1, 1.5)
    k11 = ExponentialKernel(0.3, 1.0)
    kernels = [[k00, k01], [k10, k11]]

    mv_model = HawkesProcess(mu=mu, kernels=kernels)
    events = mv_model.simulate(T_end=50.0, rng=99)

    assert len(events) > 0
    # Check that events contain valid dimension indices (0 or 1)
    for t, dim in events:
        assert dim in (0, 1)

    ll = mv_model.log_likelihood(events, T_end=50.0)
    assert np.isfinite(ll)


def test_seeded_determinism():
    """Verify that running the simulation twice with identical seed produces identical event streams."""
    kernel = ExponentialKernel(alpha=0.5, beta=1.0)
    model = HawkesProcess(mu=1.2, kernels=kernel)

    events_seed1_a = model.simulate(T_end=100.0, rng=42)
    events_seed1_b = model.simulate(T_end=100.0, rng=42)

    assert events_seed1_a == events_seed1_b

    # Verify different seed produces different events
    events_seed2 = model.simulate(T_end=100.0, rng=43)
    assert events_seed1_a != events_seed2


def test_pipeline_reproducibility_smoke():
    """Smoke test: run full data preprocess and Hawkes simulation twice with same seed, assert identical output."""
    depth_data, trades_data = generate_mock_l2_data("BTCUSDT")

    df1, _, _ = preprocess_l2_data(depth_data, trades_data, seed=100, dry_run=True)
    df2, _, _ = preprocess_l2_data(depth_data, trades_data, seed=100, dry_run=True)

    pd.testing.assert_frame_equal(df1, df2)

    kernel = ExponentialKernel(alpha=0.3, beta=0.8)
    model = HawkesProcess(mu=0.8, kernels=kernel)

    sim1 = model.simulate(T_end=30.0, rng=100)
    sim2 = model.simulate(T_end=30.0, rng=100)

    assert sim1 == sim2
