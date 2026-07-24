"""Goodness-of-Fit Validation for Hawkes Process Models (Time-Rescaling Theorem)."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from sentinel_analysis.hawkes.kernels import ExponentialKernel
from sentinel_analysis.hawkes.model import HawkesProcess


@dataclass
class GoodnessOfFitResult:
    """Dataclass holding complete goodness-of-fit validation metrics."""

    ks_stat: float
    ks_pvalue: float
    ljung_box_stat: float
    ljung_box_pvalue: float
    mean_residual: float
    std_residual: float
    is_valid_fit: bool
    qq_plot_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_transformed_residuals(
    model: HawkesProcess,
    events: np.ndarray | list[float],
) -> np.ndarray:
    """Compute transformed residuals tau_i = integral_{t_{i-1}}^{t_i} lambda(s) ds via Time-Rescaling Theorem.

    If the Hawkes process fit is accurate, tau_i ~ i.i.d. Exponential(1).

    Parameters:
        model (HawkesProcess): Fitted HawkesProcess instance.
        events (Union[np.ndarray, list[float]]): Sorted array of event timestamps.

    Returns:
        np.ndarray: Array of transformed residuals tau_i for i = 1..N-1.
    """
    event_times = np.sort(np.asarray(events, dtype=np.float64))
    N = len(event_times)
    if N < 2:
        return np.array([], dtype=np.float64)

    mu = float(model.mu[0])
    kernel = model.kernels[0][0]
    if isinstance(kernel, ExponentialKernel):
        alpha = kernel.alpha
        beta = kernel.beta
    else:
        alpha = getattr(kernel, "alpha", 0.5)
        beta = getattr(kernel, "beta", 1.0)

    residuals = np.zeros(N - 1, dtype=np.float64)

    # Recursive state for R(i-1) = sum_{j < i-1} exp(-beta * (t_{i-1} - t_j))
    R_prev = 0.0

    for i in range(1, N):
        t_prev = event_times[i - 1]
        t_curr = event_times[i]
        dt = t_curr - t_prev

        if i > 1:
            dt_prev = t_prev - event_times[i - 2]
            R_prev = np.exp(-beta * dt_prev) * (1.0 + R_prev)
        else:
            R_prev = 0.0

        # Compensator integral over [t_{i-1}, t_i]:
        # int_{t_{i-1}}^{t_i} lambda(s) ds = mu * dt + (alpha / beta) * (1 + R_prev) * (1 - exp(-beta * dt))
        if beta > 0:
            integrated = mu * dt + (alpha / beta) * (1.0 + R_prev) * (
                1.0 - np.exp(-beta * dt)
            )
        else:
            integrated = mu * dt

        residuals[i - 1] = max(1e-12, float(integrated))

    return residuals


def compute_ljung_box_test(
    residuals: np.ndarray, lags: int = 10
) -> tuple[float, float]:
    """Perform Ljung-Box test for residual autocorrelation/independence."""
    N = len(residuals)
    if N <= lags + 1:
        return 0.0, 1.0

    mean_res = np.mean(residuals)
    denom = np.sum((residuals - mean_res) ** 2)
    if denom == 0:
        return 0.0, 1.0

    q_stat = 0.0
    for k in range(1, lags + 1):
        num = np.sum((residuals[:-k] - mean_res) * (residuals[k:] - mean_res))
        rk = num / denom
        q_stat += (rk**2) / (N - k)

    q_stat *= N * (N + 2)
    p_val = 1.0 - float(stats.chi2.cdf(q_stat, df=lags))
    return float(q_stat), float(p_val)


def evaluate_goodness_of_fit(
    residuals: np.ndarray,
    alpha_threshold: float = 0.01,
) -> GoodnessOfFitResult:
    """Evaluate Kolmogorov-Smirnov and Ljung-Box statistical tests on transformed residuals."""
    if len(residuals) == 0:
        return GoodnessOfFitResult(
            ks_stat=1.0,
            ks_pvalue=0.0,
            ljung_box_stat=1.0,
            ljung_box_pvalue=0.0,
            mean_residual=0.0,
            std_residual=0.0,
            is_valid_fit=False,
        )

    # 1. Kolmogorov-Smirnov test against Exponential(scale=1.0)
    ks_res = stats.kstest(residuals, "expon", args=(0.0, 1.0))
    ks_stat = float(ks_res.statistic)
    ks_pval = float(ks_res.pvalue)

    # 2. Ljung-Box test for independence
    lags = min(10, max(1, len(residuals) // 5))
    lb_stat, lb_pval = compute_ljung_box_test(residuals, lags=lags)

    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))

    # A fit is valid if KS test does not reject Exp(1) distribution
    is_valid = ks_pval > alpha_threshold

    return GoodnessOfFitResult(
        ks_stat=ks_stat,
        ks_pvalue=ks_pval,
        ljung_box_stat=lb_stat,
        ljung_box_pvalue=lb_pval,
        mean_residual=mean_res,
        std_residual=std_res,
        is_valid_fit=is_valid,
    )


def plot_qq_residuals(
    residuals: np.ndarray,
    output_path: Path,
    title: str = "Hawkes Process Residuals Q-Q Plot (Time-Rescaling)",
) -> Path:
    """Generate high-quality Q-Q plot comparing empirical transformed residuals against Exp(1).

    Saves PNG image artifact to output_path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    N = len(residuals)

    # Empirical vs Theoretical Quantiles for Exp(1)
    sorted_res = np.sort(residuals)
    quantiles = (np.arange(1, N + 1) - 0.5) / N
    theoretical_quantiles = -np.log(1.0 - quantiles)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.scatter(
        theoretical_quantiles,
        sorted_res,
        alpha=0.6,
        color="#1f77b4",
        edgecolors="none",
        s=18,
        label="Transformed Residuals \\(\\tau_i\\)",
    )

    max_val = max(float(np.max(sorted_res)), float(np.max(theoretical_quantiles)))
    ax.plot(
        [0, max_val],
        [0, max_val],
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label="Ideal \\(\\text{Exp}(1)\\) Line (1:1)",
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Theoretical Quantiles (Standard Exponential)", fontsize=10)
    ax.set_ylabel("Empirical Transformed Residual Quantiles", fontsize=10)
    ax.set_xlim([0, max_val * 1.05])
    ax.set_ylim([0, max_val * 1.05])
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(output_path, format="png")
    plt.close(fig)

    return output_path
