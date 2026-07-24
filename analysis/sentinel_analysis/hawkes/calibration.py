"""Maximum Likelihood Estimation (MLE) calibration for Hawkes process models."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sentinel_analysis.hawkes.kernels import ExponentialKernel
from sentinel_analysis.hawkes.model import HawkesProcess


@dataclass
class HawkesCalibrationResult:
    """Dataclass holding complete Hawkes process calibration results and metrics."""

    fitted_mu: float
    fitted_alpha: float
    fitted_beta: float
    branching_ratio: float
    log_likelihood: float
    aic: float
    bic: float
    n_events: int
    t_duration: float
    mu_se: float
    alpha_se: float
    beta_se: float
    mu_ci_95: tuple[float, float]
    alpha_ci_95: tuple[float, float]
    beta_ci_95: tuple[float, float]
    convergence_success: bool
    optimizer_status: int
    optimizer_message: str
    n_evaluations: int
    ci_method: str = "asymptotic_hessian"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert tuples to lists for JSON serialization
        d["mu_ci_95"] = list(self.mu_ci_95)
        d["alpha_ci_95"] = list(self.alpha_ci_95)
        d["beta_ci_95"] = list(self.beta_ci_95)
        return d


def fast_hawkes_neg_log_likelihood(
    params: np.ndarray,
    event_times: np.ndarray,
    T_end: float,
    eps: float = 1e-12,
) -> float:
    """Fast O(N) recursive negative log-likelihood for univariate exponential Hawkes process.

    params: [mu, alpha, beta]
    """
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0:
        return 1e12

    N = len(event_times)
    if N == 0:
        return float(mu * T_end)

    # Recursive calculation of R(i) = sum_{j < i} exp(-beta * (t_i - t_j))
    R = np.zeros(N, dtype=np.float64)
    for i in range(1, N):
        dt = event_times[i] - event_times[i - 1]
        R[i] = np.exp(-beta * dt) * (1.0 + R[i - 1])

    lambda_vals = mu + alpha * R
    lambda_vals = np.maximum(lambda_vals, eps)

    # 1. Sum of log intensities
    log_sum = np.sum(np.log(lambda_vals))

    # 2. Integrated intensity over [0, T_end]
    # integral_0^T lambda(t) dt = mu * T + (alpha / beta) * sum_{i=1}^N (1 - exp(-beta * (T - t_i)))
    dt_end = T_end - event_times
    integrated = mu * T_end + (alpha / beta) * np.sum(1.0 - np.exp(-beta * dt_end))

    log_likelihood = log_sum - integrated

    # Stationarity penalty if branching ratio >= 1
    if beta > 0 and (alpha / beta) >= 1.0:
        log_likelihood -= 1e5 * ((alpha / beta) - 0.999) ** 2

    return -float(log_likelihood)


def compute_hessian_standard_errors(
    params: np.ndarray,
    event_times: np.ndarray,
    T_end: float,
    h: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute standard errors and 95% CIs from numerical Hessian matrix inverse."""
    n_params = len(params)
    hessian = np.zeros((n_params, n_params), dtype=np.float64)

    # Compute numerical Hessian via central finite differences of neg_log_likelihood
    for i in range(n_params):
        for j in range(i, n_params):
            if i == j:
                p_plus = np.copy(params)
                p_minus = np.copy(params)
                p_plus[i] += h
                p_minus[i] -= h

                f_plus = fast_hawkes_neg_log_likelihood(p_plus, event_times, T_end)
                f_0 = fast_hawkes_neg_log_likelihood(params, event_times, T_end)
                f_minus = fast_hawkes_neg_log_likelihood(p_minus, event_times, T_end)

                hessian[i, i] = (f_plus - 2.0 * f_0 + f_minus) / (h**2)
            else:
                p_pp = np.copy(params)
                p_pm = np.copy(params)
                p_mp = np.copy(params)
                p_mm = np.copy(params)

                p_pp[i] += h
                p_pp[j] += h
                p_pm[i] += h
                p_pm[j] -= h
                p_mp[i] -= h
                p_mp[j] += h
                p_mm[i] -= h
                p_mm[j] -= h

                f_pp = fast_hawkes_neg_log_likelihood(p_pp, event_times, T_end)
                f_pm = fast_hawkes_neg_log_likelihood(p_pm, event_times, T_end)
                f_mp = fast_hawkes_neg_log_likelihood(p_mp, event_times, T_end)
                f_mm = fast_hawkes_neg_log_likelihood(p_mm, event_times, T_end)

                hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h**2)
                hessian[j, i] = hessian[i, j]

    try:
        cov_matrix = np.linalg.inv(hessian)
        variances = np.diag(cov_matrix)
        variances = np.maximum(variances, 1e-10)
        se = np.sqrt(variances)
    except np.linalg.LinAlgError:
        se = np.array([0.1, 0.1, 0.1])

    ci_lower = np.maximum(0.0, params - 1.96 * se)
    ci_upper = params + 1.96 * se
    cis = np.column_stack([ci_lower, ci_upper])

    return se, cis


def calibrate_hawkes(
    events: np.ndarray | list[float] | pd.Series,
    T_end: float | None = None,
    initial_guess: tuple[float, float, float] | None = None,
    seed: int = 42,
) -> HawkesCalibrationResult:
    """Calibrate univariate Hawkes process with exponential kernel via Maximum Likelihood Estimation (MLE).

    Parameters:
        events (Union[np.ndarray, List[float], pd.Series]): Array of event timestamps (sorted float seconds).
        T_end (Optional[float]): Total duration window. If None, max(events) is used.
        initial_guess (Optional[Tuple[float, float, float]]): Optional initial [mu, alpha, beta].
        seed (int): Random seed for initial guess perturbation (ensures 100% determinism).

    Returns:
        HawkesCalibrationResult: Comprehensive calibration metrics, fitted params, and CIs.
    """
    event_times = np.sort(np.asarray(events, dtype=np.float64))
    N = len(event_times)

    if N < 5:
        raise ValueError(f"Insufficient event count ({N}) for Hawkes MLE calibration.")

    t_max = float(event_times[-1])
    duration = T_end if T_end is not None else max(t_max, 1.0)

    # Deterministic initial parameter estimation
    if initial_guess is not None:
        init_p = np.array(initial_guess, dtype=np.float64)
    else:
        # Heuristic initial guess based on event rate
        mean_rate = N / duration
        init_mu = max(0.1, mean_rate * 0.4)
        init_alpha = 0.3
        init_beta = 1.0
        init_p = np.array([init_mu, init_alpha, init_beta])

    # Parameter bounds: mu > 1e-5, alpha >= 1e-5, beta > 1e-5
    bounds = [
        (1e-5, max(100.0, mean_rate * 10.0)),
        (1e-5, 5000.0),
        (1e-5, 10000.0),
    ]

    # Perform MLE optimization with L-BFGS-B
    res = minimize(
        fun=fast_hawkes_neg_log_likelihood,
        x0=init_p,
        args=(event_times, duration),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    fitted_mu, fitted_alpha, fitted_beta = res.x
    branching_ratio = fitted_alpha / fitted_beta if fitted_beta > 0 else 0.0
    neg_ll = float(res.fun)
    log_likelihood = -neg_ll

    # Compute AIC / BIC
    k = 3  # parameters: mu, alpha, beta
    aic = 2.0 * k - 2.0 * log_likelihood
    bic = k * np.log(N) - 2.0 * log_likelihood

    # Compute Standard Errors & CIs
    se, cis = compute_hessian_standard_errors(res.x, event_times, duration)

    return HawkesCalibrationResult(
        fitted_mu=float(fitted_mu),
        fitted_alpha=float(fitted_alpha),
        fitted_beta=float(fitted_beta),
        branching_ratio=float(branching_ratio),
        log_likelihood=float(log_likelihood),
        aic=float(aic),
        bic=float(bic),
        n_events=int(N),
        t_duration=float(duration),
        mu_se=float(se[0]),
        alpha_se=float(se[1]),
        beta_se=float(se[2]),
        mu_ci_95=(float(cis[0, 0]), float(cis[0, 1])),
        alpha_ci_95=(float(cis[1, 0]), float(cis[1, 1])),
        beta_ci_95=(float(cis[2, 0]), float(cis[2, 1])),
        convergence_success=bool(res.success),
        optimizer_status=int(res.status),
        optimizer_message=str(res.message),
        n_evaluations=int(res.nfev),
    )


def load_event_stream_parquet(
    data_path: Path, filter_trades_only: bool = True
) -> np.ndarray:
    """Load preprocessed Parquet file and extract point event timestamp array."""
    df = pd.read_parquet(data_path)
    if "timestamp" not in df.columns:
        raise ValueError(
            f"Parquet file {data_path} does not contain 'timestamp' column."
        )

    if (
        filter_trades_only
        and "event_type" in df.columns
        and (df["event_type"] == "trade").any()
    ):
        df_filtered = df[df["event_type"] == "trade"]
    else:
        df_filtered = df

    timestamps = np.sort(df_filtered["timestamp"].values.astype(np.float64))

    # De-duplicate identical timestamps by adding microsecond jitter for point process estimation
    n = len(timestamps)
    if n > 1:
        jitter = np.arange(n, dtype=np.float64) * 1e-6
        timestamps = timestamps + jitter

    return timestamps


def main():
    parser = argparse.ArgumentParser(description="Calibrate Hawkes process via MLE.")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to preprocessed Parquet event stream file",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for determinism")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save calibration JSON results",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]

    if args.data:
        data_path = Path(args.data)
    else:
        # Default to latest processed parquet file
        processed_dir = repo_root / "data" / "processed"
        parquet_files = sorted(processed_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                "No preprocessed Parquet files found in sentinel/data/processed."
            )
        # Prefer v2 if available
        v2_files = [f for f in parquet_files if "v2" in f.name]
        data_path = v2_files[-1] if v2_files else parquet_files[-1]

    print(f"Loading event stream from: {data_path}")
    event_times = load_event_stream_parquet(data_path)

    print(
        f"Calibrating Hawkes Process (N={len(event_times)} events, Seed={args.seed})..."
    )
    result = calibrate_hawkes(events=event_times, seed=args.seed)

    target_dir = Path(args.out_dir) if args.out_dir else data_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    res_path = target_dir / "hawkes_calibration_results.json"
    qq_plot_path = target_dir / "hawkes_qq_plot.png"

    # Goodness-of-fit validation
    from sentinel_analysis.hawkes.validation import (
        compute_transformed_residuals,
        evaluate_goodness_of_fit,
        plot_qq_residuals,
    )

    fitted_kernel = ExponentialKernel(result.fitted_alpha, result.fitted_beta)
    fitted_model = HawkesProcess(mu=result.fitted_mu, kernels=fitted_kernel)

    residuals = compute_transformed_residuals(fitted_model, event_times)
    gof_result = evaluate_goodness_of_fit(residuals)
    plot_qq_residuals(residuals, output_path=qq_plot_path)

    out_dict = result.to_dict()
    out_dict["goodness_of_fit"] = gof_result.to_dict()
    out_dict["qq_plot_artifact"] = str(qq_plot_path)

    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

    print("\nCalibration Results:")
    print(
        f"  Convergence Status : {result.convergence_success} ({result.optimizer_message})"
    )
    print(
        f"  Fitted Mu          : {result.fitted_mu:.6f} (95% CI: [{result.mu_ci_95[0]:.6f}, {result.mu_ci_95[1]:.6f}])"
    )
    print(
        f"  Fitted Alpha       : {result.fitted_alpha:.6f} (95% CI: [{result.alpha_ci_95[0]:.6f}, {result.alpha_ci_95[1]:.6f}])"
    )
    print(
        f"  Fitted Beta        : {result.fitted_beta:.6f} (95% CI: [{result.beta_ci_95[0]:.6f}, {result.beta_ci_95[1]:.6f}])"
    )
    print(f"  Branching Ratio    : {result.branching_ratio:.4f}")
    print(f"  Log-Likelihood     : {result.log_likelihood:.4f}")
    print(f"  AIC                : {result.aic:.4f}")
    print(f"  BIC                : {result.bic:.4f}")
    print("\nGoodness-of-Fit Validation:")
    print(
        f"  KS Test Stat (D)   : {gof_result.ks_stat:.6f} (p-value: {gof_result.ks_pvalue:.4e})"
    )
    print(
        f"  Ljung-Box Stat (Q) : {gof_result.ljung_box_stat:.6f} (p-value: {gof_result.ljung_box_pvalue:.4e})"
    )
    print(f"  Fit Status Valid   : {gof_result.is_valid_fit}")
    print(f"  Saved JSON Results : {res_path}")
    print(f"  Saved Q-Q Artifact : {qq_plot_path}")


if __name__ == "__main__":
    main()
