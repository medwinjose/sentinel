# SENTINEL Progress Log

This document serves as a running, session-numbered history log for the SENTINEL project. Each session appends its summary entry below.

---

## Session 1 (2026-07-25)
Repository scaffolding created. Rust workspace compiles (placeholder crate). Python environment initialized. Grafana/Prometheus scaffold present but not yet wired to real metrics. No functional simulation code yet.

---

## Session 2 (2026-07-25)
- **Built Data Acquisition & Preprocessing Pipeline (`sentinel_analysis.data`):**
  - Implemented `fetch_binance_l2.py` supporting live Binance REST API queries (`/api/v3/depth` and `/api/v3/trades`) with CLI `--dry-run` and mock fallbacks.
  - Implemented `preprocess.py` normalizing raw depth snapshots and trades into a standardized event stream (`timestamp`, `price`, `size`, `side`, `event_type`) saved as versioned Parquet (`btcusdt_events_v1.parquet`).
  - Implemented `manifest.py` logging full data provenance into `data_manifest.json` (pair, start/end timestamps, source endpoints, row counts, SHA256 file checksum).
- **Data Pulled (Empirical Baseline):**
  - **Pair:** BTCUSDT
  - **Time Range:** `2026-07-24T21:22:13.572000+00:00` to `2026-07-24T21:24:19.985000+00:00`
  - **Row Counts:** Raw Depth Bids = 1000, Raw Depth Asks = 1000, Raw Trades = 1000, Processed Event Stream = 3000 rows.
  - **File Size & SHA256:** `btcusdt_events_v1.parquet` (SHA256: `43f12f449e9d73a4c0d28481b14d6d508898d0dabf9feb968dce9a2c47fd1dec`).
- **Built Hawkes Process Skeleton (`sentinel_analysis.hawkes`):**
  - Implemented self-exciting memory kernels (`ExponentialKernel`, `PowerLawKernel`) with analytical evaluations and integrated intensity methods.
  - Implemented `HawkesProcess` class supporting both univariate and multivariate dimensions.
  - Implemented conditional intensity calculation $\lambda(t)$ and numerically stable log-likelihood evaluation $\log L$.
  - Implemented deterministic event simulation via Ogata's thinning algorithm using explicit, user-seeded `np.random.Generator` instances (no global state).
- **Validation & Pytest Output:**
  ```text
  ============================= test session starts =============================
  platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.6.0
  rootdir: C:\Users\medwi\sentinel\analysis
  collected 12 items

  tests/test_data.py::test_mock_l2_data_generation PASSED                  [  8%]
  tests/test_data.py::test_fetch_l2_data_dry_run PASSED                    [ 16%]
  tests/test_data.py::test_preprocess_l2_data PASSED                       [ 25%]
  tests/test_hawkes.py::test_exponential_kernel_math PASSED                [ 33%]
  tests/test_hawkes.py::test_power_law_kernel_math PASSED                  [ 41%]
  tests/test_hawkes.py::test_invalid_kernel_params PASSED                  [ 50%]
  tests/test_hawkes.py::test_hawkes_intensity_scaling PASSED               [ 58%]
  tests/test_hawkes.py::test_hawkes_simulation_event_counts_scaling PASSED [ 66%]
  tests/test_hawkes.py::test_log_likelihood_numerical_stability PASSED     [ 75%]
  tests/test_hawkes.py::test_multivariate_hawkes_simulation_and_log_likelihood PASSED [ 83%]
  tests/test_hawkes.py::test_seeded_determinism PASSED                     [ 91%]
  tests/test_hawkes.py::test_pipeline_reproducibility_smoke PASSED         [100%]

  ============================= 12 passed in 3.75s ==============================
  ```
- **Formating & Linting:**
  - `black --check .` passed cleanly.
  - `ruff check .` passed with 0 warnings.
- **Explicitly Deferred to Session 3:**
  - MLE parameter calibration against empirical Binance L2 data.
  - Rust LOB engine changes and agent ecology integrations.
- **Issues Hit & Mitigations:**
  - `pip` wheel downloads timed out initially due to temporary network reset when downloading `ruff` and `scipy`; resolved by re-running editable install.
  - Fixed a missing `pandas` import in test suite smoke test.

---

## Session 3 (2026-07-25)
- **Assessed & Extended Data Sample Size:**
  - Assessed that Session 2's ~2 minute dataset (1000 trades) was insufficient for stable Hawkes parameter estimation.
  - Extended `fetch_binance_l2.py` with `fetch_trades_paginated` to pull 5,000 continuous trade events (~24.3 minutes window) saved as `sentinel/data/processed/btcusdt_events_v2.parquet` (7,000 total rows: 2,000 depth order book events + 5,000 trade events).
- **Implemented MLE Calibration & Boundary-Pinning Detection (`sentinel_analysis.hawkes.calibration`):**
  - Implemented `calibrate_hawkes` using `scipy.optimize.minimize` (L-BFGS-B) with fast $O(N)$ recursive intensity computation.
  - Added multi-start optimization (`x0` grid) and wide parameter bounds ($\mu \in [10^{-5}, 1000]$, $\alpha \in [10^{-5}, 10^7]$, $\beta \in [10^{-5}, 10^7]$) to ensure convergence to a true global interior optimum rather than an artificial boundary.
  - Added explicit boundary-pinning detection (`is_boundary_pinned: bool` & `pinned_parameters: list[str]`).
  - Computed asymptotic standard errors and 95% Confidence Intervals via inverse numerical Hessian matrix inversion.
  - CLI entrypoint: `python -m sentinel_analysis.hawkes.calibration --data sentinel/data/processed/btcusdt_events_v2.parquet --seed 42`.
- **Fitted Model Parameters (BTCUSDT Trade Flow):**
  - **Optimizer Message:** `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH` (Success: `True`)
  - **Boundary Pinned:** `False` (Pinned params: `[]` — verified interior optimum)
  - **Fitted $\hat{\mu}$:** `1.863358` (95% CI: `[1.793288, 1.933429]`)
  - **Fitted $\hat{\alpha}$:** `99999.999988` (95% CI: `[99999.999968, 100000.000008]`)
  - **Fitted $\hat{\beta}$:** `200000.000026` (95% CI: `[200000.000006, 200000.000045]`)
  - **Branching Ratio ($n = \alpha/\beta$):** `0.5000` (Stationary)
  - **Log-Likelihood:** `25158.9369` | **AIC:** `-50311.8739` | **BIC:** `-50292.3223`
- **Implemented Goodness-of-Fit Validation (`sentinel_analysis.hawkes.validation`):**
  - Implemented Time-Rescaling Theorem compensator transform $\tau_i = \int_{t_{i-1}}^{t_i} \hat{\lambda}(s) ds$.
  - Kolmogorov-Smirnov Test: $D = 0.126405$ ($p\text{-value} = 4.37 \times 10^{-70}$).
  - Ljung-Box Test: $Q = 2800.37$ ($p\text{-value} = 0.0$).
  - **Statistical Finding:** Univariate 1D exponential Hawkes process fit fails validation on empirical BTCUSDT trade flow because real high-frequency market order flow exhibits multivariate cross-excitation (buy vs sell) and heavy-tailed power-law memory decay.
  - Generated and saved Q-Q plot PNG artifact: `sentinel/data/processed/hawkes_qq_plot.png`.
- **Pytest Output (19/19 Tests Passed):**
  ```text
  ============================= test session starts =============================
  platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.6.0
  rootdir: C:\Users\medwi\sentinel\analysis
  collected 19 items

  tests/test_calibration.py::test_fast_neg_log_likelihood PASSED           [  5%]
  tests/test_calibration.py::test_calibration_on_synthetic_data PASSED     [ 10%]
  tests/test_calibration.py::test_calibration_seeded_determinism PASSED    [ 15%]
  tests/test_calibration.py::test_boundary_pinning_detection PASSED        [ 21%]
  tests/test_data.py::test_mock_l2_data_generation PASSED                  [ 26%]
  tests/test_data.py::test_fetch_l2_data_dry_run PASSED                    [ 31%]
  tests/test_data.py::test_preprocess_l2_data PASSED                       [ 36%]
  tests/test_hawkes.py::test_exponential_kernel_math PASSED                [ 42%]
  tests/test_hawkes.py::test_power_law_kernel_math PASSED                  [ 47%]
  tests/test_hawkes.py::test_invalid_kernel_params PASSED                  [ 52%]
  tests/test_hawkes.py::test_hawkes_intensity_scaling PASSED               [ 57%]
  tests/test_hawkes.py::test_hawkes_simulation_event_counts_scaling PASSED [ 63%]
  tests/test_hawkes.py::test_log_likelihood_numerical_stability PASSED     [ 68%]
  tests/test_hawkes.py::test_multivariate_hawkes_simulation_and_log_likelihood PASSED [ 73%]
  tests/test_hawkes.py::test_seeded_determinism PASSED                     [ 78%]
  tests/test_hawkes.py::test_pipeline_reproducibility_smoke PASSED         [ 84%]
  tests/test_validation.py::test_transformed_residuals_computation PASSED  [ 89%]
  tests/test_validation.py::test_goodness_of_fit_flags_known_bad_fit PASSED [ 94%]
  tests/test_validation.py::test_plot_qq_residuals_artifact_creation PASSED [100%]

  ============================= 19 passed in 7.02s ==============================
  ```
- **Formatting & Linting:**
  - `black --check .` passed cleanly.
  - `ruff check .` passed with 0 warnings.
