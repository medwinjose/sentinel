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
