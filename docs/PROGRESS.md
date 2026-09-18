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

## Session 13 (2026-09-18)
- **Part A: Re-verification of Previous Claims:**
  - **Status:** All claims from the previous commit (`c444530`) held up under strict raw re-verification.
  - **Results:**
    - `cargo fmt --check` and both `cargo clippy` passes (with and without `metrics`) exited cleanly with zero warnings/errors.
    - All 20 tests (with and without `metrics`) passed flawlessly, including the new `test_variance_stability_regression` which proved the variance is hard-capped at 2500.0 ticks².
    - `grep -rn "unwrap()"` confirmed zero `unwrap()` calls exist anywhere in the production paths of the agent modules.
    - Numeric reconciliation of the ablation study re-confirmed the conclusion: The difference in adverse selection cost is not statistically significant at the 95% confidence interval ($t = -2.16$, $p > 0.05$).
    - Verified the circuit breaker bounds (`MAX_TICK_RETURN`, `MAX_VARIANCE`, `MAX_INVENTORY_SHIFT`, `MAX_SPREAD`) are cleanly derived from the ~100k mid-price / ~20 tick spread LOB scale, heavily documented with comments.
- **Part B: CRCI-Parity Repo Polish:**
  - Added `.github/workflows/ci.yml` running fmt, clippy, and tests across all features.
  - Rewrote `README.md` to a recruiter-facing standard mirroring CRCI. It now includes: Tagline, Badges, Problem/Solution framing, Mermaid Architecture Diagram, Key Results (backed by a checked-in CSV), Quick Start, and Documentation links.
  - Added Apache 2.0 `LICENSE` (providing necessary patent retaliation clauses suitable for a complex simulation engine).
  - Added `CHANGELOG.md` summarising the history to date.
  - Added `CONTRIBUTING.md` enforcing the strict zero-unwrap and zero-clippy rules.
  - Added `SECURITY.md` scoping the threat model to the offline simulator.
  - Checked in the statistical ablation raw output to `engine/benches/results/ablation_results.csv`.

---

## Session 3 (2026-07-25)
- **Assessed & Extended Data Sample Size:**
  - Assessed that Session 2's ~2 minute dataset (1000 trades) was insufficient for stable Hawkes parameter estimation.
  - Extended `fetch_binance_l2.py` with `fetch_trades_paginated` to pull 5,000 continuous trade events (~24.3 minutes window) saved as `sentinel/data/processed/btcusdt_events_v2.parquet` (7,000 total rows: 2,000 depth order book events + 5,000 trade events).
- **Implemented MLE Calibration, Standing Degeneracy Detection & Parameter-Scaled Hessian (`sentinel_analysis.hawkes.calibration`):**
  - Implemented `calibrate_hawkes` using `scipy.optimize.minimize` (L-BFGS-B) with fast $O(N)$ recursive intensity computation.
  - Added multi-start optimization (`x0` grid) and wide parameter bounds.
  - Added explicit boundary-pinning detection (`is_boundary_pinned: bool` & `pinned_parameters: list[str]`) and standing timestamp resolution degeneracy detection (`is_degenerate: bool`, `duplicate_timestamp_ratio: float`).
  - Fixed finite-differencing Hessian step size to be parameter-scaled ($h_i = 10^{-4} \cdot \max(1.0, |p_i|)$), replacing unscaled fixed $h = 10^{-5}$ which caused float64 cancellation artifacts.
  - Replaced deterministic linear ramp tie-breaking jitter with seeded random uniform microsecond jitter (`np.random.default_rng(42).uniform(0, 1e-6)`) in `load_event_stream_parquet`.
  - CLI entrypoint: `python -m sentinel_analysis.hawkes.calibration --data sentinel/data/processed/btcusdt_events_v2.parquet --seed 42`.
- **Empirical Calibration & Timestamp Resolution Finding:**
  - **Empirical Data Limitation:** 2,283 out of 4,999 consecutive trade pairs (45.67%) share exact duplicate timestamps due to Binance REST API 1-millisecond truncation.
  - **60-Restart Cold Multi-Start Benchmark (`ftol=1e-15`):** Running 60 cold log-uniform restarts ($\alpha, \beta \in [1, 10^8]$) with ultra-tight tolerance (`ftol=1e-15`) confirms single dominant peak regions for both jitter schemes:
    - **Deterministic Linear Ramp Jitter:** Global peak at $\hat{\beta} = 4.3885 \times 10^5, \hat{\alpha} = 1.9950 \times 10^5, \hat{\mu} = 1.862679$ ($\log L = 25,426.13$, stationary ratio $n = 0.4546$, half-life $t_{1/2} = 1.58\,\mu\text{s}$). Top 10 basins cluster tightly in $\hat{\beta} \in [417\text{k}, 475\text{k}]$ within $1.5$ LL units.
    - **Seeded Random Uniform Jitter (`seed=42`):** Global peak at $\hat{\beta} = 5.7302 \times 10^6, \hat{\alpha} = 2.5628 \times 10^6, \hat{\mu} = 1.869678$ ($\log L = 33,848.78$, stationary ratio $n = 0.4473$, half-life $t_{1/2} = 120.96\,\text{ns}$). Top 15 basins cluster in $\hat{\beta} \in [3.6\cdot 10^6, 6.4\cdot 10^6]$ with $t_{1/2} \in [108\text{ns}, 190\text{ns}]$.
  - **Log-Likelihood Number Reconciliation:** The $+8,422.65$ LL delta between the global peak of Deterministic Jitter ($\log L = 25,426.13$) and Seeded Random Uniform Jitter ($\log L = 33,848.78$) is 100% caused by the tie-breaking scheme altering inter-event gaps between tied trades ($\Delta t = 1.0\,\mu\text{s} \to \text{mean } \sim 0.1-0.2\,\mu\text{s}$).
  - **Baseline Rate Decoupling ($\hat{\mu} = N_{\text{untied}}/T$):** Mathematical and empirical proof shows that at high decay rate ($\beta \ge 10^5$), kernel decay between distinct millisecond clusters is zero ($e^{-\beta \Delta t} \approx 0$). The baseline rate decouples from $\alpha$ and branching ratio $n$, and equals the distinct millisecond event rate $\mu = N_{\text{untied}}/T = 2717 / 1458.1300 = 1.863345$ (matching fitted $\hat{\mu} = 1.863350$ within $4.1 \times 10^{-6}$).
  - **Excitation Mass Analysis:** Excitation mass decomposition shows that **100.00% of the fitted self-excitation intensity contribution comes from within-tied-cluster micro-jittered pairs**, with $0.00\%$ coming from across-cluster pairs. The fitted decay half-life ($t_{1/2} = 120.96\,\text{ns}$) is a micro-jitter artifact 4 orders of magnitude below the 1-ms data resolution limit.
  - **Key takeaway for future sessions:** Micro-jittering tied timestamps invents artificial sub-millisecond structure. Point process modeling on 1-ms truncated REST streams requires merging simultaneous events into compound events with multiplicity marks, or utilizing sub-millisecond WebSocket execution streams.
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

---

## Session 4 (2026-08-23)
- **Built Rust Limit Order Book (LOB) Core (`engine/crates/core`):**
  - Implemented `Order`, `Trade`, and `Side` core types using strict types (`u64` for quantities/timestamps, `i64` for integer tick prices).
  - Implemented deterministic `OrderBook` using `BTreeMap<i64, VecDeque<Order>>` for price-time priority. 
  - Supported aggressive limit orders (crossing the spread with partial/multiple fills) and resting unfilled quantities.
  - Implemented robust `cancel_order` with `O(1)` ID lookup and queue pruning.
- **Strict Error Handling & Validation:**
  - Designed `BookError` enum to eliminate all `unwrap()`/`expect()` calls in production library code.
  - Returns explicit variants like `InvalidPrice`, `InvalidQuantity`, `DuplicateOrderId`, `OrderNotFound`.
- **Testing & Benchmarking:**
  - Wrote deterministic hand-crafted test vectors ensuring FIFO time-priority inside price levels, partial fill processing, edge case invalid orders, and cancellation lifecycle.
  - Added microsecond benchmark (`lob_benchmark.rs`) simulating 10,000 aggressive crossing orders against a book with 20,000 resting orders.
- **Rust Validation Outputs:**
  - `cargo test --all`: 7/7 unit tests passed.
  - `cargo clippy --all-targets --all-features -- -D warnings`: 0 warnings.
  - `cargo fmt --check`: Clean formatting.
  - `cargo run --example lob_benchmark --release`: Insert/Match average latency measured at ~200 ns (median). (Mean is highly sensitive to OS noise/tail outliers; raw sample output is persisted to disk).
- **Deviations from Spec:**
  - No deviations. Built strictly as requested, limit orders only, zero unwrap library code.
