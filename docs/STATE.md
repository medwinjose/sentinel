# SENTINEL Current State

**Last Updated:** Session 2 (2026-07-25)

## What Exists
- Monorepo directory structure (`engine/`, `analysis/`, `monitoring/`, `data/`, `docs/`).
- Rust workspace root (`engine/`) containing placeholder crate `sentinel-core` (`src/lib.rs` with `pub fn placeholder() {}`).
- Python package (`analysis/`) configured with `pyproject.toml` (`sentinel-analysis` v0.1.0) with dependencies (`numpy`, `pandas`, `pyarrow`, `requests`, `scipy`, `pytest`, `black`, `ruff`).
- Binance L2 Data Pipeline (`sentinel_analysis.data`):
  - Data fetcher (`fetch_binance_l2.py`) with `--dry-run` and mock fallbacks.
  - Data preprocessor (`preprocess.py`) outputting versioned Parquet event streams (`btcusdt_events_v1.parquet`) to `sentinel/data/processed/`.
  - Data manifest logger (`manifest.py`) generating `data_manifest.json` with provenance & SHA256 checksum.
- Hawkes Process Model Skeleton (`sentinel_analysis.hawkes`):
  - Kernels (`ExponentialKernel`, `PowerLawKernel`).
  - Model (`HawkesProcess`) supporting univariate and multivariate dimensions, conditional intensity $\lambda(t)$, numerically stable log-likelihood $\log L$, and deterministic simulation via Ogata's thinning algorithm using explicit `np.random.Generator`.
- Full pytest test suite in `analysis/tests/` (12/12 tests passing).
- Monitoring stack scaffold (`monitoring/`) with minimal `prometheus.yml` and `docker-compose.yml` for Grafana & Prometheus.
- Session continuity documentation (`docs/PROGRESS.md`, `docs/STATE.md`).

## What Does Not Exist
- Hawkes process MLE fitting / parameter calibration against empirical data (deferred to Session 3).
- Limit Order Book (LOB) engine or order matching logic in Rust (`engine/`).
- 3-agent market ecology (noise trader, informed trader, spoofer).
- Trust and toxicity scoring engine.
- Avellaneda-Stoikov market maker baseline implementation.
- Metric scraping & dashboards in Grafana/Prometheus (scaffold only).

## Next Session Plan (Session 3)
- **Goal:** Hawkes Process MLE Calibration & Parameter Estimation.
- **Expected Build:** Implement maximum likelihood estimation (MLE) parameter fitting routines (Scipy `minimize` L-BFGS-B / Nelder-Mead with parameter constraints) in `sentinel_analysis.hawkes` using the empirical preprocessed Binance L2 event stream (`btcusdt_events_v1.parquet`). Validate fitted parameters via goodness-of-fit / residual analysis (time-rescaling theorem).

## Open Decisions & Risks
- **Determinism:** Seeded `np.random.Generator` required for all stochastic routines; verified by test suite.
- **Numerical Stability:** Epsilon floor ($\epsilon = 10^{-12}$) enforced inside log-likelihood calculation to prevent log-zero domain errors.
- **Risks:** Optimization landscape for multi-dimensional Hawkes processes may contain local minima; Session 3 will require multiple random initializations or global optimization strategies.
