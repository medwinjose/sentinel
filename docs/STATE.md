# SENTINEL Current State

**Last Updated:** Session 3 (2026-07-25)

## What Exists
- Monorepo directory structure (`engine/`, `analysis/`, `monitoring/`, `data/`, `docs/`).
- Rust workspace root (`engine/`) containing placeholder crate `sentinel-core` (`src/lib.rs` with `pub fn placeholder() {}`).
- Python package (`analysis/`) configured with `pyproject.toml` (`sentinel-analysis` v0.1.0) with dependencies (`matplotlib`, `numpy`, `pandas`, `pyarrow`, `requests`, `scipy`, `pytest`, `black`, `ruff`).
- Binance L2 Data Pipeline (`sentinel_analysis.data`):
  - Data fetcher (`fetch_binance_l2.py`) supporting paginated `aggTrades` fetching (5,000 trades / ~24.3 min sample window).
  - Data preprocessor (`preprocess.py`) outputting versioned Parquet event streams (`btcusdt_events_v2.parquet`, 7,000 total events).
  - Data manifest logger (`manifest.py`) generating `data_manifest.json` with provenance & SHA256 checksum.
- Hawkes Process Model & Calibration Engine (`sentinel_analysis.hawkes`):
  - Kernels (`ExponentialKernel`, `PowerLawKernel`).
  - Model (`HawkesProcess`) supporting univariate and multivariate dimensions, conditional intensity $\lambda(t)$, numerically stable log-likelihood $\log L$, and deterministic simulation via Ogata's thinning algorithm.
  - MLE Calibration (`calibration.py`) using `scipy.optimize.minimize` (L-BFGS-B) with fast $O(N)$ recursive intensity formulation, standard error estimation via Hessian matrix inversion, 95% CIs, AIC/BIC.
  - Goodness-of-fit Validation (`validation.py`) implementing Time-Rescaling Theorem compensator transform $\tau_i$, Kolmogorov-Smirnov test, Ljung-Box independence test, and Q-Q plot artifact generator (`hawkes_qq_plot.png`).
- Full pytest test suite in `analysis/tests/` (18/18 tests passing).
- Monitoring stack scaffold (`monitoring/`) with minimal `prometheus.yml` and `docker-compose.yml` for Grafana & Prometheus.
- Session continuity documentation (`docs/PROGRESS.md`, `docs/STATE.md`).

## What Does Not Exist
- Limit Order Book (LOB) engine or order matching logic in Rust (`engine/`).
- 3-agent market ecology (noise trader, informed trader, spoofer).
- Trust and toxicity scoring engine.
- Avellaneda-Stoikov market maker baseline implementation.
- Metric scraping & dashboards in Grafana/Prometheus (scaffold only).

## Next Session Plan (Session 4)
- **Goal:** High-Performance Limit Order Book (LOB) Core in Rust (`engine/`).
- **Expected Build:** Implement matching engine data structures (price-time priority order book, bid/ask price trees/skip-lists, order placement, cancellation, trade execution events) in Rust with microsecond benchmarking.

## Open Decisions & Risks
- **Empirical Model Validation:** Univariate 1D exponential Hawkes process fails i.i.d. exponential residual tests ($p < 0.001$) on empirical crypto trade flow due to buy/sell cross-excitation and power-law memory decay. Future calibration work may incorporate 2D (buy/sell) multivariate kernels or power-law decay.
- **Determinism:** All calibration initializations and simulation routines use explicit, user-seeded random state.
