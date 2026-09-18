# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Core Engine**: High-performance Rust limit order book (LOB) core with sub-microsecond latency and zero production `unwrap()`s.
- **Python Pipeline**: Binance L2 data pipeline with `aggTrades` pagination and Parquet event standardisation.
- **Hawkes Calibration**: Maximum Likelihood Estimation (MLE) engine using L-BFGS-B and boundary/degeneracy detection for empirical crypto order flow.
- **Agent Ecology**: `NoiseTrader`, `InformedTrader`, and `Spoofer` agents projecting dynamic stochastic order flow.
- **Trust Scorer**: Real-time evaluation of agent behaviour (toxicity, cancel ratios) computing a $[0, 1]$ trust score.
- **Market Maker**: Trust-aware Avellaneda-Stoikov market maker that dynamically scales risk aversion based on market toxicity.
- **Monitoring**: Real-time metrics export (`metrics` feature flag) with `tiny_http`/`prometheus` and a fully provisioned Grafana dashboard via Docker Compose.
- **Statistical Analysis**: Multi-seed automated ablation tests comparing the baseline AS model against the trust-aware modifier.
- **CI/CD**: GitHub Actions workflow running cargo checks, clippy, and the test suite across feature flags.

### Fixed
- Replaced arbitrary magic clamp numbers in the A-S variance estimator with physically meaningful circuit breakers derived from the tick scale (`MAX_TICK_RETURN`, `MAX_VARIANCE`).
- Resolved a positive feedback loop caused by thin-book conditions inflating the EWMA variance estimate.
