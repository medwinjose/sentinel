# SENTINEL Current State

**Last Updated:** Session 1 (2026-07-25)

## What Exists
- Monorepo directory structure (`engine/`, `analysis/`, `monitoring/`, `data/`, `docs/`).
- Rust workspace root (`engine/`) containing placeholder crate `sentinel-core` (`src/lib.rs` with `pub fn placeholder() {}`).
- Python package (`analysis/`) configured with `pyproject.toml` (`sentinel_analysis`).
- Monitoring stack scaffold (`monitoring/`) with minimal `prometheus.yml` and `docker-compose.yml` for Grafana & Prometheus.
- Session continuity documentation (`docs/PROGRESS.md`, `docs/STATE.md`).
- Repository configuration (`.gitignore`, `README.md`).

## What Does Not Exist
- Limit Order Book (LOB) engine or order matching logic.
- Hawkes process data calibration or intensity estimation scripts.
- 3-agent market ecology (noise trader, informed trader, spoofer).
- Trust and toxicity scoring engine.
- Avellaneda-Stoikov market maker baseline implementation.
- Metric scraping & dashboards in Grafana/Prometheus (scaffold only).

## Next Session Plan (Session 2)
- **Goal:** Data calibration.
- **Expected Build:** Hawkes process intensity estimation calibrated against Binance L2 order book dumps, data ingestion and calibration scripts in `analysis/`.

## Open Decisions & Risks
- **Packaging Decision:** Used `pyproject.toml` with `setuptools` build backend over `requirements.txt` to support standard PEP 517 build metadata, editable installs (`pip install -e .`), and clean dependency isolation for Python development.
- **Risks:** Ingestion throughput and memory footprint when processing high-frequency Binance L2 order book data dumps during Session 2.
