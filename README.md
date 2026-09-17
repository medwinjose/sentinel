# SENTINEL

SENTINEL is a trust-aware adaptive market making research project that builds an adversarial market microstructure simulator featuring a high-performance Rust limit order book engine, Hawkes-process order flow calibration from empirical L2 market data, a 3-agent market ecology (noise trader, informed trader, spoofer), a trust/toxicity scoring engine, and an Avellaneda-Stoikov market maker baseline.

## Repository Structure

- `engine/`: Rust workspace root containing performance-critical limit order book logic and core crates.
- `analysis/`: Python package (`sentinel_analysis`) for market data calibration, Hawkes process estimation, and statistical analysis.
- `monitoring/`: Docker Compose configuration for Grafana and Prometheus monitoring stack scaffold.
- `data/`: Local storage directory for raw and processed L2 order book dumps (gitignored except `.gitkeep`).
- `docs/`: Session continuity documentation (`PROGRESS.md`, `STATE.md`).

## Setup & Execution Instructions

### 1. Rust Workspace (`engine/`)

Ensure Rust (cargo) is installed, then build and test the workspace:

```bash
cd engine
cargo build
cargo test
```

### 2. Python Environment (`analysis/`)

Set up a Python virtual environment and install the analysis package in editable mode:

```bash
cd analysis
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
# source .venv/bin/activate

pip install -e .
python -c "import sentinel_analysis; print(sentinel_analysis.__version__)"
```

Why `pyproject.toml` was chosen over `requirements.txt`: `pyproject.toml` is the modern PEP 517 standard for Python packaging, facilitating seamless editable installations (`pip install -e .`), consistent tool configurations, and clean dependency declaration without setup scripts.

### 3. Monitoring Stack (`monitoring/`)

Bring up local Prometheus (port 9090) and Grafana (port 3000) containers using Docker Compose:

```bash
cd monitoring
docker-compose up -d
```

To run the simulation with Prometheus metrics export enabled:

```bash
cd engine
cargo run --features metrics
```

This will expose a `/metrics` endpoint on port `9100`. Prometheus is configured to scrape this endpoint, and Grafana is provisioned with a "Sentinel Simulation" dashboard to visualise the live telemetry.

Access Grafana at [http://localhost:3000](http://localhost:3000) (default login: admin/admin).
