# Contributing to SENTINEL

Thank you for your interest in contributing to SENTINEL! This project is a strict, deterministic market microstructure simulator and high-performance limit order book (LOB) engine.

## Hard Constraints for Contributions

We maintain several non-negotiable constraints to preserve the engine's integrity and deterministic guarantees:

1. **Zero `unwrap()` / `expect()` in Production Code**
   - The LOB and agent logic must gracefully handle all error states. Return `Result` or use safe fallbacks. `unwrap()` is strictly forbidden outside of `#[cfg(test)]` modules.
   
2. **Deterministic Execution**
   - The simulation must be 100% reproducible given the same random seed.
   - Do not rely on OS entropy (`thread_rng`), global system time, or multithreading race conditions for agent logic. Use explicitly passed, seeded random number generators (e.g., `StdRng::seed_from_u64`).

3. **Zero Clippy Warnings**
   - Your code must pass `cargo clippy --workspace --all-targets -- -D warnings` and the metrics equivalent cleanly.

4. **Formatting**
   - Your code must be formatted with `cargo fmt`.

## Workflow

1. Fork the repository and create a feature branch.
2. Ensure your changes adhere to the constraints above.
3. Run the full validation suite locally:
   ```bash
   cd engine
   cargo fmt --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo clippy --workspace --all-targets --features sentinel-agents/metrics -- -D warnings
   cargo test --workspace
   cargo test --workspace --features sentinel-agents/metrics
   ```
4. Submit a Pull Request. CI will run the validation suite automatically.

## Python Data Pipeline

If you are contributing to the `sentinel_analysis` Python package:
- Ensure the code is formatted with `black`.
- Run linting with `ruff check .`.
- Ensure the test suite passes via `pytest`.
