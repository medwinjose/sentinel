//! Prometheus metrics export for the SENTINEL simulation loop.
//!
//! Provides a lightweight `/metrics` HTTP endpoint via `tiny_http` on a
//! background thread so that Prometheus can scrape simulation telemetry in
//! real time.
//!
//! This module is gated behind the `metrics` Cargo feature flag:
//! ```toml
//! cargo run --features metrics
//! ```

use prometheus::{Encoder, IntCounterVec, Opts, Registry, TextEncoder};
use std::sync::Arc;
use std::thread;

/// Holds all registered Prometheus metrics for one simulation run.
pub struct SimulationMetrics {
    registry: Registry,

    /// Current mid price of the LOB (integer ticks, exposed as float gauge).
    pub mid_price: prometheus::Gauge,

    /// Current A-S optimal spread (delta) in ticks.
    pub spread: prometheus::Gauge,

    /// Current EWMA variance estimate (σ² in ticks²).
    pub variance_estimate: prometheus::Gauge,

    /// Market maker inventory position.
    pub mm_inventory: prometheus::Gauge,

    /// Market maker mark-to-market P&L.
    pub mm_pnl: prometheus::Gauge,

    /// Cumulative adverse selection cost.
    pub mm_adverse_selection_cost: prometheus::Gauge,

    /// Per-agent order placement counts, labelled by agent name.
    pub agent_order_count: IntCounterVec,

    /// Per-agent fill counts, labelled by agent name.
    pub agent_fill_count: IntCounterVec,

    /// Simulation step counter.
    pub sim_step: prometheus::IntCounter,
}

impl SimulationMetrics {
    /// Create a new metrics set and register all gauges/counters.
    pub fn new() -> Self {
        let registry = Registry::new();

        let mid_price =
            prometheus::Gauge::new("sentinel_mid_price", "Current LOB mid price (ticks)")
                .expect("metric creation should not fail");
        let spread =
            prometheus::Gauge::new("sentinel_spread", "Current A-S optimal spread (ticks)")
                .expect("metric creation should not fail");
        let variance_estimate = prometheus::Gauge::new(
            "sentinel_variance_estimate",
            "EWMA variance estimate (ticks²)",
        )
        .expect("metric creation should not fail");
        let mm_inventory =
            prometheus::Gauge::new("sentinel_mm_inventory", "Market maker inventory position")
                .expect("metric creation should not fail");
        let mm_pnl = prometheus::Gauge::new("sentinel_mm_pnl", "Market maker mark-to-market P&L")
            .expect("metric creation should not fail");
        let mm_adverse_selection_cost = prometheus::Gauge::new(
            "sentinel_mm_adverse_selection_cost",
            "Cumulative adverse selection cost",
        )
        .expect("metric creation should not fail");

        let agent_order_count = IntCounterVec::new(
            Opts::new("sentinel_agent_orders_total", "Per-agent order count"),
            &["agent"],
        )
        .expect("metric creation should not fail");

        let agent_fill_count = IntCounterVec::new(
            Opts::new("sentinel_agent_fills_total", "Per-agent fill count"),
            &["agent"],
        )
        .expect("metric creation should not fail");

        let sim_step = prometheus::IntCounter::new("sentinel_sim_step", "Simulation step counter")
            .expect("metric creation should not fail");

        registry
            .register(Box::new(mid_price.clone()))
            .expect("registration should not fail");
        registry
            .register(Box::new(spread.clone()))
            .expect("registration should not fail");
        registry
            .register(Box::new(variance_estimate.clone()))
            .expect("registration should not fail");
        registry
            .register(Box::new(mm_inventory.clone()))
            .expect("registration should not fail");
        registry
            .register(Box::new(mm_pnl.clone()))
            .expect("registration should not fail");
        registry
            .register(Box::new(mm_adverse_selection_cost.clone()))
            .expect("registration should not fail");
        registry
            .register(Box::new(agent_order_count.clone()))
            .expect("registration should not fail");
        registry
            .register(Box::new(agent_fill_count.clone()))
            .expect("registration should not fail");
        registry
            .register(Box::new(sim_step.clone()))
            .expect("registration should not fail");

        Self {
            registry,
            mid_price,
            spread,
            variance_estimate,
            mm_inventory,
            mm_pnl,
            mm_adverse_selection_cost,
            agent_order_count,
            agent_fill_count,
            sim_step,
        }
    }

    /// Encode all metrics in Prometheus text exposition format.
    pub fn encode(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder
            .encode(&metric_families, &mut buffer)
            .expect("encoding should not fail");
        String::from_utf8(buffer).expect("metrics are valid UTF-8")
    }
}

impl Default for SimulationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// A simple HTTP server that serves `/metrics` for Prometheus scraping.
///
/// Runs on a background thread; call [`MetricsServer::start`] to spawn it.
/// The server continues running until the process exits.
pub struct MetricsServer;

impl MetricsServer {
    /// Start the metrics HTTP server on `0.0.0.0:{port}` in a background
    /// daemon thread.
    ///
    /// Returns immediately. The `metrics` Arc is shared with the server
    /// thread so the simulation loop can update gauges while the server
    /// serves scrape requests concurrently.
    pub fn start(port: u16, metrics: Arc<SimulationMetrics>) {
        let addr = format!("0.0.0.0:{}", port);
        let server = tiny_http::Server::http(&addr).unwrap_or_else(|e| {
            panic!("Failed to bind metrics server to {}: {}", addr, e);
        });

        eprintln!(
            "[metrics] Prometheus metrics server listening on http://{}/metrics",
            addr
        );

        thread::Builder::new()
            .name("metrics-server".to_string())
            .spawn(move || {
                for request in server.incoming_requests() {
                    let url = request.url().to_string();
                    if url == "/metrics" || url == "/" {
                        let body = metrics.encode();
                        let response = tiny_http::Response::from_string(body).with_header(
                            tiny_http::Header::from_bytes(
                                &b"Content-Type"[..],
                                &b"text/plain; version=0.0.4; charset=utf-8"[..],
                            )
                            .expect("header is valid"),
                        );
                        let _ = request.respond(response);
                    } else {
                        let response =
                            tiny_http::Response::from_string("Not Found").with_status_code(404);
                        let _ = request.respond(response);
                    }
                }
            })
            .expect("Failed to spawn metrics server thread");
    }
}
