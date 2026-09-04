use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use sentinel_core::book::OrderBook;
use sentinel_core::order::{Order, Side};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

fn main() {
    println!("Sentinel Limit Order Book Benchmark");

    let num_runs = 5;
    for run in 1..=num_runs {
        println!("--- Run {} ---", run);

        let mut book = OrderBook::new();
        let num_initial_orders = 20_000;
        let num_warmup_orders = 2_000;
        let num_test_orders = 10_000;

        let seed: u64 = 42;
        let mut rng = StdRng::seed_from_u64(seed);

        // Pre-populate the book with liquidity
        for i in 0..num_initial_orders {
            let price = 100_000 - (i as i64); // Asks from 100,000 down to 80,001
            book.add_limit_order(Order {
                id: i as u64,
                side: Side::Ask,
                price,
                quantity: 10,
                timestamp: i as u64,
            })
            .expect("Failed to add ask");

            let price = 60_000 + (i as i64); // Bids from 60,000 up to 79,999
            book.add_limit_order(Order {
                id: (i + num_initial_orders) as u64,
                side: Side::Bid,
                price,
                quantity: 10,
                timestamp: i as u64,
            })
            .expect("Failed to add bid");
        }

        println!(
            "Pre-populated book with {} resting orders.",
            num_initial_orders * 2
        );

        // Warm-up phase
        for i in 0..num_warmup_orders {
            let id = (num_initial_orders * 2 + i) as u64;
            let is_bid = rng.random_bool(0.5);
            let side = if is_bid { Side::Bid } else { Side::Ask };
            let price = if is_bid { 110_000 } else { 50_000 };

            book.add_limit_order(Order {
                id,
                side,
                price,
                quantity: 1,
                timestamp: id,
            })
            .expect("Failed to add warmup order");
        }

        println!(
            "Processed {} warm-up orders (excluded from timing).",
            num_warmup_orders
        );

        let mut latencies = Vec::with_capacity(num_test_orders);

        let offset = num_initial_orders * 2 + num_warmup_orders;

        // Timed phase
        for i in 0..num_test_orders {
            let id = (offset + i) as u64;
            let is_bid = rng.random_bool(0.5);
            let side = if is_bid { Side::Bid } else { Side::Ask };
            let price = if is_bid { 110_000 } else { 50_000 };

            let order = Order {
                id,
                side,
                price,
                quantity: 5,
                timestamp: id,
            };

            let start = Instant::now();
            book.add_limit_order(order)
                .expect("Failed to add test order");
            latencies.push(start.elapsed().as_nanos());
        }

        let mut sorted_latencies = latencies.clone();
        sorted_latencies.sort_unstable();

        let p50 = sorted_latencies[num_test_orders / 2];
        let p95 = sorted_latencies[(num_test_orders as f64 * 0.95) as usize];
        let p99 = sorted_latencies[(num_test_orders as f64 * 0.99) as usize];
        let max = sorted_latencies[num_test_orders - 1];

        let total_nanos: u128 = latencies.iter().sum();
        let mean = total_nanos as f64 / num_test_orders as f64;

        let outliers_over_10k = latencies.iter().filter(|&&l| l > 10_000).count();

        for (i, &l) in latencies.iter().enumerate() {
            if l > 10_000 {
                println!(
                    "Outlier detected: test order index = {}, order id = {}, latency = {} ns",
                    i,
                    (offset + i) as u64,
                    l
                );
            }
        }

        // Persist raw samples to disk
        let filename = format!("target/benchmark_latencies_run_{}.csv", run);
        let mut file = File::create(&filename).expect("Failed to create log file");
        for latency in &latencies {
            writeln!(file, "{}", latency).expect("Failed to write to log file");
        }

        println!(
            "Processed {} aggressive limit orders (Timed Phase).",
            num_test_orders
        );
        println!("Configuration: Seed = {}, RNG = rand::rngs::StdRng", seed);
        println!("--- Latency Distribution ---");
        println!(
            "Mean : {:.2} ns (Note: mean is tail-sensitive on this workload)",
            mean
        );
        println!("p50  : {} ns", p50);
        println!("p95  : {} ns", p95);
        println!("p99  : {} ns", p99);
        println!("Max  : {} ns", max);
        println!("Outliers > 10,000 ns: {}", outliers_over_10k);
        println!("Raw samples persisted to {}", filename);
        println!();
    }
}
