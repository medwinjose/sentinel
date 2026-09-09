use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use sentinel_core::book::OrderBook;
use sentinel_core::order::{Order, Side};

fn bench_lob(c: &mut Criterion) {
    let mut group = c.benchmark_group("LOB Operations");

    // Benchmark 1: Adding passive orders (Order Placement & Price-Time Priority)
    group.bench_function("add_passive_order", |b| {
        b.iter_batched(
            || {
                let mut book = OrderBook::new();
                for i in 0..10_000 {
                    book.add_limit_order(Order {
                        id: i as u64,
                        side: Side::Ask,
                        price: 100_000 - (i as i64),
                        quantity: 10,
                        timestamp: i as u64,
                    })
                    .unwrap();
                }
                let order = Order {
                    id: 10_001,
                    side: Side::Bid,
                    price: 50_000,
                    quantity: 10,
                    timestamp: 10_001,
                };
                (book, order)
            },
            |(mut book, order)| {
                book.add_limit_order(order).unwrap();
            },
            BatchSize::SmallInput,
        )
    });

    // Benchmark 2: Adding aggressive orders (Execution / Matching)
    group.bench_function("add_aggressive_order", |b| {
        b.iter_batched(
            || {
                let mut book = OrderBook::new();
                let num_initial_orders = 20_000;
                for i in 0..num_initial_orders {
                    let price = 100_000 - (i as i64); // Asks
                    book.add_limit_order(Order {
                        id: i as u64,
                        side: Side::Ask,
                        price,
                        quantity: 10,
                        timestamp: i as u64,
                    })
                    .unwrap();

                    let price = 60_000 + (i as i64); // Bids
                    book.add_limit_order(Order {
                        id: (i + num_initial_orders) as u64,
                        side: Side::Bid,
                        price,
                        quantity: 10,
                        timestamp: i as u64,
                    })
                    .unwrap();
                }
                let mut rng = StdRng::seed_from_u64(42);
                let is_bid = rng.random_bool(0.5);
                let side = if is_bid { Side::Bid } else { Side::Ask };
                let price = if is_bid { 110_000 } else { 50_000 };
                let order = Order {
                    id: 999_999,
                    side,
                    price,
                    quantity: 5,
                    timestamp: 999_999,
                };
                (book, order)
            },
            |(mut book, order)| {
                book.add_limit_order(order).unwrap();
            },
            BatchSize::SmallInput,
        )
    });

    // Benchmark 3: Order Cancellation
    group.bench_function("cancel_order", |b| {
        b.iter_batched(
            || {
                let mut book = OrderBook::new();
                for i in 0..10_000 {
                    book.add_limit_order(Order {
                        id: i as u64,
                        side: Side::Ask,
                        price: 100_000,
                        quantity: 10,
                        timestamp: i as u64,
                    })
                    .unwrap();
                }
                book
            },
            |mut book| {
                book.cancel_order(5_000).unwrap();
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_lob);
criterion_main!(benches);
