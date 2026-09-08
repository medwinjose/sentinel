use crate::hawkes::HawkesSimulator;
use rand::RngExt;
use sentinel_core::book::OrderBook;
use sentinel_core::order::{BookError, Order, Side, Trade};

pub struct NoiseTrader {
    hawkes: HawkesSimulator,
    p_bid: f64,
    p_cross_spread: f64,
    min_qty: u64,
    max_qty: u64,
    next_order_id: u64,
}

impl NoiseTrader {
    pub fn new(
        hawkes: HawkesSimulator,
        p_bid: f64,
        p_cross_spread: f64,
        min_qty: u64,
        max_qty: u64,
        start_order_id: u64,
    ) -> Self {
        Self {
            hawkes,
            p_bid,
            p_cross_spread,
            min_qty,
            max_qty,
            next_order_id: start_order_id,
        }
    }

    /// Simulates the next arrival and submits the order to the provided limit order book.
    pub fn step<R: RngExt>(
        &mut self,
        rng: &mut R,
        book: &mut OrderBook,
    ) -> Result<Vec<Trade>, BookError> {
        // 1. Advance time using Hawkes simulator (seconds)
        let arrival_time = self.hawkes.next_arrival(rng);
        // Scale to microseconds for timestamp
        let timestamp = (arrival_time * 1_000_000.0) as u64;

        // 2. Decide side
        let is_bid = rng.random_bool(self.p_bid);
        let side = if is_bid { Side::Bid } else { Side::Ask };

        // 3. Decide size
        let quantity = rng.random_range(self.min_qty..=self.max_qty);

        // 4. Decide price
        let crosses_spread = rng.random_bool(self.p_cross_spread);

        let price = if is_bid {
            if crosses_spread {
                // Aggressive: hit the best ask. If no ask, default to something reasonable.
                book.best_ask().unwrap_or(100_000)
            } else {
                // Passive: rest at best bid or slightly lower
                book.best_bid().unwrap_or(99_990)
            }
        } else {
            if crosses_spread {
                // Aggressive: hit the best bid
                book.best_bid().unwrap_or(100_000)
            } else {
                // Passive: rest at best ask or slightly higher
                book.best_ask().unwrap_or(100_010)
            }
        };

        // 5. Submit order
        let order = Order {
            id: self.next_order_id,
            side,
            price,
            quantity,
            timestamp,
        };
        self.next_order_id += 1;

        book.add_limit_order(order)
    }
}
