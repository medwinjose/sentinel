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

    /// Generates the next order without submitting it.
    pub fn generate_order<R: RngExt>(&mut self, rng: &mut R, book: &OrderBook) -> Order {
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

        let order = Order {
            id: self.next_order_id,
            side,
            price,
            quantity,
            timestamp,
        };
        self.next_order_id += 1;

        order
    }

    /// Simulates the next arrival and submits the order to the provided limit order book.
    pub fn step<R: RngExt>(
        &mut self,
        rng: &mut R,
        book: &mut OrderBook,
    ) -> Result<Vec<Trade>, BookError> {
        let order = self.generate_order(rng, book);
        book.add_limit_order(order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn setup_agent() -> (NoiseTrader, StdRng) {
        let hawkes = HawkesSimulator::new(1.0, 0.5, 1.5);
        let agent = NoiseTrader::new(hawkes, 0.5, 0.2, 10, 50, 1);
        let rng = StdRng::seed_from_u64(42);
        (agent, rng)
    }

    #[test]
    fn test_determinism() {
        let (mut agent1, mut rng1) = setup_agent();
        let (mut agent2, mut rng2) = setup_agent();

        let book = OrderBook::new();

        // 100 generated orders should be exactly identical
        for _ in 0..100 {
            let order1 = agent1.generate_order(&mut rng1, &book);
            let order2 = agent2.generate_order(&mut rng2, &book);
            assert_eq!(order1, order2, "Orders should be byte-for-byte identical");
        }
    }

    #[test]
    fn test_order_validity() {
        let (mut agent, mut rng) = setup_agent();
        let mut book = OrderBook::new();

        for _ in 0..1000 {
            let result = agent.step(&mut rng, &mut book);
            // Should not panic and should not be a BookError (e.g. InvalidPrice/InvalidQuantity)
            assert!(result.is_ok(), "Generated order was invalid");
        }
    }

    #[test]
    fn test_spread_crossing_produces_fill() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut book = OrderBook::new();

        // Force an ask agent to be completely passive and always place an ask
        let hawkes_ask = HawkesSimulator::new(1.0, 0.5, 1.5);
        let mut agent_ask = NoiseTrader::new(hawkes_ask, 0.0, 0.0, 10, 50, 1000);

        let trades1 = agent_ask.step(&mut rng, &mut book).unwrap();
        assert!(trades1.is_empty(), "Passive order should not fill");
        assert!(book.best_ask().is_some(), "Ask should be resting");

        // Force a bid agent to be completely aggressive and always cross the spread
        let hawkes_bid = HawkesSimulator::new(1.0, 0.5, 1.5);
        let mut agent_bid = NoiseTrader::new(hawkes_bid, 1.0, 1.0, 10, 50, 1);

        let trades2 = agent_bid.step(&mut rng, &mut book).unwrap();
        assert!(
            !trades2.is_empty(),
            "Aggressive order crossing spread should produce a fill"
        );
    }
}
