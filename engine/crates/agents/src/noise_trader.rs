use crate::action::AgentAction;
use crate::hawkes::HawkesSimulator;
use rand::RngExt;
use sentinel_core::book::OrderBook;
use sentinel_core::order::{Order, Side};

pub struct NoiseTrader {
    hawkes: HawkesSimulator,
    p_bid: f64,
    p_cross_spread: f64,
    min_qty: u64,
    max_qty: u64,
    next_order_id: u64,
    next_arrival_time: Option<f64>,
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
            next_arrival_time: None,
        }
    }

    pub fn next_event_time<R: RngExt>(&mut self, rng: &mut R) -> f64 {
        if self.next_arrival_time.is_none() {
            self.next_arrival_time = Some(self.hawkes.next_arrival(rng));
        }
        self.next_arrival_time.unwrap()
    }

    pub fn pop_action<R: RngExt>(&mut self, rng: &mut R, book: &OrderBook) -> Option<AgentAction> {
        let current_time = self.next_event_time(rng);
        self.next_arrival_time = None;

        let timestamp = (current_time * 1_000_000.0) as u64;

        let is_bid = rng.random_bool(self.p_bid);
        let side = if is_bid { Side::Bid } else { Side::Ask };

        let quantity = rng.random_range(self.min_qty..=self.max_qty);

        let crosses_spread = rng.random_bool(self.p_cross_spread);

        let price = if is_bid {
            if crosses_spread {
                book.best_ask().unwrap_or(100_000)
            } else {
                book.best_bid().unwrap_or(99_990)
            }
        } else {
            if crosses_spread {
                book.best_bid().unwrap_or(100_000)
            } else {
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

        Some(AgentAction::PlaceOrder(order))
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
            let order1 = agent1.pop_action(&mut rng1, &book);
            let order2 = agent2.pop_action(&mut rng2, &book);
            assert_eq!(order1, order2, "Orders should be byte-for-byte identical");
        }
    }

    #[test]
    fn test_order_validity() {
        let (mut agent, mut rng) = setup_agent();
        let mut book = OrderBook::new();

        for _ in 0..1000 {
            if let Some(AgentAction::PlaceOrder(order)) = agent.pop_action(&mut rng, &book) {
                let result = book.add_limit_order(order);
                assert!(result.is_ok(), "Generated order was invalid");
            }
        }
    }

    #[test]
    fn test_spread_crossing_produces_fill() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut book = OrderBook::new();

        // Force an ask agent to be completely passive and always place an ask
        let hawkes_ask = HawkesSimulator::new(1.0, 0.5, 1.5);
        let mut agent_ask = NoiseTrader::new(hawkes_ask, 0.0, 0.0, 10, 50, 1000);

        if let Some(AgentAction::PlaceOrder(order)) = agent_ask.pop_action(&mut rng, &book) {
            let trades1 = book.add_limit_order(order).unwrap();
            assert!(trades1.is_empty(), "Passive order should not fill");
            assert!(book.best_ask().is_some(), "Ask should be resting");
        }

        // Force a bid agent to be completely aggressive and always cross the spread
        let hawkes_bid = HawkesSimulator::new(1.0, 0.5, 1.5);
        let mut agent_bid = NoiseTrader::new(hawkes_bid, 1.0, 1.0, 10, 50, 1);

        if let Some(AgentAction::PlaceOrder(order)) = agent_bid.pop_action(&mut rng, &book) {
            let trades2 = book.add_limit_order(order).unwrap();
            assert!(
                !trades2.is_empty(),
                "Aggressive order crossing spread should produce a fill"
            );
        }
    }
}
