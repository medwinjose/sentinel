use crate::action::AgentAction;
use crate::hawkes::HawkesSimulator;
use rand::RngExt;
use sentinel_core::book::OrderBook;
use sentinel_core::order::{Order, Side};

/// Informed Trader
///
/// NOTE: This models a stylized proxy for informed flow by utilizing a synthetic
/// precomputed future price signal. It is NOT a claim of true alpha prediction, but rather
/// an architectural setup for testing how information asymmetry affects the LOB ecology.
pub struct InformedTrader {
    hawkes: HawkesSimulator,
    signal_array: Vec<f64>,
    signal_index: usize,
    confidence_threshold: f64,
    min_qty: u64,
    max_qty: u64,
    next_order_id: u64,
    next_arrival_time: Option<f64>,
}

impl InformedTrader {
    pub fn new<R: RngExt>(
        hawkes: HawkesSimulator,
        num_steps: usize,
        confidence_threshold: f64,
        min_qty: u64,
        max_qty: u64,
        start_order_id: u64,
        rng: &mut R,
    ) -> Self {
        let mut signal_array = Vec::with_capacity(num_steps);
        let mut current_drift = 0.0;
        for _ in 0..num_steps {
            let step: f64 = rng.random_range(-1.0..1.0);
            current_drift += step;
            // Mean reversion to keep it bounded
            current_drift *= 0.95;
            signal_array.push(current_drift);
        }

        Self {
            hawkes,
            signal_array,
            signal_index: 0,
            confidence_threshold,
            min_qty,
            max_qty,
            next_order_id: start_order_id,
            next_arrival_time: None,
        }
    }

    pub fn next_event_time<R: RngExt>(&mut self, rng: &mut R) -> f64 {
        let t = self
            .next_arrival_time
            .unwrap_or_else(|| self.hawkes.next_arrival(rng));
        self.next_arrival_time = Some(t);
        t
    }

    /// Generates the next action. Returns `None` if it decides not to trade (signal below threshold)
    /// or if it runs out of signals.
    pub fn pop_action<R: RngExt>(&mut self, rng: &mut R, book: &OrderBook) -> Option<AgentAction> {
        let current_time = self.next_event_time(rng);
        self.next_arrival_time = None;

        let timestamp = (current_time * 1_000_000.0) as u64;

        if self.signal_index >= self.signal_array.len() {
            return None;
        }

        let current_signal = self.signal_array[self.signal_index];
        self.signal_index += 1;

        if current_signal.abs() < self.confidence_threshold {
            return None;
        }

        let is_bid = current_signal > 0.0;
        let side = if is_bid { Side::Bid } else { Side::Ask };
        let quantity = rng.random_range(self.min_qty..=self.max_qty);

        let price = if is_bid {
            book.best_ask().unwrap_or(100_000)
        } else {
            book.best_bid().unwrap_or(100_000)
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

    fn setup_agent() -> (InformedTrader, StdRng) {
        let hawkes = HawkesSimulator::new(1.0, 0.5, 1.5);
        let mut rng = StdRng::seed_from_u64(42);
        let agent = InformedTrader::new(hawkes, 1000, 0.5, 10, 50, 1, &mut rng);
        (agent, rng)
    }

    #[test]
    fn test_determinism() {
        let (mut agent1, mut rng1) = setup_agent();
        let (mut agent2, mut rng2) = setup_agent();

        let book = OrderBook::new();

        for _ in 0..100 {
            let order1 = agent1.pop_action(&mut rng1, &book);
            let order2 = agent2.pop_action(&mut rng2, &book);
            assert_eq!(order1, order2);
        }
    }

    #[test]
    fn test_signal_correlation() {
        let (mut agent, mut rng) = setup_agent();
        let book = OrderBook::new();

        let mut bid_count = 0;
        let mut ask_count = 0;

        for _ in 0..1000 {
            let previous_signal_index = agent.signal_index;
            if let Some(AgentAction::PlaceOrder(order)) = agent.pop_action(&mut rng, &book) {
                let signal = agent.signal_array[previous_signal_index];
                assert!(signal.abs() >= agent.confidence_threshold);
                if signal > 0.0 {
                    assert_eq!(order.side, Side::Bid);
                    bid_count += 1;
                } else {
                    assert_eq!(order.side, Side::Ask);
                    ask_count += 1;
                }
            }
        }

        // Assert we actually generated some orders
        assert!(bid_count + ask_count > 0);
    }
}
