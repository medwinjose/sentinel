use crate::action::AgentAction;
use crate::hawkes::HawkesSimulator;
use rand::RngExt;
use sentinel_core::book::OrderBook;
use sentinel_core::order::{Order, Side};
use std::collections::VecDeque;

/// Spoofer Agent
///
/// NOTE: This models a manipulative order-placement pattern for the purpose of testing
/// the trust/toxicity scorer. It is NOT an endorsement or how-to for market manipulation.
/// The spoofer places large limit orders away from the touch, not intended to fill,
/// and cancels them after a random dwell time.
pub struct Spoofer {
    hawkes: HawkesSimulator,
    p_bid: f64,
    distance_from_touch: i64,
    min_qty: u64,
    max_qty: u64,
    next_order_id: u64,
    min_dwell_time: f64,
    max_dwell_time: f64,
    p_accidental_fill: f64,

    // Internal queue of (cancel_time, order_id)
    cancel_queue: VecDeque<(f64, u64)>,
    // The time of the next new order arrival
    next_arrival_time: Option<f64>,
}

impl Spoofer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hawkes: HawkesSimulator,
        p_bid: f64,
        distance_from_touch: i64,
        min_qty: u64,
        max_qty: u64,
        start_order_id: u64,
        min_dwell_time: f64,
        max_dwell_time: f64,
        p_accidental_fill: f64,
    ) -> Self {
        Self {
            hawkes,
            p_bid,
            distance_from_touch,
            min_qty,
            max_qty,
            next_order_id: start_order_id,
            min_dwell_time,
            max_dwell_time,
            p_accidental_fill,
            cancel_queue: VecDeque::new(),
            next_arrival_time: None,
        }
    }

    pub fn next_event_time<R: RngExt>(&mut self, rng: &mut R) -> f64 {
        let arrival_time = *self
            .next_arrival_time
            .get_or_insert_with(|| self.hawkes.next_arrival(rng));

        if let Some(&(cancel_time, _)) = self.cancel_queue.front() {
            if cancel_time < arrival_time {
                return cancel_time;
            }
        }
        arrival_time
    }

    pub fn pop_action<R: RngExt>(&mut self, rng: &mut R, book: &OrderBook) -> Option<AgentAction> {
        let current_time = self.next_event_time(rng);

        // Check if we need to cancel first (if cancel_time <= current_time)
        if let Some(&(cancel_time, order_id)) = self.cancel_queue.front() {
            // using exact floating point equality is safe here because current_time was selected from next_event_time
            if cancel_time <= current_time {
                self.cancel_queue.pop_front();
                return Some(AgentAction::CancelOrder {
                    timestamp: (cancel_time * 1_000_000.0) as u64,
                    order_id,
                });
            }
        }

        // Otherwise it's an arrival
        self.next_arrival_time = None; // Reset so next call generates a new arrival
        let timestamp = (current_time * 1_000_000.0) as u64;

        let is_bid = rng.random_bool(self.p_bid);
        let side = if is_bid { Side::Bid } else { Side::Ask };
        let quantity = rng.random_range(self.min_qty..=self.max_qty);

        // Decide price
        let accidental = rng.random_bool(self.p_accidental_fill);
        let price = if is_bid {
            let best_bid = book.best_bid().unwrap_or(99_900);
            if accidental {
                // cross the spread or place at best bid
                book.best_ask().unwrap_or(best_bid)
            } else {
                best_bid.saturating_sub(self.distance_from_touch)
            }
        } else {
            let best_ask = book.best_ask().unwrap_or(100_100);
            if accidental {
                book.best_bid().unwrap_or(best_ask)
            } else {
                best_ask.saturating_add(self.distance_from_touch)
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

        // Schedule a cancellation
        let dwell_time = rng.random_range(self.min_dwell_time..=self.max_dwell_time);
        let cancel_time = current_time + dwell_time;

        // Insert into cancel queue, keeping it sorted by cancel_time
        // For simplicity and small numbers of open orders, we can just push and sort,
        // or find the insertion index.
        let idx = self
            .cancel_queue
            .binary_search_by(|(t, _)| t.total_cmp(&cancel_time))
            .unwrap_or_else(|x| x);
        self.cancel_queue.insert(idx, (cancel_time, order.id));

        Some(AgentAction::PlaceOrder(order))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn setup_agent() -> (Spoofer, StdRng) {
        let hawkes = HawkesSimulator::new(1.0, 0.5, 1.5);
        let agent = Spoofer::new(hawkes, 0.5, 50, 100, 500, 1, 0.1, 0.5, 0.05);
        let rng = StdRng::seed_from_u64(42);
        (agent, rng)
    }

    #[test]
    fn test_spoofer_generates_cancels() {
        let (mut agent, mut rng) = setup_agent();
        let book = OrderBook::new();

        let mut place_count = 0;
        let mut cancel_count = 0;

        for _ in 0..1000 {
            if let Some(action) = agent.pop_action(&mut rng, &book) {
                match action {
                    AgentAction::PlaceOrder(_) => place_count += 1,
                    AgentAction::CancelOrder { .. } => cancel_count += 1,
                }
            }
        }

        // Dwell times are short (0.1 to 0.5), so cancels should be happening frequently
        assert!(place_count > 0);
        assert!(cancel_count > 0);
        // By step 1000, roughly half should be cancels.
        assert!(cancel_count as f64 > (place_count as f64) * 0.5);
    }
}
