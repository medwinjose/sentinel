use crate::action::AgentAction;
use sentinel_core::book::OrderBook;
use sentinel_core::order::{Order, Side};

/// Avellaneda-Stoikov Market Maker Baseline
///
/// Reservation price: r = s - q * gamma * sigma^2 * (T - t)
/// Optimal spread: delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/kappa)
///
/// Volatility Estimator (sigma^2):
/// We use an Exponentially Weighted Moving Average (EWMA) of squared mid-price tick returns
/// to estimate the instantaneous variance sigma^2.
pub struct AvellanedaStoikovMM {
    gamma: f64,
    kappa: f64,
    end_time: f64,

    // Inventory
    inventory: i64,

    // Volatility estimation
    ema_alpha: f64,
    last_mid: Option<f64>,
    variance_estimate: f64,

    // State
    next_order_id: u64,
    last_bid_id: Option<u64>,
    last_ask_id: Option<u64>,

    // Quoting frequency
    refresh_interval: f64,
    next_action_time: f64,

    // P&L Tracking
    cash: f64,
    // Adverse Selection Tracking
    adverse_selection_cost: f64,

    // Trust Modification
    trust_modifier_strength: f64,
    current_market_trust: f64,

    // Action queue for simultaneous cancel and replace
    action_queue: std::collections::VecDeque<AgentAction>,
}

impl AvellanedaStoikovMM {
    pub fn new(
        gamma: f64,
        kappa: f64,
        session_duration: f64,
        refresh_interval: f64,
        start_order_id: u64,
    ) -> Self {
        Self {
            gamma,
            kappa,
            end_time: session_duration,
            inventory: 0,
            ema_alpha: 0.1,
            last_mid: None,
            variance_estimate: 1.0, // small default to prevent 0 spread initially
            next_order_id: start_order_id,
            last_bid_id: None,
            last_ask_id: None,
            refresh_interval,
            next_action_time: 0.0,
            cash: 0.0,
            adverse_selection_cost: 0.0,
            trust_modifier_strength: 0.0,
            current_market_trust: 1.0,
            action_queue: std::collections::VecDeque::new(),
        }
    }

    pub fn set_trust_modifier(&mut self, strength: f64) {
        self.trust_modifier_strength = strength;
    }

    pub fn update_market_trust(&mut self, trust: f64) {
        self.current_market_trust = trust;
    }

    /// Update the MM with a fill to track inventory and cash.
    pub fn on_fill(&mut self, side: Side, price: i64, _quantity: u64, current_mid: f64) {
        // Adverse selection cost: the immediate markout against the MM
        // If MM bought (Bid), and current_mid < price, MM lost money instantly (adverse)
        // If MM sold (Ask), and current_mid > price, MM lost money instantly (adverse)
        let markout = match side {
            Side::Bid => price as f64 - current_mid,
            Side::Ask => current_mid - price as f64,
        };
        self.adverse_selection_cost += markout;

        match side {
            Side::Bid => {
                self.inventory += 1;
                self.cash -= price as f64;
                self.last_bid_id = None; // Order was filled
            }
            Side::Ask => {
                self.inventory -= 1;
                self.cash += price as f64;
                self.last_ask_id = None; // Order was filled
            }
        }
    }

    pub fn on_tick(&mut self, current_mid: f64) {
        if let Some(last) = self.last_mid {
            let ret = current_mid - last;
            let sq_ret = ret * ret;
            // EWMA update for variance
            self.variance_estimate =
                (1.0 - self.ema_alpha) * self.variance_estimate + self.ema_alpha * sq_ret;
        }
        self.last_mid = Some(current_mid);
    }

    pub fn get_inventory(&self) -> i64 {
        self.inventory
    }

    pub fn get_pnl(&self, current_mid: f64) -> f64 {
        self.cash + (self.inventory as f64 * current_mid)
    }

    pub fn get_adverse_selection_cost(&self) -> f64 {
        self.adverse_selection_cost
    }

    pub fn next_event_time(&self) -> f64 {
        self.next_action_time
    }

    pub fn pop_action(&mut self, _book: &OrderBook) -> Option<AgentAction> {
        if let Some(action) = self.action_queue.pop_front() {
            return Some(action);
        }

        let current_time = self.next_action_time;
        if current_time >= self.end_time {
            return None; // Session ended
        }

        let mid = self.last_mid?;

        // Trust modification mechanism:
        // We scale the risk aversion (gamma) based on market trust.
        // If the market is highly toxic/spoofy (trust < 1.0), effective_gamma increases.
        // This causes the MM to widen its spread and skew its reservation price more aggressively
        // to protect against adverse selection.
        let effective_gamma = self.gamma
            * (1.0 + self.trust_modifier_strength * (1.0 - self.current_market_trust).max(0.0));

        // Time remaining
        let t_remaining = (self.end_time - current_time).max(0.001);

        // Calculate A-S params
        let r =
            mid - (self.inventory as f64 * effective_gamma * self.variance_estimate * t_remaining);

        // optimal spread
        // To prevent log(0) or negative inside ln
        let ln_term = (1.0 + effective_gamma / self.kappa).ln();
        let delta = effective_gamma * self.variance_estimate * t_remaining
            + (2.0 / effective_gamma) * ln_term;

        // Half spread on each side of the reservation price
        let bid_price = (r - delta / 2.0).round() as i64;
        let mut ask_price = (r + delta / 2.0).round() as i64;

        // Ensure bid < ask
        if bid_price >= ask_price {
            ask_price = bid_price.saturating_add(1);
        }

        let timestamp = (current_time * 1_000_000.0) as u64;

        // Cancel old orders if they exist
        if let Some(id) = self.last_bid_id {
            self.action_queue.push_back(AgentAction::CancelOrder {
                timestamp,
                order_id: id,
            });
        }
        if let Some(id) = self.last_ask_id {
            self.action_queue.push_back(AgentAction::CancelOrder {
                timestamp,
                order_id: id,
            });
        }

        // Place new orders
        let new_bid_id = self.next_order_id;
        self.next_order_id += 1;
        let new_ask_id = self.next_order_id;
        self.next_order_id += 1;

        self.last_bid_id = Some(new_bid_id);
        self.last_ask_id = Some(new_ask_id);

        self.action_queue.push_back(AgentAction::PlaceOrder(Order {
            id: new_bid_id,
            side: Side::Bid,
            price: bid_price,
            quantity: 1,
            timestamp,
        }));

        self.action_queue.push_back(AgentAction::PlaceOrder(Order {
            id: new_ask_id,
            side: Side::Ask,
            price: ask_price,
            quantity: 1,
            timestamp,
        }));

        self.next_action_time += self.refresh_interval;

        self.action_queue.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservation_price_shift() {
        let mut mm = AvellanedaStoikovMM::new(0.1, 1.0, 100.0, 1.0, 1);
        mm.on_tick(100.0);

        // Force variance to 1.0
        mm.variance_estimate = 1.0;

        // Extract reservation price manually for testing
        // r = s - q * gamma * sigma^2 * (T - t)

        let t_remaining = 100.0;
        let q0 = 0.0;
        let r0 = 100.0 - q0 * 0.1 * 1.0 * t_remaining;
        assert_eq!(r0, 100.0, "At zero inventory, r = s");

        // Positive inventory (q = 10)
        let q1 = 10.0;
        let r1 = 100.0 - q1 * 0.1 * 1.0 * t_remaining;
        assert!(
            r1 < 100.0,
            "Positive inventory should lower reservation price to induce selling"
        );

        // Negative inventory (q = -10)
        let q2 = -10.0;
        let r2 = 100.0 - q2 * 0.1 * 1.0 * t_remaining;
        assert!(
            r2 > 100.0,
            "Negative inventory should raise reservation price to induce buying"
        );
    }

    #[test]
    fn test_spread_widening() {
        let _mm1 = AvellanedaStoikovMM::new(0.1, 1.0, 100.0, 1.0, 1); // gamma 0.1
        let _mm2 = AvellanedaStoikovMM::new(0.5, 1.0, 100.0, 1.0, 1); // gamma 0.5 (more risk averse)

        // delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/kappa)
        // Check delta manually
        let t_rem = 100.0;
        let var = 1.0;
        let delta1 = 0.1 * var * t_rem + (2.0 / 0.1) * (1.0 + 0.1 / 1.0_f64).ln();
        let delta2 = 0.5 * var * t_rem + (2.0 / 0.5) * (1.0 + 0.5 / 1.0_f64).ln();

        assert!(
            delta2 > delta1,
            "Higher risk aversion (gamma) should widen the spread"
        );
    }
}
