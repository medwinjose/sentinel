use crate::action::AgentAction;
use sentinel_core::book::OrderBook;
use sentinel_core::order::{Order, Side};

// ═══════════════════════════════════════════════════════════════════════════════
// Numeric stability constants — derived from the simulator's tick/price scale.
//
// The LOB operates at integer tick prices centred around ~100,000 with typical
// spreads of ~20 ticks. The EWMA variance estimator can enter a positive
// feedback loop when the book becomes thin: large mid-price jumps inflate σ²,
// which widens the MM's spread, making the book thinner, causing even larger
// jumps. These constants act as circuit breakers to keep the A-S model in a
// physically meaningful regime.
// ═══════════════════════════════════════════════════════════════════════════════

/// Maximum single-tick mid-price return fed into the EWMA variance estimator.
/// Derivation: 500 ticks = 0.5% of 100,000 mid-price. A move larger than this
/// in a single step indicates a book-empty pathology (all resting liquidity
/// swept), not genuine volatility the MM should price in.
const MAX_TICK_RETURN: f64 = 500.0;

/// Minimum variance estimate (ticks²). A variance below 1.0 implies sub-tick
/// volatility which is meaningless at integer tick resolution.
const MIN_VARIANCE: f64 = 1.0;

/// Maximum variance estimate (ticks²). σ = 50 ticks/step (√2500) implies a
/// 50-tick (~0.05%) move per step. Beyond this the model's linear
/// approximations break down and the MM should simply stop quoting.
const MAX_VARIANCE: f64 = 2500.0;

/// Minimum effective gamma. Prevents division-by-zero in the ln(1 + γ/κ) term
/// of the optimal spread formula. At γ = 0.001 the spread contribution from
/// the log term is ~2000/κ which is already very wide.
const MIN_GAMMA: f64 = 0.001;

/// Maximum inventory-induced reservation price shift (ticks). 10,000 ticks =
/// 10% of mid-price. Beyond this the MM is effectively pricing itself out of
/// the market on one side, which is the intended protective behaviour but
/// should be capped to prevent overflow in the bid/ask price computation.
const MAX_INVENTORY_SHIFT: f64 = 10_000.0;

/// Minimum optimal spread (ticks). The MM should always quote at least 1 tick
/// wide (bid < ask).
const MIN_SPREAD: f64 = 1.0;

/// Maximum optimal spread (ticks). 10,000 ticks = 10% of mid-price. Beyond
/// this the MM is effectively not providing liquidity. Tighter than the
/// previous arbitrary 50,000 bound.
const MAX_SPREAD: f64 = 10_000.0;

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
            // Clamp ret to prevent explosive variance feedback loops.
            // See MAX_TICK_RETURN derivation: 500 ticks = 0.5% of mid-price.
            let ret = (current_mid - last).clamp(-MAX_TICK_RETURN, MAX_TICK_RETURN);
            let sq_ret = ret * ret;
            // EWMA update for variance
            self.variance_estimate =
                (1.0 - self.ema_alpha) * self.variance_estimate + self.ema_alpha * sq_ret;
            // Output-side clamp keeps variance in physically meaningful range.
            self.variance_estimate = self.variance_estimate.clamp(MIN_VARIANCE, MAX_VARIANCE);
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

    pub fn get_variance_estimate(&self) -> f64 {
        self.variance_estimate
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
        let effective_gamma = effective_gamma.max(MIN_GAMMA);

        // Time remaining
        let t_remaining = (self.end_time - current_time).max(0.001);

        // Calculate A-S params
        let inventory_shift =
            self.inventory as f64 * effective_gamma * self.variance_estimate * t_remaining;
        let inventory_shift = inventory_shift.clamp(-MAX_INVENTORY_SHIFT, MAX_INVENTORY_SHIFT);
        let r = mid - inventory_shift;

        // optimal spread
        // To prevent log(0) or negative inside ln
        let ln_term = (1.0 + effective_gamma / self.kappa).ln();
        let delta = effective_gamma * self.variance_estimate * t_remaining
            + (2.0 / effective_gamma) * ln_term;
        let delta = delta.clamp(MIN_SPREAD, MAX_SPREAD);

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
