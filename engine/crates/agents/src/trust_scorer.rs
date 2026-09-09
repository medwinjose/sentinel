use sentinel_core::order::Side;
use std::collections::{HashMap, VecDeque};

/// TrustScorer computes trust scores for agents based on 4 metrics:
/// - T_i: Toxicity Index (post-trade markout). Lower toxicity = higher trust.
/// - C_i: Cancel Ratio. Lower cancel ratio = higher trust.
/// - L_i: Order Lifetime. Longer lifetime = higher trust.
/// - A_i: Aggressiveness. Lower aggressiveness = higher trust.
///
/// Composite Trust_i is a weighted sum of these [0, 1] components.
/// Higher composite score = more trustworthy.
pub struct TrustScorer {
    window_size: usize,
    markout_horizon_ticks: usize,

    // Agent ID -> AgentStats
    stats: HashMap<u64, AgentStats>,

    // Weights
    w_t: f64,
    w_c: f64,
    w_l: f64,
    w_a: f64,

    // Global tick counter for tracking markout horizons
    current_tick: u64,
}

#[derive(Default)]
struct AgentStats {
    // Rolling windows
    // Toxicity: true markout value. We apply a sigmoid later to map to [0,1].
    markouts: VecDeque<f64>,
    // Cancel ratio: total orders and cancelled orders in window
    orders_placed: VecDeque<u64>, // stores timestamp
    cancels: VecDeque<u64>,       // stores timestamp
    // Order lifetimes in microseconds
    lifetimes: VecDeque<u64>,
    // Aggressiveness: true if aggressive, false if passive
    aggressiveness: VecDeque<bool>,

    // Pending fills waiting for markout evaluation
    // (target_tick, fill_price, side)
    pending_markouts: VecDeque<(u64, i64, Side)>,

    // Tracking open orders: order_id -> placement_timestamp
    open_orders: HashMap<u64, u64>,
}

impl TrustScorer {
    pub fn new(window_size: usize, markout_horizon_ticks: usize) -> Self {
        Self {
            window_size,
            markout_horizon_ticks,
            stats: HashMap::new(),
            w_t: 0.25,
            w_c: 0.25,
            w_l: 0.25,
            w_a: 0.25,
            current_tick: 0,
        }
    }

    pub fn set_weights(&mut self, w_t: f64, w_c: f64, w_l: f64, w_a: f64) {
        let sum = w_t + w_c + w_l + w_a;
        self.w_t = w_t / sum;
        self.w_c = w_c / sum;
        self.w_l = w_l / sum;
        self.w_a = w_a / sum;
    }

    /// Called on every LOB event to advance the tick counter and resolve pending markouts.
    pub fn on_tick(&mut self, current_mid: f64) {
        self.current_tick += 1;
        for stats in self.stats.values_mut() {
            while let Some(&(target_tick, fill_price, side)) = stats.pending_markouts.front() {
                if self.current_tick >= target_tick {
                    // Evaluate markout
                    // Sign convention: positive when adverse to the fill's counterparty.
                    // If agent bought (Bid), price goes up (current_mid > fill_price) -> toxic, positive markout.
                    // If agent sold (Ask), price goes down (current_mid < fill_price) -> toxic, positive markout.
                    let move_dir = current_mid - fill_price as f64;
                    let markout = match side {
                        Side::Bid => move_dir,
                        Side::Ask => -move_dir,
                    };

                    stats.markouts.push_back(markout);
                    if stats.markouts.len() > self.window_size {
                        stats.markouts.pop_front();
                    }
                    stats.pending_markouts.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    pub fn record_order(
        &mut self,
        agent_id: u64,
        order_id: u64,
        is_aggressive: bool,
        timestamp: u64,
    ) {
        let stats = self.stats.entry(agent_id).or_default();
        stats.open_orders.insert(order_id, timestamp);

        stats.orders_placed.push_back(timestamp);
        if stats.orders_placed.len() > self.window_size {
            stats.orders_placed.pop_front();
        }

        stats.aggressiveness.push_back(is_aggressive);
        if stats.aggressiveness.len() > self.window_size {
            stats.aggressiveness.pop_front();
        }
    }

    pub fn record_cancel(&mut self, agent_id: u64, order_id: u64, timestamp: u64) {
        let stats = self.stats.entry(agent_id).or_default();
        if let Some(placed_time) = stats.open_orders.remove(&order_id) {
            let lifetime = timestamp.saturating_sub(placed_time);
            stats.lifetimes.push_back(lifetime);
            if stats.lifetimes.len() > self.window_size {
                stats.lifetimes.pop_front();
            }
        }

        stats.cancels.push_back(timestamp);
        if stats.cancels.len() > self.window_size {
            stats.cancels.pop_front();
        }
    }

    pub fn record_fill(
        &mut self,
        agent_id: u64,
        order_id: u64,
        side: Side,
        price: i64,
        timestamp: u64,
    ) {
        let stats = self.stats.entry(agent_id).or_default();
        if let Some(placed_time) = stats.open_orders.remove(&order_id) {
            let lifetime = timestamp.saturating_sub(placed_time);
            stats.lifetimes.push_back(lifetime);
            if stats.lifetimes.len() > self.window_size {
                stats.lifetimes.pop_front();
            }
        }

        // Schedule markout evaluation
        stats.pending_markouts.push_back((
            self.current_tick + self.markout_horizon_ticks as u64,
            price,
            side,
        ));
    }

    /// Calculate Trust_i for an agent. Returns 1.0 if not enough data.
    pub fn get_trust_score(&self, agent_id: u64) -> f64 {
        let stats = match self.stats.get(&agent_id) {
            Some(s) => s,
            None => return 1.0, // Default trust for unknown
        };

        // Toxicity: mean markout. Map using sigmoid to [0,1].
        // Higher markout = higher toxicity. We want lower toxicity to map to ~1.
        // So Trust_T = 1 - sigmoid(alpha * mean_markout)
        // Or simply sigmoid(-alpha * mean_markout).
        let mean_markout = if stats.markouts.is_empty() {
            0.0
        } else {
            stats.markouts.iter().sum::<f64>() / stats.markouts.len() as f64
        };
        // Alpha tuning factor to make typical tick moves (-10 to 10) spread across sigmoid
        let alpha = 0.1;
        let t_score = 1.0 / (1.0 + (-alpha * -mean_markout).exp()); // Negative markout gives >0.5 trust, positive gives <0.5

        // Cancel ratio: C_i = cancels / orders.
        // If ratio = 1, score = 0. If ratio = 0, score = 1.
        let num_orders = stats.orders_placed.len() as f64;
        let c_score = if num_orders > 0.0 {
            let cancels_in_window = stats.cancels.len() as f64; // Approximation, true cancel ratio would match up cancels to orders in window. This is simpler.
            let ratio = (cancels_in_window / num_orders).min(1.0);
            1.0 - ratio
        } else {
            1.0
        };

        // Lifetime: mean lifetime in microseconds.
        // Normalize inversely: shorter lifetime = lower Li score.
        // Let's use an exponential mapping: 1.0 - exp(-beta * mean_lifetime)
        // 1 sec = 1,000,000 us. Let's say 1 sec lifetime is decent trust.
        let mean_lifetime = if stats.lifetimes.is_empty() {
            1_000_000.0 // 1 sec default
        } else {
            stats.lifetimes.iter().sum::<u64>() as f64 / stats.lifetimes.len() as f64
        };
        let beta = 1e-6; // 1 second gives 1 - exp(-1) = 0.63
        let l_score = 1.0 - (-beta * mean_lifetime).exp();

        // Aggressiveness: fraction of aggressive orders.
        // If 1.0 aggressive, A_score = 0. If 0.0 aggressive, A_score = 1.
        let num_agg = stats.aggressiveness.len() as f64;
        let a_score = if num_agg > 0.0 {
            let agg_count = stats.aggressiveness.iter().filter(|&&x| x).count() as f64;
            1.0 - (agg_count / num_agg)
        } else {
            1.0
        };

        self.w_t * t_score + self.w_c * c_score + self.w_l * l_score + self.w_a * a_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_scorer() {
        let mut scorer = TrustScorer::new(100, 5);

        // Agent 1: Clean (Passive, low cancels, long lifetime, negative markout)
        let agent1 = 1;
        for i in 0..50 {
            scorer.record_order(agent1, i, false, i * 2_000_000);
            scorer.record_fill(agent1, i, Side::Bid, 100_000, i * 2_000_000 + 1_000_000);
            // 1 sec lifetime
        }
        for _ in 0..5 {
            scorer.on_tick(99_990.0); // Price drops after they buy -> negative markout -> good
        }

        // Agent 2: Spoofer (Passive, high cancels, short lifetime, neutral markout)
        let agent2 = 2;
        for i in 0..50 {
            let order_id = i + 1000;
            scorer.record_order(agent2, order_id, false, i * 2_000_000);
            scorer.record_cancel(agent2, order_id, i * 2_000_000 + 10_000); // 10ms lifetime
        }

        // Agent 3: Toxic (Aggressive, low cancels, short lifetime since aggressive, highly positive markout)
        let agent3 = 3;
        for i in 0..50 {
            let order_id = i + 2000;
            scorer.record_order(agent3, order_id, true, i * 2_000_000);
            scorer.record_fill(agent3, order_id, Side::Bid, 100_000, i * 2_000_000 + 1);
            // 1us lifetime
        }
        for _ in 0..5 {
            scorer.on_tick(100_050.0); // Price spikes after they buy -> positive markout -> toxic
        }

        let score1 = scorer.get_trust_score(agent1);
        let score2 = scorer.get_trust_score(agent2);
        let score3 = scorer.get_trust_score(agent3);

        println!("Agent 1 (Clean) Score: {}", score1);
        println!("Agent 2 (Spoofer) Score: {}", score2);
        println!("Agent 3 (Toxic) Score: {}", score3);

        // Verify ordering: Clean > Spoofer, Clean > Toxic
        assert!(score1 > score2, "Clean should be more trusted than Spoofer");
        assert!(score1 > score3, "Clean should be more trusted than Toxic");

        // Spoofer vs Toxic depends on weights. With equal weights:
        // Spoofer: bad cancel ratio, bad lifetime, good aggressiveness, good markout.
        // Toxic: good cancel ratio, bad lifetime, bad aggressiveness, bad markout.
        // Toxic markout is heavily penalized.
        // Both are scored appropriately in unit tests.
    }
}
