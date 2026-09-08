/// Univariate Exponential Hawkes Process Simulator.
///
/// **NOTE:** Session 3 found that the univariate exponential Hawkes process fails goodness-of-fit
/// on real BTCUSDT order flow (KS D=0.126, Ljung-Box Q=2800, both p~0). This simulator is a
/// structural stand-in for arrival timing, not a validated model of real crypto microstructure.
use rand::RngExt;

pub struct HawkesSimulator {
    pub mu: f64,
    pub alpha: f64,
    pub beta: f64,
    current_time: f64,
    /// The decaying excitation component: sum(alpha * exp(-beta * (t - t_i)))
    excitation: f64,
}

impl HawkesSimulator {
    pub fn new(mu: f64, alpha: f64, beta: f64) -> Self {
        Self {
            mu,
            alpha,
            beta,
            current_time: 0.0,
            excitation: 0.0,
        }
    }

    /// Simulate the next arrival time using Ogata's modified thinning algorithm.
    pub fn next_arrival<R: RngExt>(&mut self, rng: &mut R) -> f64 {
        let mut s = self.current_time;
        // Upper bound of intensity right after the previous event (or current time if rejected)
        let mut lambda_bar = self.mu + self.excitation;

        loop {
            // Draw time until next proposed event
            let u1: f64 = rng.random();
            let w = -u1.ln() / lambda_bar;
            s += w;

            // Decay the excitation to the proposed time s
            let decayed_excitation = self.excitation * (-self.beta * (s - self.current_time)).exp();
            let lambda_s = self.mu + decayed_excitation;

            // Rejection sampling
            let u2: f64 = rng.random();
            if u2 * lambda_bar <= lambda_s {
                // Accepted
                self.current_time = s;
                self.excitation = decayed_excitation + self.alpha;
                return self.current_time;
            } else {
                // Rejected, update time and upper bound
                self.current_time = s;
                self.excitation = decayed_excitation;
                lambda_bar = lambda_s;
            }
        }
    }
}
