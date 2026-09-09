use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sentinel_agents::action::AgentAction;
use sentinel_agents::hawkes::HawkesSimulator;
use sentinel_agents::informed_trader::InformedTrader;
use sentinel_agents::noise_trader::NoiseTrader;
use sentinel_agents::spoofer::Spoofer;
use sentinel_core::book::OrderBook;

/// Derivation Scheme:
/// The master seed initializes a master RNG. We then derive independent
/// seeds for each agent by generating u64 values from the master RNG.
/// This guarantees deterministic simulation runs across all agents while
/// maintaining statistical independence between them.
fn derive_rngs(master_seed: u64) -> (StdRng, StdRng, StdRng) {
    let mut master_rng = StdRng::seed_from_u64(master_seed);
    let noise_rng = StdRng::seed_from_u64(master_rng.next_u64());
    let informed_rng = StdRng::seed_from_u64(master_rng.next_u64());
    let spoofer_rng = StdRng::seed_from_u64(master_rng.next_u64());
    (noise_rng, informed_rng, spoofer_rng)
}

#[test]
fn test_three_agent_ecology() {
    let master_seed = 42;
    let (mut rng_noise, mut rng_informed, mut rng_spoofer) = derive_rngs(master_seed);

    let mut book = OrderBook::new();

    // 1. Initialize Noise Trader
    let hawkes_noise = HawkesSimulator::new(1.0, 0.5, 1.5);
    let mut noise = NoiseTrader::new(hawkes_noise, 0.5, 0.2, 10, 50, 10_000);

    // 2. Initialize Informed Trader
    let hawkes_informed = HawkesSimulator::new(1.0, 0.5, 1.5);
    let mut informed = InformedTrader::new(
        hawkes_informed,
        1000,
        0.5,
        10,
        50,
        20_000,
        &mut rng_informed,
    );

    // 3. Initialize Spoofer
    let hawkes_spoofer = HawkesSimulator::new(1.0, 0.5, 1.5);
    let mut spoofer = Spoofer::new(
        hawkes_spoofer,
        0.5,
        50,   // distance from touch
        500,  // min qty
        1000, // max qty
        30_000,
        0.1,  // min dwell
        0.5,  // max dwell
        0.05, // accidental fill prob
    );

    let mut steps = 0;
    let mut spoofer_cancels = 0;
    let mut spoofer_places = 0;

    // Simulation loop
    while steps < 1000 {
        // Find next event
        let t_noise = noise.next_event_time(&mut rng_noise);

        // Wait, informed trader can return None if it runs out of signals or signal is below threshold.
        // It doesn't have an explicit next_event_time() that accounts for "no trade".
        // Let's modify InformedTrader to expose next_event_time or we just use peek/pop.

        let t_informed = informed.next_event_time(&mut rng_informed);
        let t_spoofer = spoofer.next_event_time(&mut rng_spoofer);

        // Find minimum time
        let mut min_t = t_noise;
        let mut agent_idx = 0;

        if t_informed < min_t {
            min_t = t_informed;
            agent_idx = 1;
        }
        if t_spoofer < min_t {
            // min_t = t_spoofer; // unused assignment
            agent_idx = 2;
        }

        let action = match agent_idx {
            0 => noise.pop_action(&mut rng_noise, &book),
            1 => informed.pop_action(&mut rng_informed, &book),
            2 => spoofer.pop_action(&mut rng_spoofer, &book),
            _ => unreachable!(),
        };

        if let Some(act) = action {
            match act {
                AgentAction::PlaceOrder(order) => {
                    let _ = book.add_limit_order(order);
                    if agent_idx == 2 {
                        spoofer_places += 1;
                    }
                }
                AgentAction::CancelOrder { order_id, .. } => {
                    let _ = book.cancel_order(order_id);
                    if agent_idx == 2 {
                        spoofer_cancels += 1;
                    }
                }
            }
        }
        steps += 1;
    }

    assert!(spoofer_places > 0);
    assert!(spoofer_cancels > 0);
    // Most spoof orders should be cancelled
    assert!(spoofer_cancels as f64 > (spoofer_places as f64) * 0.5);
}
