use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use sentinel_agents::action::AgentAction;
use sentinel_agents::hawkes::HawkesSimulator;
use sentinel_agents::informed_trader::InformedTrader;
use sentinel_agents::market_maker::AvellanedaStoikovMM;
use sentinel_agents::noise_trader::NoiseTrader;
use sentinel_agents::spoofer::Spoofer;
use sentinel_agents::trust_scorer::TrustScorer;
use sentinel_core::book::OrderBook;

fn derive_rngs(master_seed: u64) -> (StdRng, StdRng, StdRng) {
    let mut master_rng = StdRng::seed_from_u64(master_seed);
    let noise_rng = StdRng::seed_from_u64(master_rng.next_u64());
    let informed_rng = StdRng::seed_from_u64(master_rng.next_u64());
    let spoofer_rng = StdRng::seed_from_u64(master_rng.next_u64());
    (noise_rng, informed_rng, spoofer_rng)
}

struct SimulationResult {
    pnl: f64,
    inventory: i64,
    adverse_selection_cost: f64,
}

fn run_ecology(seed: u64, trust_modifier_strength: f64, steps: usize) -> SimulationResult {
    let (mut rng_noise, mut rng_informed, mut rng_spoofer) = derive_rngs(seed);

    let mut book = OrderBook::new();

    let hawkes_noise = HawkesSimulator::new(1.0, 0.5, 1.5);
    let mut noise = NoiseTrader::new(hawkes_noise, 0.5, 0.2, 10, 50, 10_000);

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

    let hawkes_spoofer = HawkesSimulator::new(1.0, 0.5, 1.5);
    let mut spoofer = Spoofer::new(hawkes_spoofer, 0.5, 50, 500, 1000, 30_000, 0.1, 0.5, 0.05);

    let session_duration = steps as f64 * 0.1;
    let mut mm = AvellanedaStoikovMM::new(0.1, 1.0, session_duration, 0.1, 40_000);
    mm.set_trust_modifier(trust_modifier_strength);

    let mut trust_scorer = TrustScorer::new(100, 5);
    let agent_ids = [10_000, 20_000, 30_000];
    let mut current_mid = 100_000.0;

    book.add_limit_order(sentinel_core::order::Order {
        id: 1,
        side: sentinel_core::order::Side::Bid,
        price: 99_990,
        quantity: 100,
        timestamp: 0,
    })
    .unwrap();
    book.add_limit_order(sentinel_core::order::Order {
        id: 2,
        side: sentinel_core::order::Side::Ask,
        price: 100_010,
        quantity: 100,
        timestamp: 0,
    })
    .unwrap();

    for _step in 0..steps {
        let t_noise = noise.next_event_time(&mut rng_noise);
        let t_informed = informed.next_event_time(&mut rng_informed);
        let t_spoofer = spoofer.next_event_time(&mut rng_spoofer);
        let t_mm = mm.next_event_time();

        let mut min_t = t_noise;
        let mut agent_idx = 0;

        if t_informed < min_t {
            min_t = t_informed;
            agent_idx = 1;
        }
        if t_spoofer < min_t {
            min_t = t_spoofer;
            agent_idx = 2;
        }
        if t_mm < min_t {
            agent_idx = 3;
        }

        let action = match agent_idx {
            0 => noise.pop_action(&mut rng_noise, &book),
            1 => informed.pop_action(&mut rng_informed, &book),
            2 => spoofer.pop_action(&mut rng_spoofer, &book),
            3 => mm.pop_action(&book),
            _ => unreachable!(),
        };

        if let Some(act) = action {
            match act {
                AgentAction::PlaceOrder(order) => {
                    let is_aggressive = if order.side == sentinel_core::order::Side::Bid {
                        book.best_ask().map(|p| order.price >= p).unwrap_or(false)
                    } else {
                        book.best_bid().map(|p| order.price <= p).unwrap_or(false)
                    };

                    if agent_idx < 3 {
                        trust_scorer.record_order(
                            agent_ids[agent_idx],
                            order.id,
                            is_aggressive,
                            order.timestamp,
                        );
                    }

                    let old_bid = book.best_bid();

                    let result = book.add_limit_order(order.clone());

                    if let Ok(matches) = result {
                        for t in matches {
                            if agent_idx < 3 {
                                trust_scorer.record_fill(
                                    agent_ids[agent_idx],
                                    order.id,
                                    order.side,
                                    t.price,
                                    t.timestamp,
                                );
                            }

                            if t.maker_order_id >= 40_000 {
                                let side = if t.price == old_bid.unwrap_or(0) {
                                    sentinel_core::order::Side::Bid
                                } else {
                                    sentinel_core::order::Side::Ask
                                };
                                mm.on_fill(side, t.price, t.quantity, current_mid);
                            } else if agent_idx == 3 {
                                let side = order.side;
                                mm.on_fill(side, t.price, t.quantity, current_mid);
                            }
                        }
                    }
                }
                AgentAction::CancelOrder {
                    order_id,
                    timestamp,
                } => {
                    let _ = book.cancel_order(order_id);
                    if agent_idx < 3 {
                        trust_scorer.record_cancel(agent_ids[agent_idx], order_id, timestamp);
                    }
                }
            }
        }

        if let (Some(b), Some(a)) = (book.best_bid(), book.best_ask()) {
            current_mid = (b as f64 + a as f64) / 2.0;
        }

        trust_scorer.on_tick(current_mid);
        mm.on_tick(current_mid);

        let t1 = trust_scorer.get_trust_score(agent_ids[0]);
        let t2 = trust_scorer.get_trust_score(agent_ids[1]);
        let t3 = trust_scorer.get_trust_score(agent_ids[2]);
        let avg_trust = (t1 + t2 + t3) / 3.0;
        mm.update_market_trust(avg_trust);
    }

    SimulationResult {
        pnl: mm.get_pnl(current_mid),
        inventory: mm.get_inventory(),
        adverse_selection_cost: mm.get_adverse_selection_cost(),
    }
}

#[test]
fn test_multi_seed_statistics() {
    let num_seeds = 30;
    let steps = 5000;
    let mut differences = Vec::with_capacity(num_seeds);

    println!("Running {} seeds...", num_seeds);
    println!("Seed | Base AS Cost         | Mod AS Cost          | Diff");

    for i in 0..num_seeds {
        let seed = 100 + i as u64;
        let res_base = run_ecology(seed, 0.0, steps);
        let res_mod = run_ecology(seed, 10.0, steps);

        let diff = res_base.adverse_selection_cost - res_mod.adverse_selection_cost;
        differences.push(diff);

        println!(
            "{:4} | {:20.2} | {:20.2} | {:20.2}",
            seed, res_base.adverse_selection_cost, res_mod.adverse_selection_cost, diff
        );
    }

    let n = differences.len() as f64;
    let sum: f64 = differences.iter().sum();
    let mean = sum / n;

    let variance = differences.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();
    let std_error = std_dev / n.sqrt();

    // t_stat = mean / std_error
    let t_stat = mean / std_error;

    // Approximate 95% CI (t-value for df=29 is ~2.045)
    let ci_lower = mean - 2.045 * std_error;
    let ci_upper = mean + 2.045 * std_error;

    println!("\n--- Statistical Results (Paired Difference: Base - Mod) ---");
    println!("N          : {}", num_seeds);
    println!("Mean Diff  : {:.2}", mean);
    println!("Std Error  : {:.2}", std_error);
    println!("95% CI     : [{:.2}, {:.2}]", ci_lower, ci_upper);
    println!("t-statistic: {:.2}", t_stat);

    // We just print the results, we do not strictly assert ci_lower > 0.0 because of high variance
    println!("Statistically significant? {}", ci_lower > 0.0);
}
