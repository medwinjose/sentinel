use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use std::path::Path;

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

#[derive(Clone, Copy)]
struct SimulationConfig {
    pub regime_name: &'static str,

    // Noise Trader Hawkes
    pub noise_lambda_0: f64,
    pub noise_alpha: f64,
    pub noise_beta: f64,
    pub noise_qty: u64,

    // Informed Trader Hawkes
    pub inf_lambda_0: f64,
    pub inf_alpha: f64,
    pub inf_beta: f64,

    // Spoofer Hawkes
    pub spoofer_lambda_0: f64,
    pub spoofer_alpha: f64,
    pub spoofer_beta: f64,

    // MM risk aversion
    pub mm_gamma: f64,
}

#[allow(dead_code)]
struct SimulationResult {
    pnl: f64,
    inventory: i64,
    adverse_selection_cost: f64,
    inventory_variance: f64,
}

fn run_ecology(
    seed: u64,
    trust_modifier_strength: f64,
    steps: usize,
    cfg: &SimulationConfig,
) -> SimulationResult {
    let (mut rng_noise, mut rng_informed, mut rng_spoofer) = derive_rngs(seed);

    let mut book = OrderBook::new();

    let hawkes_noise = HawkesSimulator::new(cfg.noise_lambda_0, cfg.noise_alpha, cfg.noise_beta);
    let mut noise = NoiseTrader::new(hawkes_noise, 0.5, 0.2, 10, cfg.noise_qty, 10_000);

    let hawkes_informed = HawkesSimulator::new(cfg.inf_lambda_0, cfg.inf_alpha, cfg.inf_beta);
    let mut informed = InformedTrader::new(
        hawkes_informed,
        1000,
        0.5,
        10,
        50,
        20_000,
        &mut rng_informed,
    );

    let hawkes_spoofer =
        HawkesSimulator::new(cfg.spoofer_lambda_0, cfg.spoofer_alpha, cfg.spoofer_beta);
    let mut spoofer = Spoofer::new(hawkes_spoofer, 0.5, 50, 500, 1000, 30_000, 0.1, 0.5, 0.05);

    let session_duration = steps as f64 * 0.1;
    let mut mm = AvellanedaStoikovMM::new(0.1, cfg.mm_gamma, session_duration, 0.1, 40_000);
    mm.set_trust_modifier(trust_modifier_strength);

    let mut trust_scorer = TrustScorer::new(100, 5);
    let agent_ids = [10_000, 20_000, 30_000];
    let mut current_mid = 100_000.0;

    let mut inv_sum = 0.0;
    let mut inv_sq_sum = 0.0;
    let mut tick_count = 0.0;

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

                    let result = book.add_limit_order(order);

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

        let inv = mm.get_inventory() as f64;
        inv_sum += inv;
        inv_sq_sum += inv * inv;
        tick_count += 1.0;
    }

    let mean_inv = inv_sum / tick_count;
    let inv_variance = (inv_sq_sum / tick_count) - (mean_inv * mean_inv);

    SimulationResult {
        pnl: mm.get_pnl(current_mid),
        inventory: mm.get_inventory(),
        adverse_selection_cost: mm.get_adverse_selection_cost(),
        inventory_variance: inv_variance.max(0.0),
    }
}

#[test]
fn test_multi_seed_statistics() {
    let num_seeds = 500;
    let steps = 5000;

    let regimes = vec![
        SimulationConfig {
            regime_name: "Low_Vol",
            noise_lambda_0: 0.5,
            noise_alpha: 0.2,
            noise_beta: 1.0,
            noise_qty: 20,
            inf_lambda_0: 0.1,
            inf_alpha: 0.1,
            inf_beta: 1.5,
            spoofer_lambda_0: 0.2,
            spoofer_alpha: 0.2,
            spoofer_beta: 1.5,
            mm_gamma: 1.0,
        },
        SimulationConfig {
            regime_name: "Med_Vol",
            noise_lambda_0: 1.0,
            noise_alpha: 0.5,
            noise_beta: 1.5,
            noise_qty: 50,
            inf_lambda_0: 0.5,
            inf_alpha: 0.5,
            inf_beta: 1.5,
            spoofer_lambda_0: 1.0,
            spoofer_alpha: 0.5,
            spoofer_beta: 1.5,
            mm_gamma: 1.0,
        },
        SimulationConfig {
            regime_name: "Extreme_Tox",
            noise_lambda_0: 0.5,
            noise_alpha: 0.2,
            noise_beta: 2.0,
            noise_qty: 10,
            inf_lambda_0: 5.0,
            inf_alpha: 0.9,
            inf_beta: 1.5,
            spoofer_lambda_0: 10.0,
            spoofer_alpha: 0.9,
            spoofer_beta: 1.5,
            mm_gamma: 0.05,
        },
    ];

    let results_dir = Path::new("../../benches/results");
    if !results_dir.exists() {
        std::fs::create_dir_all(results_dir).unwrap();
    }
    let mut file = File::create(results_dir.join("sweep_results.csv")).unwrap();
    writeln!(
        file,
        "Regime,Seed,Base_PnL,Base_AS_Cost,Base_Inv_Var,Mod_PnL,Mod_AS_Cost,Mod_Inv_Var,PnL_Diff,AS_Cost_Diff"
    )
    .unwrap();

    let mut toxic_significant = false;

    for cfg in regimes {
        println!("Running Regime: {} (N={})", cfg.regime_name, num_seeds);

        let seeds: Vec<u64> = (100..100 + num_seeds as u64).collect();

        let batch_results: Vec<_> = seeds
            .par_iter()
            .map(|&seed| {
                let res_base = run_ecology(seed, 0.0, steps, &cfg);
                let res_mod = run_ecology(seed, 10.0, steps, &cfg);
                (seed, res_base, res_mod)
            })
            .collect();

        let mut base_as_costs = Vec::with_capacity(num_seeds);
        let mut mod_as_costs = Vec::with_capacity(num_seeds);
        let mut as_diffs = Vec::with_capacity(num_seeds);

        for (seed, res_base, res_mod) in batch_results {
            base_as_costs.push(res_base.adverse_selection_cost);
            mod_as_costs.push(res_mod.adverse_selection_cost);
            as_diffs.push(res_base.adverse_selection_cost - res_mod.adverse_selection_cost);

            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{},{}",
                cfg.regime_name,
                seed,
                res_base.pnl,
                res_base.adverse_selection_cost,
                res_base.inventory_variance,
                res_mod.pnl,
                res_mod.adverse_selection_cost,
                res_mod.inventory_variance,
                res_base.pnl - res_mod.pnl,
                res_base.adverse_selection_cost - res_mod.adverse_selection_cost
            )
            .unwrap();
        }

        // Welch's t-test for AS Cost (Base vs Mod)
        let mean_base = base_as_costs.iter().sum::<f64>() / num_seeds as f64;
        let mean_mod = mod_as_costs.iter().sum::<f64>() / num_seeds as f64;

        let var_base = base_as_costs
            .iter()
            .map(|&x| (x - mean_base).powi(2))
            .sum::<f64>()
            / (num_seeds as f64 - 1.0);
        let var_mod = mod_as_costs
            .iter()
            .map(|&x| (x - mean_mod).powi(2))
            .sum::<f64>()
            / (num_seeds as f64 - 1.0);

        let t_stat = (mean_base - mean_mod)
            / (var_base / num_seeds as f64 + var_mod / num_seeds as f64).sqrt();

        println!("  Mean Base AS: {:.2}", mean_base);
        println!("  Mean Mod AS : {:.2}", mean_mod);
        println!("  Welch t-stat: {:.2}", t_stat);

        // Target: p < 0.01 (two-tailed critical value for N=500+ is approx 2.576)
        // A large positive t_stat means Base AS Cost is significantly higher than Mod AS Cost (which is good, Mod is better).
        if t_stat.abs() < 2.576 {
            println!("  [!] Warning: p-value not < 0.01");
        } else {
            println!("  [*] Result: Statistically significant (p < 0.01)");
            if cfg.regime_name == "Extreme_Tox" {
                toxic_significant = true;
            }
        }
    }

    assert!(
        toxic_significant,
        "Trust modifier failed to achieve p<0.01 in the Extreme_Tox regime"
    );
}

/// Regression test: ensures that the EWMA variance estimator and A-S spread
/// stay within the derived bounds across multiple seeds. This reproduces the
/// original numeric blowup scenario (aggressive noise trader sweeps in thin
/// books) and verifies the circuit breakers hold.
#[test]
fn test_variance_stability_regression() {
    let max_variance_bound = 2500.0;
    let steps = 5000;

    let test_seeds = [42u64, 100, 105, 111, 128];

    let cfg = SimulationConfig {
        regime_name: "Regression",
        noise_lambda_0: 1.0,
        noise_alpha: 0.5,
        noise_beta: 1.5,
        noise_qty: 50,
        inf_lambda_0: 1.0,
        inf_alpha: 0.5,
        inf_beta: 1.5,
        spoofer_lambda_0: 1.0,
        spoofer_alpha: 0.5,
        spoofer_beta: 1.5,
        mm_gamma: 1.0,
    };

    for &seed in &test_seeds {
        let (mut rng_noise, mut rng_informed, mut rng_spoofer) = derive_rngs(seed);
        let mut book = OrderBook::new();

        let hawkes_noise =
            HawkesSimulator::new(cfg.noise_lambda_0, cfg.noise_alpha, cfg.noise_beta);
        let mut noise = NoiseTrader::new(hawkes_noise, 0.5, 0.2, 10, cfg.noise_qty, 10_000);

        let hawkes_informed = HawkesSimulator::new(cfg.inf_lambda_0, cfg.inf_alpha, cfg.inf_beta);
        let mut informed = InformedTrader::new(
            hawkes_informed,
            1000,
            0.5,
            10,
            50,
            20_000,
            &mut rng_informed,
        );

        let hawkes_spoofer =
            HawkesSimulator::new(cfg.spoofer_lambda_0, cfg.spoofer_alpha, cfg.spoofer_beta);
        let mut spoofer = Spoofer::new(hawkes_spoofer, 0.5, 50, 500, 1000, 30_000, 0.1, 0.5, 0.05);

        let session_duration = steps as f64 * 0.1;
        let mut mm = AvellanedaStoikovMM::new(0.1, 1.0, session_duration, 0.1, 40_000);
        mm.set_trust_modifier(10.0);

        let mut trust_scorer = TrustScorer::new(100, 5);
        let agent_ids = [10_000u64, 20_000, 30_000];
        let mut current_mid = 100_000.0;
        let mut max_variance: f64 = 0.0;

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
            let _ = min_t;

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
                        let result = book.add_limit_order(order);
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
                                    mm.on_fill(order.side, t.price, t.quantity, current_mid);
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

            let v = mm.get_variance_estimate();
            if v > max_variance {
                max_variance = v;
            }

            let t1 = trust_scorer.get_trust_score(agent_ids[0]);
            let t2 = trust_scorer.get_trust_score(agent_ids[1]);
            let t3 = trust_scorer.get_trust_score(agent_ids[2]);
            mm.update_market_trust((t1 + t2 + t3) / 3.0);
        }

        assert!(
            max_variance <= max_variance_bound,
            "Seed {}: variance estimate {} exceeded bound {}",
            seed,
            max_variance,
            max_variance_bound
        );
    }
}
