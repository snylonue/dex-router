//! Offline regressions against the bundled Ethereum pool snapshot.
use std::fs;

use dexrouter_optim::{
    SolveAssessment, SolveOptions,
    market::Market,
    snapshot::{RawUniswapV3Pool, build_full_market_route_with_tvl},
    solve_price_with_options,
    utility::UtilityConjugate,
};

const SNAPSHOT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/uniswap_v3_pools.json",
);

fn snapshot() -> Vec<RawUniswapV3Pool> {
    let path = SNAPSHOT_PATH;
    serde_json::from_reader(fs::File::open(path).expect("bundled pool snapshot")).unwrap()
}

#[test]
#[ignore = "expensive bundled-snapshot regression; run with --release --ignored"]
fn real_pool_residuals() {
    let pools = snapshot();
    for (label, cutoff, absolute_residual_value) in [
        ("filtered-strict", 1000.0, 1e-8),
        ("full-application", 0.0, 1e-6),
    ] {
        let full = build_full_market_route_with_tvl(pools.clone(), cutoff).unwrap();
        let options = SolveOptions {
            absolute_residual_value,
            ..SolveOptions::default()
        };
        let result = solve_price_with_options(
            full.route.clone(),
            full.initial_price.clone(),
            options.clone(),
        )
        .unwrap();
        println!(
            "{label}: pools={} tokens={} status={:?} assessment={:?} iterations={} evaluations={} cost={:.12e} kkt={:.3e} deficit={:.3e}",
            full.route.markets.len(),
            full.route.tokens,
            result.status,
            result.assessment,
            result.iterations,
            result.evaluations,
            result.cost,
            result.max_kkt_residual_ratio,
            result.max_deficit_value
        );
        assert!(result.is_usable(), "{label}: {result:?}");
        assert!(result.iterations <= options.optimizer.max_iterations);
        assert!(result.evaluations <= options.optimizer.max_evaluations);
        // Rebuild the trades independently of the solver diagnostics. Accumulate
        // the already rounded trades in f64 with Neumaier compensation.
        let mut net = vec![(0.0_f64, 0.0_f64); full.route.tokens];
        let mut turnover = vec![(0.0_f64, 0.0_f64); full.route.tokens];
        for (market, (a, b)) in &full.route.markets {
            let (input, output) = market.arbitrage([result.price[*a], result.price[*b]]);
            for (local, token) in [*a, *b].into_iter().enumerate() {
                add(&mut net[token], output[local] - input[local]);
                add(&mut turnover[token], output[local] + input[local]);
            }
        }
        let lower = full.route.objective.lower_bounds();
        let upper = full.route.objective.upper_bounds();
        let utility_grad = full.route.objective.grad(&result.price);
        for i in 0..full.route.tokens {
            let gradient = net[i].0 + net[i].1 + utility_grad[i];
            let price = result.price[i];
            assert!(price.is_finite() && price > 0.0 && price >= lower[i] && price <= upper[i]);
            let residual = if (price == lower[i] && gradient >= 0.0)
                || (price == upper[i] && gradient <= 0.0)
            {
                0.0
            } else {
                gradient
            };
            let allowance = options.absolute_residual_value
                + options.relative_residual_tolerance * (turnover[i].0 + turnover[i].1) * price;
            assert!(
                residual.abs() * price <= allowance,
                "{label} {}: residual_value={} allowance={allowance}",
                full.tokens[i],
                residual * price
            );
        }
    }
}

fn add((sum, correction): &mut (f64, f64), value: f64) {
    let next = *sum + value;
    *correction += if sum.abs() >= value.abs() {
        (*sum - next) + value
    } else {
        (value - next) + *sum
    };
    *sum = next;
}

#[test]
#[ignore = "expensive bundled-snapshot floating-point precision diagnosis"]
fn diagnose_full_market_price_precision() {
    let full = build_full_market_route_with_tvl(snapshot(), 0.0).unwrap();
    let options = SolveOptions::default();
    let result =
        solve_price_with_options(full.route.clone(), full.initial_price.clone(), options).unwrap();
    println!(
        "full strict: status={:?} assessment={:?} kkt={:.6e} deficit={:.6e}",
        result.status, result.assessment, result.max_kkt_residual_ratio, result.max_deficit_value
    );
    if !result.is_usable() {
        assert!(matches!(
            result.assessment,
            SolveAssessment::NumericalStagnation | SolveAssessment::BudgetExhausted
        ));
        assert!(result.max_kkt_residual_ratio > 1.0 || result.max_deficit_ratio > 1.0);
    }
    for symbol in ["wstETH", "weETH"] {
        let Some(i) = full.tokens.iter().position(|token| token == symbol) else {
            continue;
        };
        let center = result.price[i];
        for candidate in [center.next_down(), center, center.next_up()] {
            let mut price = result.price.clone();
            price[i] = candidate;
            let evaluation = full.route.evaluate(&price);
            println!(
                "{symbol} p={candidate:.17e}: gradient_value={:.6e} turnover_value={:.6e}",
                evaluation.gradient[i] * candidate,
                evaluation.turnover[i] * candidate
            );
        }
    }
}

#[test]
#[ignore = "offline solver ablation; run release with nocapture and one test thread"]
fn diagnose_stable_difference_ablation() {
    let pools = snapshot();
    for (label, cutoff, tolerance) in [("filtered", 1000.0, 1e-8), ("full", 0.0, 1e-6)] {
        let full = build_full_market_route_with_tvl(pools.clone(), cutoff).unwrap();
        for stable_differences in [false, true] {
            for newton_polish in [false, true] {
                let options = SolveOptions {
                    stable_differences,
                    newton_polish,
                    absolute_residual_value: tolerance,
                    ..SolveOptions::default()
                };
                let start = std::time::Instant::now();
                let result = solve_price_with_options(
                    full.route.clone(),
                    full.initial_price.clone(),
                    options,
                )
                .unwrap();
                println!(
                    "{label} differences={stable_differences} polish={newton_polish} status={:?} assessment={:?} FG={} iterations={} KKT={:.6e} seconds={:.6}",
                    result.status,
                    result.assessment,
                    result.evaluations,
                    result.iterations,
                    result.max_kkt_residual_ratio,
                    start.elapsed().as_secs_f64()
                );
                // Diagnostic, not an assertion that every ablation converges.
                assert!(result.price.iter().all(|p| p.is_finite() && *p > 0.0));
            }
        }
    }
}
