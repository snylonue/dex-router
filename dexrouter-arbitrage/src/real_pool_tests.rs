//! Offline regressions against the optional local Ethereum snapshot.
use super::*;
use dexrouter_optim::{SolveAssessment, market::Market};

fn snapshot() -> Vec<RawUniswapV3Pool> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../uniswap_v3_pools.json");
    serde_json::from_reader(fs::File::open(path).expect("local pool snapshot")).unwrap()
}

#[test]
#[ignore = "requires local uniswap_v3_pools.json; real-pool residual regression"]
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
#[ignore = "requires local pool snapshot; diagnoses full-market floating-point precision"]
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
