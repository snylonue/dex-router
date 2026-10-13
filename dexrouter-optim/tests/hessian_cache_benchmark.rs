//! Opt-in, offline before/after benchmark; no wall-clock assertions.
use std::{fs::File, time::Instant};

use dexrouter_lbfgsb::{Bound, OptimizeResult, Options, minimize};
use dexrouter_optim::{
    snapshot::{ANCHOR_TOKEN, RawUniswapV3Pool, build_full_market_route_with_tvl},
    utility::UtilityConjugate,
};
use ndarray::Array1;

fn fingerprint(result: &OptimizeResult<f64>) -> u64 {
    // Include every price and gradient, not just the objective value.
    result
        .x
        .iter()
        .chain(&result.gradient)
        .chain([&result.f, &result.projected_gradient_inf_norm])
        .fold(0xcbf29ce484222325, |hash, value| {
            (hash ^ value.to_bits()).wrapping_mul(0x100000001b3)
        })
}

#[test]
#[ignore = "release-only offline Hessian cache performance comparison"]
fn fixed_anchor_cold_and_warm_start() {
    let snapshot = std::env::var("DEXROUTER_BENCH_SNAPSHOT").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/uniswap_v3_pools.json"
        )
        .to_owned()
    });
    let pools: Vec<RawUniswapV3Pool> =
        serde_json::from_reader(File::open(snapshot).unwrap()).unwrap();
    let mut full = build_full_market_route_with_tvl(pools, 0.0).unwrap();
    full.route.markets.truncate(600);
    let anchor = full
        .tokens
        .iter()
        .position(|name| name == ANCHOR_TOKEN)
        .unwrap();
    let lower = full.route.objective.lower_bounds();
    let upper = full.route.objective.upper_bounds();
    let bounds: Vec<_> = (0..full.route.tokens)
        .map(|i| {
            if i == anchor {
                Bound::both(1.0, 1.0)
            } else {
                Bound::new(
                    Some((lower[i] / full.initial_price[i]).max(1e-12)),
                    upper[i]
                        .is_finite()
                        .then_some(upper[i] / full.initial_price[i]),
                )
            }
        })
        .collect();
    let options = Options {
        max_iterations: 2000,
        max_evaluations: 50_000,
        ..Options::default()
    };
    println!(
        "pools={} tokens={} memory={} fixed_anchor={} max_iterations={}",
        full.route.markets.len(),
        full.route.tokens,
        options.memory,
        anchor,
        options.max_iterations
    );
    let solve = |initial: Vec<f64>| {
        minimize(initial, bounds.clone(), options.clone(), |x, g| {
            let price = Array1::from_vec(x.to_vec()) * &full.initial_price;
            let evaluation = full.route.evaluate(&price);
            for i in 0..g.len() {
                g[i] = evaluation.gradient[i] * full.initial_price[i];
            }
            Ok(evaluation.cost)
        })
        .unwrap()
    };
    let repetitions = std::env::var("DEXROUTER_BENCH_REPETITIONS")
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(3);
    assert!(repetitions > 0);
    let mut expected = None;
    for repetition in 0..repetitions {
        let start = Instant::now();
        let cold = solve(vec![1.0; full.route.tokens]);
        let cold_seconds = start.elapsed().as_secs_f64();
        let mut initial: Vec<_> = cold.x.iter().map(|x| x * 1.001).collect();
        initial[anchor] = 1.0;
        let start = Instant::now();
        let warm = solve(initial);
        let warm_seconds = start.elapsed().as_secs_f64();
        for (label, result, seconds) in
            [("cold", &cold, cold_seconds), ("warm", &warm, warm_seconds)]
        {
            assert!(result.f.is_finite());
            assert_eq!(result.x[anchor], 1.0);
            assert!(result.x.iter().zip(&bounds).all(|(&x, b)| x.is_finite()
                && b.lower.is_none_or(|l| x >= l)
                && b.upper.is_none_or(|u| x <= u)));
            println!(
                "run={repetition} {label}: seconds={seconds:.6} status={:?} iterations={} evaluations={} cost={:.17e} pg={:.17e} fingerprint={:016x}",
                result.status,
                result.iterations,
                result.evaluations,
                result.f,
                result.projected_gradient_inf_norm,
                fingerprint(result)
            );
        }
        let signature = (
            cold.status,
            cold.iterations,
            cold.evaluations,
            fingerprint(&cold),
            warm.status,
            warm.iterations,
            warm.evaluations,
            fingerprint(&warm),
        );
        if let Some(expected) = expected {
            assert_eq!(
                signature, expected,
                "repeated runs must be numerically identical"
            );
        } else {
            expected = Some(signature);
        }
    }
}
