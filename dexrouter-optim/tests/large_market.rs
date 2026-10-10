//! Opt-in scale tests: cargo test -p dexrouter-optim --release --test large_market -- --ignored --nocapture
use std::{sync::Barrier, time::Instant};

use dexrouter_lbfgsb::Status;
use dexrouter_optim::{
    Route, SolveDiagnostics,
    market::{Market, MarketEvaluation, UniswapV2, UniswapV3},
    solve_price_with_init_p_checked,
    utility::BasketLiquidation,
};
use ndarray::Array1;

#[path = "support/convergence.rs"]
mod convergence;

#[derive(Clone)]
enum Pool {
    V2(UniswapV2),
    V3(UniswapV3),
}

impl Market for Pool {
    fn value_change(
        &self,
        v: [f64; 2],
        reference_v: [f64; 2],
        trade: MarketEvaluation,
        reference: MarketEvaluation,
    ) -> f64 {
        match self {
            Self::V2(pool) => pool.value_change(v, reference_v, trade, reference),
            Self::V3(pool) => pool.value_change(v, reference_v, trade, reference),
        }
    }
    fn evaluate(&self, v: [f64; 2]) -> MarketEvaluation {
        match self {
            Self::V2(pool) => pool.evaluate(v),
            Self::V3(pool) => pool.evaluate(v),
        }
    }
    fn arbitrage(&self, v: [f64; 2]) -> ([f64; 2], [f64; 2]) {
        match self {
            Self::V2(pool) => pool.arbitrage(v),
            Self::V3(pool) => pool.arbitrage(v),
        }
    }
}

fn market(
    tokens: usize,
    pools: usize,
    capital: f64,
) -> (Route<BasketLiquidation, Pool>, Array1<f64>) {
    let prices = Array1::from_iter((0..tokens).map(|i| {
        if i == 0 {
            1.0
        } else {
            10.0_f64.powf((i % 19) as f64 / 3.0 - 3.0)
        }
    }));
    let base = 1000.0 * capital;
    let markets = (0..pools)
        .map(|i| {
            // A connected star backbone plus cross-token pools. Repeated pairs
            // have different liquidity; this is not duplication of one test pool.
            let a = if i % 3 == 0 { 0 } else { 1 + i % (tokens - 1) };
            let mut b = 1 + (i * 17 + i / tokens) % (tokens - 1);
            if a == b {
                b = (b + 1) % tokens;
            }
            let value = base * (1.0 + (i % 11) as f64 * 0.1);
            let pool = if i % 2 == 0 {
                Pool::V2(UniswapV2::new(value / prices[a], value / prices[b], 0.997))
            } else {
                let sqrt_price = (prices[a] / prices[b]).sqrt();
                let liquidity = value / (prices[a] * prices[b]).sqrt();
                let ticks: Vec<_> = (0..33)
                    .map(|tick| sqrt_price * 2.0_f64.powf(1.0 - tick as f64 / 16.0))
                    .collect();
                let mut liquidities = vec![liquidity; 32];
                liquidities.push(0.0);
                Pool::V3(UniswapV3::new(sqrt_price, ticks, liquidities, 0.997))
            };
            (pool, (a, b))
        })
        .collect();
    let inputs = Array1::from_iter(
        prices
            .iter()
            .enumerate()
            .map(|(i, price)| if i == 0 { 0.0 } else { base * 0.1 / price }),
    );
    let initial = Array1::from_iter(
        prices
            .iter()
            .enumerate()
            .map(|(i, price)| price * (1.05 + (i % 7) as f64 * 0.005)),
    );
    (
        Route {
            objective: BasketLiquidation { out: 0, inputs },
            markets,
            tokens,
        },
        initial,
    )
}

fn solve_and_validate(tokens: usize, pools: usize, capital: f64) -> SolveDiagnostics {
    let (route, initial) = market(tokens, pools, capital);
    let supply_value = route.objective.inputs.dot(&initial);
    let start = Instant::now();
    let result = solve_price_with_init_p_checked(route.clone(), initial).unwrap();
    let seconds = start.elapsed().as_secs_f64();
    // Accumulate net flows directly: avoid the pools x tokens dense matrices
    // returned by Route::arbitrage for a large sparse market graph.
    let mut net = Array1::<f64>::zeros(tokens);
    for (pool, (a, b)) in &route.markets {
        let (input, output) = pool.arbitrage([result.price[*a], result.price[*b]]);
        net[*a] += output[0] - input[0];
        net[*b] += output[1] - input[1];
    }
    let max_relative_residual = (1..tokens)
        .map(|i| (net[i] + route.objective.inputs[i]).abs() * result.price[i] / supply_value)
        .fold(0.0, f64::max);
    let max_token_residual = (1..tokens)
        .map(|i| (net[i] + route.objective.inputs[i]).abs() / route.objective.inputs[i])
        .fold(0.0, f64::max);
    let gap = (result.cost - net[0]).abs() / supply_value;
    println!(
        "tokens={tokens} pools={pools} capital={capital:.0e} seconds={seconds:.4} status={:?} iterations={} evaluations={} scaled_gradient={:.3e} relative_flow_residual={max_relative_residual:.3e} relative_token_flow_residual={max_token_residual:.3e} relative_duality_gap={gap:.3e}",
        result.status, result.iterations, result.evaluations, result.max_scaled_projected_gradient
    );
    // Validate economic residuals independently of the solver's termination
    // reason. The callers check strict termination after measuring all cases.
    assert!((result.price[0] - 1.0).abs() < 1e-8);
    assert!(net[0] > 0.0);
    assert!(
        max_relative_residual < 1e-6,
        "relative net flow residual={max_relative_residual}"
    );
    assert!(
        max_token_residual < 1e-6,
        "relative per-token net flow residual={max_token_residual}"
    );
    assert!(gap < 1e-6, "relative primal-dual gap={gap}");
    result
}

#[test]
#[ignore = "large mixed-pool scaling benchmark"]
fn large_mixed_markets() {
    let statuses: Vec<_> = [(32, 512), (128, 4096), (256, 16384)]
        .into_iter()
        .map(|(tokens, pools)| (tokens, pools, solve_and_validate(tokens, pools, 1.0).status))
        .collect();
    assert!(
        statuses.iter().all(|(_, _, status)| matches!(
            status,
            Status::ConvergedProjectedGradient | Status::ConvergedFunctionReduction
        )),
        "strict solver termination failed: {statuses:?}"
    );
}

#[test]
#[ignore = "large reserve/liquidity scaling benchmark"]
fn high_capital_market() {
    let ordinary = solve_and_validate(128, 4096, 1.0);
    let large = solve_and_validate(128, 4096, 1e6);
    let max_relative_price_difference = ordinary
        .price
        .iter()
        .zip(&large.price)
        .map(|(&a, &b)| (a - b).abs() / a)
        .fold(0.0, f64::max);
    println!(
        "capital_scale_invariance: max_relative_price_difference={max_relative_price_difference:.3e}"
    );
    assert!(max_relative_price_difference < 1e-5);
    assert!(
        [ordinary.status, large.status]
            .iter()
            .all(|status| matches!(
                status,
                Status::ConvergedProjectedGradient | Status::ConvergedFunctionReduction
            )),
        "strict solver termination failed: ordinary={:?}, high_capital={:?}",
        ordinary.status,
        large.status
    );
}

#[test]
#[ignore = "controlled optimizer, objective scaling and precision diagnosis"]
fn diagnose_synthetic_convergence() {
    use convergence::{Config, check_gradient, run};
    for capital in [1.0, 1e6] {
        let (route, initial) = market(128, 4096, capital);
        let scale = route.utility(&initial).max(1.0);
        println!("synthetic capital={capital:.0e} initial_cost={scale:.12e}");
        check_gradient(&route, &initial, "synthetic-initial");
        let configs = [
            Config::default(),
            Config {
                label: "normalized",
                cost_scale: scale,
                ..Config::default()
            },
            Config {
                label: "factr-1e7",
                factr: 1e7,
                ..Config::default()
            },
            Config {
                label: "stable-sum",
                stable_sum: true,
                ..Config::default()
            },
            Config {
                label: "maxls-100",
                maxls: 100,
                ..Config::default()
            },
        ];
        for config in configs {
            run(&route, &initial, config);
        }
    }
}

#[test]
#[ignore = "four concurrent large-market solves"]
fn concurrent_large_market() {
    let (route, initial) = market(128, 4096, 1.0);
    let expected = solve_price_with_init_p_checked(route.clone(), initial.clone()).unwrap();
    let barrier = Barrier::new(4);
    let start = Instant::now();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    let actual =
                        solve_price_with_init_p_checked(route.clone(), initial.clone()).unwrap();
                    assert_eq!(actual.price, expected.price);
                    assert_eq!(actual.gradient, expected.gradient);
                    assert_eq!(actual.cost, expected.cost);
                    assert_eq!(actual.status, expected.status);
                    assert_eq!(actual.iterations, expected.iterations);
                    assert_eq!(actual.evaluations, expected.evaluations);
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    });
    println!(
        "concurrent: workers=4 tokens=128 pools=4096 seconds={:.4} status={:?}",
        start.elapsed().as_secs_f64(),
        expected.status
    );
}
