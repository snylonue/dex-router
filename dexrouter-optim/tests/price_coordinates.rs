//! Convex reformulation and independent primal checks for stiff V2 liquidation.
use dexrouter_optim::{
    Route, SolveOptions,
    market::{Market, UniswapV2, UniswapV3},
    solve_price_with_options,
    utility::{BasketLiquidation, NonnegativeLinear},
};
use ndarray::{Array1, arr1};

fn chain(stiffness: f64, fee: f64) -> (Route<BasketLiquidation, UniswapV2>, Array1<f64>, f64) {
    let n = 12_usize;
    let units = Array1::from_iter((0..n).map(|i| {
        if i == 0 {
            1.0
        } else {
            10_f64.powi((i % 7) as i32 - 3)
        }
    }));
    let reserves: Vec<_> = (1..n)
        .map(|i| {
            if i % 2 == 0 {
                1000.0 * stiffness
            } else {
                1000.0
            }
        })
        .collect();
    let markets = (1..n)
        .map(|i| {
            (
                UniswapV2::new(
                    reserves[i - 1] / units[i - 1],
                    reserves[i - 1] / units[i],
                    fee,
                ),
                (i - 1, i),
            )
        })
        .collect();
    let mut inputs = Array1::zeros(n);
    inputs[n - 1] = 10.0 / units[n - 1];
    // Independent exact primal route: there is only one path to the output.
    let mut proceeds = 10.0;
    for &r in reserves.iter().rev() {
        proceeds = r * (fee * proceeds) / (r + fee * proceeds);
    }
    (
        Route {
            objective: BasketLiquidation { out: 0, inputs },
            markets,
            tokens: n,
        },
        units,
        proceeds,
    )
}

#[test]
fn sqrt_price_liquidation_matches_unique_primal_path() {
    for stiffness in [1.0, 1e3, 1e5] {
        for fee in [1.0, 0.997] {
            let (route, initial, proceeds) = chain(stiffness, fee);
            let result = solve_price_with_options(
                route.clone(),
                initial,
                SolveOptions {
                    sqrt_prices: true,
                    ..SolveOptions::default()
                },
            )
            .unwrap();
            println!(
                "sqrt chain stiffness={stiffness} fee={fee}: {:?} FG={} it={} KKT={}",
                result.assessment,
                result.evaluations,
                result.iterations,
                result.max_kkt_residual_ratio
            );
            assert!(result.is_usable(), "{result:?}");
            let mut net = route.objective.inputs.clone();
            for (m, (a, b)) in &route.markets {
                let (input, output) = m.arbitrage([result.price[*a], result.price[*b]]);
                net[*a] += output[0] - input[0];
                net[*b] += output[1] - input[1];
            }
            assert!(
                (net[0] - proceeds).abs() < 1e-6,
                "{net:?}, expected {proceeds}"
            );
            assert!((result.cost - proceeds).abs() < 1e-6);
            for i in 1..route.tokens {
                assert!(net[i].abs() * result.price[i] < 1e-6);
            }
            assert_eq!(result.price[0], 1.0);
        }
    }
}

#[test]
fn sqrt_price_arbitrage_matches_analytic_parallel_pool_optimum() {
    for unit in [1e-9, 1.0, 1e9] {
        let route = Route {
            objective: NonnegativeLinear {
                c: arr1(&[1.0, 0.0]),
            },
            markets: vec![
                (UniswapV2::new(1000.0, 2000.0 / unit, 1.0), (0, 1)),
                (UniswapV2::new(2000.0, 1000.0 / unit, 1.0), (0, 1)),
            ],
            tokens: 2,
        };
        let result = solve_price_with_options(
            route.clone(),
            arr1(&[1.5, 1.7 * unit]),
            SolveOptions {
                sqrt_prices: true,
                ..SolveOptions::default()
            },
        )
        .unwrap();
        assert!(result.is_usable(), "{result:?}");
        // Minimize 3000 + 3000 z^2 - 4000 sqrt(2) z, z=sqrt(p1/unit).
        assert!((result.price[1] / unit - 8.0 / 9.0).abs() < 1e-8);
        let mut net = [0.0, 0.0];
        for (market, _) in &route.markets {
            let (input, output) = market.arbitrage([result.price[0], result.price[1]]);
            for i in 0..2 {
                net[i] += output[i] - input[i];
            }
        }
        assert_eq!(result.price[0], 1.0);
        assert!(net[1].abs() * unit < 1e-6);
        assert!((net[0] - 1000.0 / 3.0).abs() < 1e-6);
        assert!((result.cost - 1000.0 / 3.0).abs() < 1e-8);
    }
}

#[test]
fn sqrt_price_mode_requires_explicit_model_guarantees() {
    let route = Route {
        objective: BasketLiquidation {
            out: 0,
            inputs: arr1(&[0.0, 1.0]),
        },
        markets: vec![(
            UniswapV3::new(1.0, vec![2.0, 0.5], vec![100.0], 0.997),
            (0, 1),
        )],
        tokens: 2,
    };
    assert!(
        solve_price_with_options(
            route,
            arr1(&[1.0, 1.0]),
            SolveOptions {
                sqrt_prices: true,
                ..SolveOptions::default()
            }
        )
        .is_err()
    );
    let (mut route, initial, _) = chain(1.0, 1.0);
    route.objective.inputs[1] = -1.0;
    assert!(
        solve_price_with_options(
            route,
            initial,
            SolveOptions {
                sqrt_prices: true,
                ..SolveOptions::default()
            }
        )
        .is_err()
    );
}

#[test]
fn sqrt_coordinates_share_hard_budget_and_warm_start_contract() {
    let (route, initial, _) = chain(1e3, 0.997);
    for budget in [1, 2, 3, 10, 50] {
        let mut options = SolveOptions {
            sqrt_prices: true,
            ..SolveOptions::default()
        };
        options.optimizer.max_evaluations = budget;
        let result = solve_price_with_options(route.clone(), initial.clone(), options).unwrap();
        assert!(result.evaluations <= budget);
    }
    let options = SolveOptions {
        sqrt_prices: true,
        ..SolveOptions::default()
    };
    let first = solve_price_with_options(route.clone(), initial, options.clone()).unwrap();
    let warm = solve_price_with_options(route, first.price.clone(), options).unwrap();
    assert!(first.is_usable() && warm.is_usable());
    assert_eq!(first.price, warm.price);
    assert_eq!(warm.evaluations, 1);
}

#[test]
#[ignore = "task-specific coordinate/solver ablation; run release with nocapture"]
fn diagnose_stiff_chain() {
    for stiffness in [1.0, 1e3, 1e5] {
        for sqrt_prices in [false, true] {
            for newton_polish in [false, true] {
                let (route, initial, proceeds) = chain(stiffness, 0.997);
                let start = std::time::Instant::now();
                let result = solve_price_with_options(
                    route,
                    initial,
                    SolveOptions {
                        sqrt_prices,
                        newton_polish,
                        ..SolveOptions::default()
                    },
                )
                .unwrap();
                println!(
                    "stiffness={stiffness} sqrt={sqrt_prices} polish={newton_polish} {:?} FG={} it={} kkt={:.3e} gap={:.3e} seconds={:.6}",
                    result.assessment,
                    result.evaluations,
                    result.iterations,
                    result.max_kkt_residual_ratio,
                    (result.cost - proceeds).abs(),
                    start.elapsed().as_secs_f64()
                );
            }
        }
    }
}
