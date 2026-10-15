//! Task-level comparison: identical physical starts, bounds and economic KKT.
use dexrouter_newton::Options as NewtonOptions;
use dexrouter_optim::{
    Route, SolveDiagnostics, SolveOptions,
    market::{Market, UniswapV2, UniswapV3},
    solve_price_with_newton, solve_price_with_options,
    utility::{BasketLiquidation, NonnegativeLinear, UtilityConjugate},
};
use ndarray::{Array1, arr1};

// Rebuild trades rather than trusting optimizer gradients or termination codes.
fn verify<U: UtilityConjugate, M: Market, S: std::fmt::Debug>(
    route: &Route<U, M>,
    r: &SolveDiagnostics<S>,
    o: &SolveOptions,
) {
    assert!(r.is_usable(), "{r:?}");
    let mut net = route.objective.grad(&r.price).to_vec();
    let mut turnover = vec![0.0; route.tokens];
    for (m, (a, b)) in &route.markets {
        let (input, output) = m.arbitrage([r.price[*a], r.price[*b]]);
        for (k, i) in [*a, *b].into_iter().enumerate() {
            net[i] += output[k] - input[k];
            turnover[i] += output[k] + input[k];
        }
    }
    let lower = route.objective.lower_bounds();
    let upper = route.objective.upper_bounds();
    for i in 0..route.tokens {
        assert!(r.price[i] >= lower[i] && r.price[i] <= upper[i]);
        let g = if (r.price[i] == lower[i] && net[i] >= 0.0)
            || (r.price[i] == upper[i] && net[i] <= 0.0)
        {
            0.0
        } else {
            net[i]
        };
        let allowance =
            o.absolute_residual_value + o.relative_residual_tolerance * turnover[i] * r.price[i];
        assert!(
            g.abs() * r.price[i] <= allowance * 1.001,
            "token={i} g={g} allowance={allowance} {r:?}"
        );
    }
}
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
fn stiff_liquidation_matches_independent_primal_and_lbfgsb() {
    for tree in [false, true] {
        for stiffness in [1.0, 1e3, 1e5] {
            for fee in [1.0, 0.997] {
                let (route, initial, proceeds) = chain(stiffness, fee);
                let o = SolveOptions {
                    newton_tree_precondition: tree,
                    ..Default::default()
                };
                let n = NewtonOptions {
                    pgtol: o.optimizer.pgtol,
                    max_iterations: 5000,
                    max_evaluations: 50000,
                    ..Default::default()
                };
                let result =
                    solve_price_with_newton(route.clone(), initial.clone(), o.clone(), n).unwrap();
                let baseline = solve_price_with_options(route.clone(), initial, o.clone()).unwrap();
                println!(
                    "chain stiffness={stiffness} fee={fee} newton={:?}/{:?} FG={} HVP={} KKT={} lbfgsb FG={} KKT={}",
                    result.status,
                    result.assessment,
                    result.evaluations,
                    result.hessian_products,
                    result.max_kkt_residual_ratio,
                    baseline.evaluations,
                    baseline.max_kkt_residual_ratio
                );
                verify(&route, &result, &o);
                verify(&route, &baseline, &o);
                assert!((result.cost - proceeds).abs() < 1e-6);
                assert!((baseline.cost - proceeds).abs() < 1e-6);
                let mut net = route.objective.inputs.clone();
                for (m, (a, b)) in &route.markets {
                    let (i, o) = m.arbitrage([result.price[*a], result.price[*b]]);
                    net[*a] += o[0] - i[0];
                    net[*b] += o[1] - i[1];
                }
                assert!((net[0] - proceeds).abs() < 1e-6);
                assert_eq!(result.price[0], 1.0);
            }
        }
    }
}
#[test]
fn analytic_arbitrage_is_invariant_to_token_units() {
    for tree in [false, true] {
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
            let o = SolveOptions {
                newton_tree_precondition: tree,
                ..Default::default()
            };
            let result = solve_price_with_newton(
                route.clone(),
                arr1(&[1.5, 1.7 * unit]),
                o.clone(),
                NewtonOptions {
                    pgtol: o.optimizer.pgtol,
                    max_iterations: o.optimizer.max_iterations,
                    max_evaluations: o.optimizer.max_evaluations,
                    ..Default::default()
                },
            )
            .unwrap();
            verify(&route, &result, &o);
            assert!((result.price[1] / unit - 8.0 / 9.0).abs() < 1e-8);
            assert!((result.cost - 1000.0 / 3.0).abs() < 1e-8);
        }
    }
}
#[test]
fn v3_tick_crossing_and_zero_liquidity_match_lbfgsb() {
    for tree in [false, true] {
        for amount in [1.0, 30.0, 100.0] {
            let route = Route {
                objective: BasketLiquidation {
                    out: 0,
                    inputs: arr1(&[0.0, amount]),
                },
                markets: vec![(
                    UniswapV3::new(
                        1.0,
                        vec![3.0, 1.2, 1.1, 0.8, 0.5],
                        vec![50.0, 0.0, 100.0, 200.0],
                        0.997,
                    ),
                    (0, 1),
                )],
                tokens: 2,
            };
            let o = SolveOptions {
                newton_tree_precondition: tree,
                ..Default::default()
            };
            let result = solve_price_with_newton(
                route.clone(),
                arr1(&[1.0, 1.0]),
                o.clone(),
                NewtonOptions {
                    pgtol: o.optimizer.pgtol,
                    max_iterations: o.optimizer.max_iterations,
                    max_evaluations: o.optimizer.max_evaluations,
                    ..Default::default()
                },
            )
            .unwrap();
            let baseline =
                solve_price_with_options(route.clone(), arr1(&[1.0, 1.0]), o.clone()).unwrap();
            println!(
                "v3 amount={amount} newton={:?}/{:?} FG={} HVP={} KKT={} lbfgsb FG={} cost_difference={}",
                result.status,
                result.assessment,
                result.evaluations,
                result.hessian_products,
                result.max_kkt_residual_ratio,
                baseline.evaluations,
                result.cost - baseline.cost
            );
            verify(&route, &result, &o);
            verify(&route, &baseline, &o);
            assert!((result.cost - baseline.cost).abs() < 1e-6);
        }
    }
}
#[test]
fn global_budgets_warm_starts_and_unsupported_modes() {
    for tree in [false, true] {
        let (route, initial, _) = chain(1.0, 0.997);
        for cap in [1, 2, 3, 10, 50] {
            let mut o = SolveOptions {
                newton_tree_precondition: tree,
                ..Default::default()
            };
            o.optimizer.max_evaluations = cap;
            let n = NewtonOptions {
                max_evaluations: cap,
                max_hessian_products: 3,
                ..Default::default()
            };
            let r = solve_price_with_newton(route.clone(), initial.clone(), o, n).unwrap();
            assert!(r.evaluations <= cap);
            assert!(r.hessian_products <= 3);
            if !r.is_usable() {
                assert!(r.max_kkt_residual_ratio > 1.0 || r.max_deficit_ratio > 1.0);
            }
        }
        let r = solve_price_with_newton(
            route.clone(),
            initial.clone(),
            SolveOptions {
                newton_tree_precondition: tree,
                ..Default::default()
            },
            NewtonOptions::default(),
        )
        .unwrap();
        let warm = solve_price_with_newton(
            route.clone(),
            r.price.clone(),
            SolveOptions {
                newton_tree_precondition: tree,
                ..Default::default()
            },
            NewtonOptions::default(),
        )
        .unwrap();
        assert!(r.is_usable() && warm.is_usable());
        assert_eq!(r.price, warm.price);
        assert_eq!(warm.evaluations, 1);
        assert_eq!(warm.hessian_products, 0);
        assert!(
            solve_price_with_newton(
                route,
                initial,
                SolveOptions {
                    sqrt_prices: true,
                    ..Default::default()
                },
                NewtonOptions::default()
            )
            .is_err()
        );
    }
}
