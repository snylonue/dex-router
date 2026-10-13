//! Equivalent preprocessing must retain the economic objective and optimal trades.
use dexrouter_optim::{
    Route, SolveOptions,
    market::{Market, UniswapV2, UniswapV3},
    solve_price_with_options,
    utility::{BasketLiquidation, NonnegativeLinear, UtilityConjugate},
};
use ndarray::{Array1, arr1};

fn basket() -> Route<BasketLiquidation, UniswapV2> {
    Route {
        objective: BasketLiquidation {
            out: 0,
            inputs: arr1(&[0.0, 10.0, 20.0, 0.0]),
        },
        markets: vec![
            (UniswapV2::new(1000.0, 2000.0, 0.997), (0, 1)),
            (UniswapV2::new(700.0, 3000.0, 0.997), (0, 2)),
        ],
        tokens: 4,
    }
}

#[test]
fn eliminate_anchor_and_inert_token_and_split_at_fixed_vertices() {
    let route = basket();
    let initial = arr1(&[1.3, 0.4, 0.3, 17.0]);
    let result =
        solve_price_with_options(route.clone(), initial.clone(), SolveOptions::default()).unwrap();
    let baseline = solve_price_with_options(
        route.clone(),
        initial,
        SolveOptions {
            decompose: false,
            analytic_curvature: false,
            graph_precondition: false,
            ..SolveOptions::default()
        },
    )
    .unwrap();
    assert!(result.is_usable(), "{result:?}");
    assert!(baseline.is_usable(), "{baseline:?}");
    assert_eq!(result.optimizer_variables, 2);
    assert_eq!(result.optimization_blocks, 2);
    assert_eq!(baseline.optimization_blocks, 1);
    assert_eq!(result.price[0], 1.0);
    assert_eq!(result.price[3], 17.0);
    assert!((result.cost - baseline.cost).abs() < 1e-7);
    let (input, output) = route.arbitrage(result.price);
    let (old_input, old_output) = route.arbitrage(baseline.price);
    assert!(input.abs_diff_eq(&old_input, 1e-5));
    assert!(output.abs_diff_eq(&old_output, 1e-5));
    assert_eq!(route.markets.len(), 2);
    assert_eq!(route.objective.lower_bounds(), arr1(&[1.0, 0.0, 0.0, 0.0]));
}

#[test]
fn accepted_solution_is_an_exact_zero_iteration_warm_start() {
    let route = basket();
    let first = solve_price_with_options(
        route.clone(),
        arr1(&[1.3, 0.4, 0.3, 17.0]),
        SolveOptions::default(),
    )
    .unwrap();
    assert!(first.is_usable());
    let warm =
        solve_price_with_options(route.clone(), first.price.clone(), SolveOptions::default())
            .unwrap();
    assert!(warm.is_usable());
    assert_eq!(warm.price, first.price);
    assert_eq!(warm.cost, first.cost);
    assert_eq!(warm.gradient, first.gradient);
    assert_eq!(warm.iterations, 0);
    assert_eq!(warm.evaluations, 1);
    assert_eq!(route.arbitrage(warm.price), route.arbitrage(first.price));
}

struct Fixed;
impl UtilityConjugate for Fixed {
    fn value(&self, p: &Array1<f64>) -> f64 {
        p[0] - 2.0 * p[1]
    }
    fn grad(&self, _: &Array1<f64>) -> Array1<f64> {
        arr1(&[1.0, -2.0])
    }
    fn lower_bounds(&self) -> Array1<f64> {
        arr1(&[1.0, 2.0])
    }
    fn upper_bounds(&self) -> Array1<f64> {
        arr1(&[1.0, 2.0])
    }
}

#[test]
fn all_fixed_coordinates_need_no_optimizer_call() {
    let route: Route<_, UniswapV2> = Route {
        objective: Fixed,
        markets: vec![],
        tokens: 2,
    };
    let result =
        solve_price_with_options(route, arr1(&[3.0, 4.0]), SolveOptions::default()).unwrap();
    assert!(result.is_usable());
    assert_eq!(result.price, arr1(&[1.0, 2.0]));
    assert_eq!(result.gradient, arr1(&[1.0, -2.0]));
    assert_eq!(result.optimizer_variables, 0);
    assert_eq!(result.optimization_blocks, 0);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.evaluations, 1);
}

#[test]
fn disconnected_components_do_not_acquire_artificial_anchors() {
    let route = Route {
        objective: NonnegativeLinear {
            c: arr1(&[1.0, 0.0, 2.0, 0.0, 0.0]),
        },
        markets: vec![
            (UniswapV2::new(100.0, 200.0, 1.0), (0, 1)),
            (UniswapV2::new(100.0, 200.0, 1.0), (0, 1)),
            (UniswapV2::new(100.0, 300.0, 1.0), (2, 3)),
        ],
        tokens: 5,
    };
    let result = solve_price_with_options(
        route,
        arr1(&[3.0, 0.4, 7.0, 0.3, 19.0]),
        SolveOptions::default(),
    )
    .unwrap();
    assert!(result.is_usable(), "{result:?}");
    assert_eq!(result.optimization_blocks, 2);
    assert_eq!(result.optimizer_variables, 4);
    assert_eq!(result.price[4], 19.0);
    // Zero-arbitrage prices need not sit on utility lower bounds. The solver
    // must not force either component's reference token to 1 or to its bound.
    assert!(result.price[0] >= 1.0);
    assert!(result.price[2] >= 2.0);
    assert!((result.price[0] / result.price[1] - 2.0).abs() < 1e-7);
    assert!((result.price[2] / result.price[3] - 3.0).abs() < 1e-7);
}

struct Coupled;
impl UtilityConjugate for Coupled {
    fn value(&self, p: &Array1<f64>) -> f64 {
        let a = p[0] - 2.0;
        let b = p[1] - 3.0;
        a * a + b * b - a * b
    }
    fn grad(&self, p: &Array1<f64>) -> Array1<f64> {
        let a = p[0] - 2.0;
        let b = p[1] - 3.0;
        arr1(&[2.0 * a - b, 2.0 * b - a])
    }
    fn lower_bounds(&self) -> Array1<f64> {
        arr1(&[0.0, 0.0])
    }
    fn upper_bounds(&self) -> Array1<f64> {
        arr1(&[f64::INFINITY; 2])
    }
}

#[test]
fn unknown_coupled_utility_is_never_split_or_treated_as_inert() {
    let route: Route<_, UniswapV2> = Route {
        objective: Coupled,
        markets: vec![],
        tokens: 2,
    };
    let result =
        solve_price_with_options(route, arr1(&[1.0, 1.0]), SolveOptions::default()).unwrap();
    assert!(result.is_usable(), "{result:?}");
    assert_eq!(result.optimization_blocks, 1);
    assert_eq!(result.optimizer_variables, 2);
    assert!(result.price.abs_diff_eq(&arr1(&[2.0, 3.0]), 1e-7));
}

#[test]
fn analytic_curvature_removes_probe_evaluations() {
    let route = basket();
    let initial = arr1(&[1.3, 0.4, 0.3, 17.0]);
    let mut options = SolveOptions::default();
    options.optimizer.max_iterations = 1;
    options.decompose = false;
    let analytic =
        solve_price_with_options(route.clone(), initial.clone(), options.clone()).unwrap();
    options.analytic_curvature = false;
    let numerical = solve_price_with_options(route, initial, options).unwrap();
    assert!(
        analytic.evaluations < numerical.evaluations,
        "{analytic:?} vs {numerical:?}"
    );
}

fn check_pool_curvature(pool: &impl Market, prices: &[[f64; 2]]) {
    for &p in prices {
        let w = pool.relative_curvature(p).unwrap();
        let direction = [0.3 * p[0], -0.2 * p[1]];
        let h = 1e-5;
        let mut plus = p;
        let mut minus = p;
        for i in 0..2 {
            plus[i] += h * direction[i];
            minus[i] -= h * direction[i];
        }
        let hi = pool.evaluate(plus);
        let lo = pool.evaluate(minus);
        let hv = [
            ((hi.output[0] - hi.input[0]) - (lo.output[0] - lo.input[0])) / (2.0 * h),
            ((hi.output[1] - hi.input[1]) - (lo.output[1] - lo.input[1])) / (2.0 * h),
        ];
        let expected = [w * 0.5 / p[0], -w * 0.5 / p[1]];
        for i in 0..2 {
            assert!(
                (hv[i] - expected[i]).abs() <= 1e-7 * expected[i].abs().max(1.0),
                "p={p:?} hv={hv:?} expected={expected:?}"
            );
        }
    }
}

#[test]
fn analytic_v2_curvature_matches_gradient_derivatives_in_both_directions() {
    check_pool_curvature(
        &UniswapV2::new(100.0, 200.0, 0.997),
        &[[0.7, 1.0], [4.0, 1.0], [2.0, 1.0]],
    );
    check_pool_curvature(&UniswapV2::new(100.0, 200.0, 1.0), &[[2.0, 1.0]]);
}

#[test]
fn analytic_v3_curvature_handles_ticks_gaps_and_exhaustion() {
    let pool = UniswapV3::new(
        1.0,
        vec![2.0, 1.2, 0.8, 0.6, 0.4],
        vec![300.0, 200.0, 0.0, 100.0, 0.0],
        0.997,
    );
    check_pool_curvature(
        &pool,
        &[
            [0.25, 1.0],
            [0.49, 1.0],
            [0.81, 1.0],
            [1.0, 1.0],
            [1.21, 1.0],
            [2.25, 1.0],
            [9.0, 1.0],
        ],
    );
    check_pool_curvature(
        &UniswapV3::new(1.0, vec![2.0, 0.5], vec![100.0, 0.0], 1.0),
        &[[1.0, 1.0]],
    );
}

#[test]
fn decomposition_and_graph_recovery_respect_global_budgets() {
    for budget in [1, 2, 3, 5, 10, 30, 100] {
        let mut options = SolveOptions::default();
        options.optimizer.max_evaluations = budget;
        options.restart_interval = 1;
        options.optimizer.max_iterations = 10;
        let result =
            solve_price_with_options(basket(), arr1(&[1.3, 0.4, 0.3, 17.0]), options).unwrap();
        assert!(result.evaluations <= budget, "budget={budget}: {result:?}");
        assert!(result.iterations <= 10);
    }
}
