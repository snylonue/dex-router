use argmin::core::{CostFunction, Gradient};
use dexrouter_lbfgsb::Status;
use dexrouter_optim::{
    Route,
    market::UniswapV2,
    solve_price_with_init_p_checked,
    utility::{BasketLiquidation, UtilityConjugate},
};
use ndarray::{Array1, Axis, arr1};
use std::sync::Barrier;

#[test]
fn checked_solver_liquidates_with_nonuniform_initial_prices() {
    let route = Route {
        objective: BasketLiquidation {
            out: 0,
            inputs: arr1(&[0.0, 10.0]),
        },
        markets: vec![(UniswapV2::new(1000.0, 2000.0, 0.997), (0, 1))],
        tokens: 2,
    };
    let result = solve_price_with_init_p_checked(route.clone(), arr1(&[1.3, 0.4])).unwrap();
    let expected_price = 1000.0 * 2000.0 * 0.997 / (2000.0_f64 + 10.0 * 0.997).powi(2);
    assert!((result.price[0] - 1.0).abs() < 1e-8);
    assert!((result.price[1] - expected_price).abs() < 1e-8);
    // Economic acceptance is independent of the generic termination status.
    assert!(result.is_usable(), "{result:?}");
    assert!(result.max_kkt_residual_ratio <= 1.0);
    assert!(result.max_deficit_ratio <= 1.0);
    assert!(result.iterations > 0);
    assert!(result.evaluations > result.iterations);
    assert_eq!(result.cost, route.cost(&result.price).unwrap());
    assert_eq!(result.gradient, route.gradient(&result.price).unwrap());
    assert!(result.max_scaled_projected_gradient <= 1e-5);
    let (inputs, outputs) = route.arbitrage(result.price);
    let net = (outputs - inputs).sum_axis(Axis(0));
    assert!((net[1] + 10.0).abs() < 3e-5);
    assert!(net[0] > 0.0);
}

#[test]
fn concurrent_route_solves_match_serial_results() {
    let cases: Vec<_> = (0..16)
        .map(|case| {
            let route = Route {
                objective: BasketLiquidation {
                    out: 0,
                    inputs: arr1(&[0.0, 1.0 + case as f64]),
                },
                markets: vec![(UniswapV2::new(1000.0, 2000.0, 0.997), (0, 1))],
                tokens: 2,
            };
            let initial = arr1(&[1.3, 0.4 + case as f64 * 0.01]);
            let expected = solve_price_with_init_p_checked(route.clone(), initial.clone()).unwrap();
            assert!(
                expected.is_usable(),
                "{:?}: KKT ratio {}",
                expected.assessment,
                expected.max_kkt_residual_ratio
            );
            (route, initial, expected)
        })
        .collect();
    let barrier = Barrier::new(8);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    for (route, initial, expected) in &cases {
                        barrier.wait();
                        let actual =
                            solve_price_with_init_p_checked(route.clone(), initial.clone())
                                .unwrap();
                        assert_eq!(actual.price, expected.price);
                        assert_eq!(actual.cost, expected.cost);
                        assert_eq!(actual.gradient, expected.gradient);
                        assert_eq!(actual.status, expected.status);
                        assert_eq!(actual.assessment, expected.assessment);
                        assert_eq!(
                            actual.max_kkt_residual_ratio,
                            expected.max_kkt_residual_ratio
                        );
                        assert_eq!(actual.max_deficit_ratio, expected.max_deficit_ratio);
                        assert_eq!(actual.iterations, expected.iterations);
                        assert_eq!(actual.evaluations, expected.evaluations);
                        assert_eq!(actual.max_abs_gradient, expected.max_abs_gradient);
                        assert_eq!(
                            actual.max_projected_gradient,
                            expected.max_projected_gradient
                        );
                        assert_eq!(
                            actual.max_scaled_projected_gradient,
                            expected.max_scaled_projected_gradient
                        );
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    });
}

// A normalized quadratic isolates the adapter's price/gradient scaling and
// lower, upper, and unbounded coordinate handling from market formulas.
#[derive(Clone)]
struct Quadratic {
    lower: Array1<f64>,
    upper: Array1<f64>,
}

impl UtilityConjugate for Quadratic {
    fn value(&self, v: &Array1<f64>) -> f64 {
        let d = v / &arr1(&[2.0, 1000.0, 0.01]) - arr1(&[0.5, 3.0, 6.0]);
        0.5 * d.dot(&d)
    }

    fn grad(&self, v: &Array1<f64>) -> Array1<f64> {
        let scale = arr1(&[2.0, 1000.0, 0.01]);
        (v / &scale - arr1(&[0.5, 3.0, 6.0])) / scale
    }

    fn lower_bounds(&self) -> Array1<f64> {
        self.lower.clone()
    }

    fn upper_bounds(&self) -> Array1<f64> {
        self.upper.clone()
    }
}

fn quadratic_route() -> Route<Quadratic, UniswapV2> {
    Route {
        objective: Quadratic {
            lower: arr1(&[4.0, 0.0, f64::NEG_INFINITY]),
            upper: arr1(&[f64::INFINITY, 2000.0, f64::INFINITY]),
        },
        markets: vec![],
        tokens: 3,
    }
}

#[test]
fn checked_solver_scales_bounds_and_projects_infeasible_start() {
    let result =
        solve_price_with_init_p_checked(quadratic_route(), arr1(&[2.0, 1000.0, 0.01])).unwrap();
    assert!(result.price.abs_diff_eq(&arr1(&[4.0, 2000.0, 0.06]), 1e-8));
    assert_eq!(result.status, Status::ConvergedProjectedGradient);
    assert!((result.cost - 1.625).abs() < 1e-8);
    assert!((result.max_abs_gradient - 0.75).abs() < 1e-8);
    assert!(result.max_projected_gradient < 1e-8);
    assert!(result.max_scaled_projected_gradient < 1e-8);
}

#[test]
fn checked_solver_rejects_invalid_initial_prices() {
    for initial in [
        arr1(&[1.0, 2.0]),
        arr1(&[0.0, 1.0, 1.0]),
        arr1(&[-1.0, 1.0, 1.0]),
        arr1(&[f64::NAN, 1.0, 1.0]),
        arr1(&[f64::INFINITY, 1.0, 1.0]),
    ] {
        assert!(solve_price_with_init_p_checked(quadratic_route(), initial).is_err());
    }
}

#[test]
fn checked_solver_rejects_invalid_bounds() {
    let initial = arr1(&[2.0, 1000.0, 0.01]);
    let mut route = quadratic_route();
    route.objective.upper = arr1(&[1.0]);
    assert!(solve_price_with_init_p_checked(route, initial.clone()).is_err());

    let mut route = quadratic_route();
    route.objective.upper[0] = 3.0;
    assert!(solve_price_with_init_p_checked(route, initial.clone()).is_err());

    let mut route = quadratic_route();
    route.objective.lower[0] = f64::NAN;
    assert!(solve_price_with_init_p_checked(route, initial).is_err());
}
