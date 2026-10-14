use dexrouter_lbfgsb::Status;
use dexrouter_optim::{
    Route, SolveAssessment, SolveOptions,
    market::{Market, UniswapV2},
    solve_price_with_options,
    utility::{BasketLiquidation, NonnegativeLinear, UtilityConjugate},
};
use ndarray::arr1;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn liquidation() -> Route<BasketLiquidation, UniswapV2> {
    Route {
        objective: BasketLiquidation {
            out: 0,
            inputs: arr1(&[0.0, 10.0]),
        },
        markets: vec![(UniswapV2::new(1000.0, 2000.0, 0.997), (0, 1))],
        tokens: 2,
    }
}

#[test]
fn mathematical_bounds_are_exact_and_numerical_guard_is_relative() {
    let objective = NonnegativeLinear {
        c: arr1(&[1.0, 0.0, 0.0]),
    };
    assert_eq!(objective.lower_bounds(), arr1(&[1.0, 0.0, 0.0]));
    let route: Route<_, UniswapV2> = Route {
        objective,
        markets: vec![],
        tokens: 3,
    };
    let result =
        solve_price_with_options(route, arr1(&[1.3, 1e-16, 1e12]), SolveOptions::default())
            .unwrap();
    assert_eq!(result.price[0], 1.0);
    assert_eq!(result.numerical_lower_bounds[1], 1e-16 * 1e-12);
    assert_eq!(result.numerical_lower_bounds[2], 1.0);
    assert!(result.is_usable());
}

#[test]
fn multiple_nonzero_utility_coefficients_do_not_fix_an_anchor() {
    let objective = NonnegativeLinear {
        c: arr1(&[1.0, 2.0]),
    };
    assert_eq!(objective.price_anchor(), None);
    let route: Route<_, UniswapV2> = Route {
        objective,
        markets: vec![],
        tokens: 2,
    };
    let result =
        solve_price_with_options(route, arr1(&[1.3, 3.0]), SolveOptions::default()).unwrap();
    assert_eq!(result.price, arr1(&[1.3, 3.0]));
}

#[test]
fn generic_convergence_does_not_override_economic_rejection() {
    let mut options = SolveOptions::default();
    options.precondition = false;
    options.optimizer.pgtol = 1e12;
    let result = solve_price_with_options(liquidation(), arr1(&[1.3, 0.4]), options).unwrap();
    assert_eq!(result.status, Status::ConvergedProjectedGradient);
    assert_eq!(result.assessment, SolveAssessment::NumericalStagnation);
    assert!(!result.is_usable());
    assert!(result.max_deficit_ratio > 1.0);
}

#[test]
fn evaluation_budget_includes_probes_and_verification() {
    for budget in [1, 2, 3, 5, 10, 30] {
        let mut options = SolveOptions::default();
        options.optimizer.max_evaluations = budget;
        let result = solve_price_with_options(liquidation(), arr1(&[1.3, 0.4]), options).unwrap();
        assert!(result.evaluations <= budget, "budget={budget}: {result:?}");
        if !result.is_usable() {
            assert!(result.max_kkt_residual_ratio > 1.0 || result.max_deficit_ratio > 1.0);
        }
    }
}

#[test]
fn warm_budget_is_finite_and_preserves_strict_configuration() {
    let mut options = SolveOptions::default();
    options.optimizer.max_iterations = usize::MAX;
    options.optimizer.max_evaluations = usize::MAX;
    options.optimizer.pgtol = 1e-9;
    options.relative_price_floor = 1e-15;
    options.relative_residual_tolerance = 1e-12;
    options.absolute_residual_value = 1e-18;
    let warm = options.clone().with_warm_start_budget();
    assert_eq!(warm.optimizer.max_iterations, 2000);
    assert_eq!(warm.optimizer.max_evaluations, 2000);
    // Only the two hard budgets may change.
    options.optimizer.max_iterations = 2000;
    options.optimizer.max_evaluations = 2000;
    assert_eq!(format!("{warm:?}"), format!("{options:?}"));
    let result = solve_price_with_options(liquidation(), arr1(&[1.3, 0.4]), warm).unwrap();
    assert!(result.evaluations <= 2000);
    assert!(result.price.iter().all(|p| p.is_finite() && *p > 0.0));
    if result.is_usable() {
        assert!(result.max_kkt_residual_ratio <= 1.0 && result.max_deficit_ratio <= 1.0);
    }
    options.optimizer.max_iterations = 3;
    options.optimizer.max_evaluations = 5;
    let warm = options.with_warm_start_budget();
    assert_eq!(warm.optimizer.max_iterations, 3);
    assert_eq!(warm.optimizer.max_evaluations, 5);
}

// A convex, stiff coupled objective models coordinate recovery near a sensitive
// fee boundary: each sweep moves prices ~1e-12 but leaves a large residual.
#[derive(Clone)]
struct StiffCoupled;
impl UtilityConjugate for StiffCoupled {
    fn value(&self, p: &ndarray::Array1<f64>) -> f64 {
        0.5e12 * (p[0] - p[1]).powi(2) + 0.5 * ((p[0] - 2.0).powi(2) + (p[1] - 2.0).powi(2))
    }
    fn grad(&self, p: &ndarray::Array1<f64>) -> ndarray::Array1<f64> {
        let coupling = 1e12 * (p[0] - p[1]);
        arr1(&[coupling + p[0] - 2.0, -coupling + p[1] - 2.0])
    }
    fn lower_bounds(&self) -> ndarray::Array1<f64> {
        arr1(&[0.0, 0.0])
    }
    fn upper_bounds(&self) -> ndarray::Array1<f64> {
        arr1(&[f64::INFINITY, f64::INFINITY])
    }
}

#[test]
fn tiny_coordinate_recoveries_do_not_restart_forever() {
    let route: Route<_, UniswapV2> = Route {
        objective: StiffCoupled,
        markets: vec![],
        tokens: 2,
    };
    let mut options = SolveOptions::default();
    options.graph_precondition = false;
    // Isolate recovery from L-BFGS steps. The previous watchdog was bypassed
    // whenever a sweep changed either price, even with zero solver iterations.
    options.optimizer.pgtol = 1e12;
    options.optimizer.max_iterations = 20_000;
    options.optimizer.max_evaluations = 20_000;
    let result = solve_price_with_options(route.clone(), arr1(&[1.0, 1.0]), options).unwrap();
    assert_eq!(
        result.assessment,
        SolveAssessment::NumericalStagnation,
        "{result:?}"
    );
    assert_eq!(result.status, Status::ConvergedProjectedGradient);
    assert!(!result.is_usable());
    assert!(result.max_kkt_residual_ratio > 1.0);
    assert!(result.restarts <= 9, "{result:?}");
    assert!(result.evaluations < 2000, "{result:?}");
    assert_ne!(result.price, arr1(&[1.0, 1.0]));
    assert!(result.price.iter().all(|p| (*p - 1.0).abs() < 1e-9));
    let verified = route.evaluate(&result.price);
    assert_eq!(verified.gradient, result.gradient);
    assert!(verified.cost <= route.evaluate(&arr1(&[1.0, 1.0])).cost);
}

#[test]
fn exhausted_warm_solve_returns_diagnostic_prices_without_acceptance() {
    let route: Route<_, UniswapV2> = Route {
        objective: StiffCoupled,
        markets: vec![],
        tokens: 2,
    };
    let mut options = SolveOptions::default();
    options.optimizer.max_evaluations = 10;
    options.optimizer.pgtol = 1e12;
    options.graph_precondition = false;
    let result =
        solve_price_with_options(route, arr1(&[1.0, 1.0]), options.with_warm_start_budget())
            .unwrap();
    assert_eq!(result.assessment, SolveAssessment::BudgetExhausted);
    assert_eq!(result.status, Status::MaxEvaluations);
    assert!(result.evaluations <= 10);
    assert!(!result.is_usable());
    assert!(result.max_kkt_residual_ratio > 1.0);
    assert!(result.price.iter().all(|p| p.is_finite() && *p > 0.0));
}

#[test]
fn invalid_optimizer_options_are_rejected_even_at_stationary_start() {
    let route: Route<_, UniswapV2> = Route {
        objective: NonnegativeLinear { c: arr1(&[1.0]) },
        markets: vec![],
        tokens: 1,
    };
    let mut options = SolveOptions::default();
    options.optimizer.memory = 0;
    assert!(solve_price_with_options(route, arr1(&[1.0]), options).is_err());
}

#[test]
fn objective_and_gradient_share_one_pool_evaluation() {
    struct Counted(Arc<AtomicUsize>);
    impl Market for Counted {
        fn arbitrage(&self, _: [f64; 2]) -> ([f64; 2], [f64; 2]) {
            self.0.fetch_add(1, Ordering::Relaxed);
            ([0.0; 2], [0.0; 2])
        }
    }
    let count = Arc::new(AtomicUsize::new(0));
    let route = Route {
        objective: NonnegativeLinear {
            c: arr1(&[1.0, 0.0]),
        },
        markets: vec![(Counted(count.clone()), (0, 1))],
        tokens: 2,
    };
    let evaluation = route.evaluate(&arr1(&[1.0, 1.0]));
    assert_eq!(count.load(Ordering::Relaxed), 1);
    assert_eq!(evaluation.cost, 0.0);
    assert_eq!(evaluation.gradient, arr1(&[0.0, 0.0]));
}

#[test]
fn accepted_result_passes_independent_residuals() {
    let result =
        solve_price_with_options(liquidation(), arr1(&[1.3, 0.4]), SolveOptions::default())
            .unwrap();
    assert!(result.is_usable(), "{result:?}");
    assert!(result.max_kkt_residual_ratio <= 1.0);
    assert!(result.max_deficit_ratio <= 1.0);
}
