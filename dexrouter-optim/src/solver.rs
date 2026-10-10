//! Routing-specific adaptation and acceptance; the optimizer stays task-agnostic.
use dexrouter_lbfgsb::{Bound, LineSearchFailure, Options, Status, minimize};
use ndarray::Array1;

use crate::{Route, RouteEvaluation, market::Market, utility::UtilityConjugate};

#[derive(Debug, Clone)]
pub struct SolveOptions {
    /// Generic optimizer options; applied to the normalized objective.
    pub optimizer: Options<f64>,
    /// Dimensionless positive guard relative to the initial price, not a utility constraint.
    pub relative_price_floor: f64,
    pub normalize_objective: bool,
    /// Enable diagonal curvature scaling and bounded coordinate recovery.
    pub precondition: bool,
    /// Reset curvature history and recompute the affine coordinate scales.
    pub restart_interval: usize,
    pub use_utility_anchor: bool,
    /// Acceptance uses per-token turnover, never the number of pools/tokens.
    pub relative_residual_tolerance: f64,
    /// Absolute residual value in the objective's chosen price unit.
    pub absolute_residual_value: f64,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            optimizer: Options {
                pgtol: 1e-10,
                max_iterations: 5000,
                max_evaluations: 50_000,
                ..Options::default()
            },
            relative_price_floor: 1e-12,
            normalize_objective: true,
            precondition: true,
            restart_interval: 500,
            use_utility_anchor: true,
            relative_residual_tolerance: 1e-8,
            absolute_residual_value: 1e-8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveAssessment {
    Converged,
    Approximate,
    NumericalStagnation,
    BudgetExhausted,
    Rejected,
}

#[derive(Debug, Clone)]
pub struct SolveDiagnostics {
    pub price: Array1<f64>,
    pub cost: f64,
    pub gradient: Array1<f64>,
    /// All joint and coordinate evaluations, including probes and final verification.
    pub evaluations: usize,
    pub iterations: usize,
    pub restarts: usize,
    /// Generic optimizer termination is preserved, independently of acceptance.
    pub status: Status,
    pub line_search_failure: Option<LineSearchFailure>,
    pub assessment: SolveAssessment,
    pub max_abs_gradient: f64,
    pub max_projected_gradient: f64,
    pub max_scaled_projected_gradient: f64,
    /// Worst per-token KKT residual divided by its configured allowance.
    pub max_kkt_residual_ratio: f64,
    pub max_deficit_value: f64,
    pub max_deficit_ratio: f64,
    pub numerical_lower_bounds: Array1<f64>,
    pub objective_scale: f64,
}

impl SolveDiagnostics {
    pub fn is_usable(&self) -> bool {
        matches!(
            self.assessment,
            SolveAssessment::Converged | SolveAssessment::Approximate
        )
    }
}

pub fn solve_price_with_init_p<U: UtilityConjugate, M: Market>(
    route: Route<U, M>,
    p: Array1<f64>,
) -> Array1<f64> {
    let result = solve_price_with_init_p_checked(route, p).expect("price solve failed");
    assert!(
        result.is_usable(),
        "unacceptable price solution: {:?} (KKT ratio {}, deficit ratio {})",
        result.assessment,
        result.max_kkt_residual_ratio,
        result.max_deficit_ratio
    );
    result.price
}

/// Returns finite diagnostics even on rejection; callers must inspect is_usable.
pub fn solve_price_with_init_p_checked<U: UtilityConjugate, M: Market>(
    route: Route<U, M>,
    p: Array1<f64>,
) -> anyhow::Result<SolveDiagnostics> {
    solve_price_with_options(route, p, SolveOptions::default())
}

pub fn solve_price_with_options<U: UtilityConjugate, M: Market>(
    route: Route<U, M>,
    initial: Array1<f64>,
    options: SolveOptions,
) -> anyhow::Result<SolveDiagnostics> {
    options.optimizer.validate()?;
    anyhow::ensure!(
        initial.len() == route.tokens && route.tokens > 0,
        "initial price length must match nonzero token count"
    );
    anyhow::ensure!(
        initial.iter().all(|v| v.is_finite() && *v > 0.0),
        "initial price must be finite and positive"
    );
    anyhow::ensure!(
        options.relative_price_floor.is_finite()
            && options.relative_price_floor > 0.0
            && options.relative_price_floor < 1.0,
        "relative price floor must be in (0, 1)"
    );
    anyhow::ensure!(
        options.relative_residual_tolerance.is_finite()
            && options.relative_residual_tolerance >= 0.0
            && options.absolute_residual_value.is_finite()
            && options.absolute_residual_value > 0.0,
        "invalid residual tolerances"
    );
    anyhow::ensure!(
        options.restart_interval > 0,
        "restart interval must be positive"
    );
    let math_lower = route.objective.lower_bounds();
    let math_upper = route.objective.upper_bounds();
    anyhow::ensure!(
        math_lower.len() == route.tokens && math_upper.len() == route.tokens,
        "bound lengths must match token count"
    );
    anyhow::ensure!(
        route.objective.grad(&initial).len() == route.tokens,
        "utility gradient length must match token count"
    );
    anyhow::ensure!(
        route
            .markets
            .iter()
            .all(|(_, (a, b))| *a < route.tokens && *b < route.tokens),
        "market token index out of range"
    );
    for (&l, &u) in math_lower.iter().zip(&math_upper) {
        anyhow::ensure!(
            !l.is_nan() && !u.is_nan() && l <= u && l != f64::INFINITY && u != f64::NEG_INFINITY,
            "invalid utility bounds"
        );
    }
    let mut lower = Array1::from_iter(
        math_lower
            .iter()
            .zip(&initial)
            .map(|(&l, &p)| l.max((p * options.relative_price_floor).max(f64::MIN_POSITIVE))),
    );
    let mut upper = math_upper.clone();
    if options.use_utility_anchor {
        if let Some((index, price)) = route.objective.price_anchor() {
            anyhow::ensure!(
                index < route.tokens
                    && price.is_finite()
                    && price > 0.0
                    && price >= math_lower[index]
                    && price <= math_upper[index],
                "invalid utility anchor"
            );
            lower[index] = price;
            upper[index] = price;
        }
    }
    anyhow::ensure!(
        lower.iter().zip(&upper).all(|(&l, &u)| l <= u),
        "numerical price guard conflicts with utility upper bound"
    );
    let mut price = Array1::from_iter(
        initial
            .iter()
            .zip(&lower)
            .zip(&upper)
            .map(|((&p, &l), &u)| p.max(l).min(u)),
    );
    let mut evaluations = 0;
    let mut evaluate = |p: &Array1<f64>| -> anyhow::Result<RouteEvaluation> {
        evaluations += 1;
        let evaluation = route.evaluate(p);
        anyhow::ensure!(
            evaluation.cost.is_finite()
                && evaluation.gradient.iter().all(|v| v.is_finite())
                && evaluation.turnover.iter().all(|v| v.is_finite()),
            "non-finite route evaluation"
        );
        Ok(evaluation)
    };
    let first = evaluate(&price)?;
    let objective_scale = if options.normalize_objective {
        first.cost.abs().max(1.0)
    } else {
        1.0
    };
    let mut iterations = 0;
    let mut restarts = 0;
    let mut status = Status::ConvergedProjectedGradient;
    let mut line_search_failure = None;
    let mut scale = initial.clone();
    let mut pgtol = options.optimizer.pgtol;
    // Probe/solver/verification costs share a single hard evaluation budget.
    drop(evaluate);
    let mut current = first;
    loop {
        let ratios = residuals(&price, &current, &math_lower, &math_upper, &options);
        if ratios.0 <= 1.0 && ratios.2 <= 1.0 {
            break;
        }
        if iterations >= options.optimizer.max_iterations {
            status = Status::MaxIterations;
            break;
        }
        if evaluations + 1 >= options.optimizer.max_evaluations {
            status = Status::MaxEvaluations;
            break;
        }
        if options.precondition
            && evaluations + 2 * route.tokens + 2 < options.optimizer.max_evaluations
        {
            for i in 0..route.tokens {
                if lower[i] == upper[i] {
                    continue;
                }
                let h = price[i] * 1e-4;
                let mut plus = price.clone();
                let mut minus = price.clone();
                plus[i] = (price[i] + h).min(upper[i]);
                minus[i] = (price[i] - h).max(lower[i]);
                if plus[i] <= minus[i] {
                    continue;
                }
                let gp = route.evaluate(&plus).gradient[i];
                let gm = route.evaluate(&minus).gradient[i];
                evaluations += 2;
                let curvature = (gp - gm) / (plus[i] - minus[i]);
                if curvature.is_finite() && curvature > 0.0 {
                    scale[i] = (objective_scale / curvature)
                        .sqrt()
                        .clamp(initial[i] * 1e-6, initial[i] * 1e6);
                }
            }
        }
        let bounds: Vec<_> = lower
            .iter()
            .zip(&upper)
            .zip(&scale)
            .zip(&price)
            .map(|(((&l, &u), &s), &p)| {
                Bound::new(Some((l - p) / s), u.is_finite().then_some((u - p) / s))
            })
            .collect();
        let mut optimizer = options.optimizer.clone();
        optimizer.max_iterations =
            (options.optimizer.max_iterations - iterations).min(options.restart_interval);
        optimizer.max_evaluations = options.optimizer.max_evaluations - evaluations - 1;
        optimizer.pgtol = pgtol;
        // Start at zero around the current physical price. Multiplicative
        // p / scale -> x * scale round trips can move a price by one ulp.
        let result = minimize(vec![0.0; route.tokens], bounds, optimizer, |x, g| {
            let p = physical_price(x, &price, &scale, &lower, &upper);
            let (evaluation, cost_change) = route.evaluate_centered(&p, Some((&price, &current)));
            let gradient = &evaluation.gradient * &scale / objective_scale;
            g.copy_from_slice(gradient.as_slice().unwrap());
            Ok(cost_change / objective_scale)
        })?;
        evaluations += result.evaluations;
        iterations += result.iterations;
        let next_price = physical_price(&result.x, &price, &scale, &lower, &upper);
        let unchanged = next_price == price;
        price = next_price;
        status = result.status;
        line_search_failure = result.line_search_failure;
        current = route.evaluate(&price);
        evaluations += 1;
        let ratios = residuals(&price, &current, &math_lower, &math_upper, &options);
        if ratios.0 <= 1.0 && ratios.2 <= 1.0 {
            break;
        }
        if matches!(status, Status::LineSearchFailed | Status::MaxIterations)
            && options.precondition
            && iterations < options.optimizer.max_iterations
            && evaluations + 2 < options.optimizer.max_evaluations
        {
            let before = price.clone();
            coordinate_sweep(
                &route,
                &mut price,
                &lower,
                &upper,
                &options,
                &mut evaluations,
            );
            let (recovered, change) = route.evaluate_centered(&price, Some((&before, &current)));
            if change.is_finite() && change <= 0.0 {
                current = recovered;
            } else {
                price = before.clone();
            }
            evaluations += 1;
            iterations += 1;
            let ratios = residuals(&price, &current, &math_lower, &math_upper, &options);
            if ratios.0 <= 1.0 && ratios.2 <= 1.0 {
                break;
            }
            if evaluations >= options.optimizer.max_evaluations {
                status = Status::MaxEvaluations;
                break;
            }
            if price != before {
                restarts += 1;
                continue;
            }
        }
        if status == Status::MaxEvaluations
            || (status == Status::LineSearchFailed && result.iterations == 0)
            || (unchanged && status != Status::ConvergedProjectedGradient)
            || (result.iterations == 0 && restarts >= 8)
        {
            break;
        }
        if matches!(
            status,
            Status::ConvergedProjectedGradient | Status::ConvergedFunctionReduction
        ) {
            // A generic convergence test is not routing acceptance. Retry with
            // fresh curvature and a tighter generic tolerance, within budget.
            pgtol *= 0.1;
        }
        restarts += 1;
    }
    anyhow::ensure!(
        price.iter().all(|v| v.is_finite() && *v > 0.0)
            && current.cost.is_finite()
            && current.gradient.iter().all(|v| v.is_finite()),
        "non-finite final solver result"
    );
    let (kkt_ratio, deficit_value, deficit_ratio) =
        residuals(&price, &current, &math_lower, &math_upper, &options);
    let accepted = kkt_ratio <= 1.0 && deficit_ratio <= 1.0;
    let assessment = if accepted {
        if status == Status::ConvergedProjectedGradient {
            SolveAssessment::Converged
        } else {
            SolveAssessment::Approximate
        }
    } else if matches!(status, Status::MaxIterations | Status::MaxEvaluations) {
        SolveAssessment::BudgetExhausted
    } else if matches!(
        status,
        Status::LineSearchFailed | Status::ConvergedFunctionReduction
    ) {
        SolveAssessment::NumericalStagnation
    } else {
        SolveAssessment::Rejected
    };
    let max_abs_gradient = current.gradient.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let projected: Vec<_> = price
        .iter()
        .zip(&current.gradient)
        .zip(math_lower.iter().zip(&math_upper))
        .map(|((&p, &g), (&l, &u))| kkt_gradient(p, g, l, u))
        .collect();
    Ok(SolveDiagnostics {
        max_projected_gradient: projected.iter().map(|g| g.abs()).fold(0.0, f64::max),
        max_scaled_projected_gradient: projected
            .iter()
            .zip(&initial)
            .map(|(&g, &s)| (g * s).abs())
            .fold(0.0, f64::max),
        price,
        cost: current.cost,
        gradient: current.gradient,
        evaluations,
        iterations,
        restarts,
        status,
        line_search_failure,
        assessment,
        max_abs_gradient,
        max_kkt_residual_ratio: kkt_ratio,
        max_deficit_value: deficit_value,
        max_deficit_ratio: deficit_ratio,
        numerical_lower_bounds: lower,
        objective_scale,
    })
}

// Recover narrow fee/tick regions by solving bounded convex coordinates.
// Probes touch only incident pools and share the global evaluation budget.
fn coordinate_sweep<U: UtilityConjugate, M: Market>(
    route: &Route<U, M>,
    price: &mut Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    options: &SolveOptions,
    evaluations: &mut usize,
) {
    for i in 0..route.tokens {
        if lower[i] == upper[i] || *evaluations + 2 >= options.optimizer.max_evaluations {
            continue;
        }
        let original = price[i];
        // Reserve one full evaluation for verification by the caller.
        let budget = (options.optimizer.max_evaluations - *evaluations - 1).min(128);
        let mut probe = |x: f64| {
            price[i] = x;
            *evaluations += 1;
            let mut gradient = crate::numerics::Sum::default();
            gradient.add(route.objective.grad(price)[i]);
            let mut turnover = crate::numerics::Sum::default();
            for (market, (a, b)) in &route.markets {
                if *a != i && *b != i {
                    continue;
                }
                let trade = market.evaluate([price[*a], price[*b]]);
                for (local, token) in [*a, *b].into_iter().enumerate() {
                    if token == i {
                        gradient.add(trade.output[local] - trade.input[local]);
                        turnover.add(trade.output[local] + trade.input[local]);
                    }
                }
            }
            let g = gradient.total();
            let allowance = options.absolute_residual_value
                + options.relative_residual_tolerance * turnover.total() * x;
            (g, g.abs() * x / allowance)
        };
        let (g, ratio) = probe(original);
        if ratio <= 0.25 || (original == lower[i] && g >= 0.0) || (original == upper[i] && g <= 0.0)
        {
            continue;
        }
        let mut lo = original;
        let mut hi = original;
        let mut best = original;
        let mut best_ratio = ratio;
        let mut bracketed = false;
        let mut last_x = original;
        for _ in 1..budget {
            let x = if bracketed {
                lo + (hi - lo) * 0.5
            } else if g > 0.0 {
                (last_x * 0.5).max(lower[i])
            } else {
                (last_x * 2.0).min(upper[i]).min(f64::MAX)
            };
            if x == last_x || !x.is_finite() {
                break;
            }
            let (gx, rx) = probe(x);
            last_x = x;
            if !gx.is_finite() || !rx.is_finite() {
                break;
            }
            if rx < best_ratio {
                best = x;
                best_ratio = rx;
            }
            if rx <= 0.25 || (x == lower[i] && gx >= 0.0) || (x == upper[i] && gx <= 0.0) {
                best = x;
                break;
            }
            if (g > 0.0 && gx <= 0.0) || (g < 0.0 && gx >= 0.0) {
                bracketed = true;
            }
            if gx < 0.0 {
                lo = x;
            } else {
                hi = x;
            }
        }
        drop(probe);
        price[i] = best;
    }
}

fn physical_price(
    x: &[f64],
    origin: &Array1<f64>,
    scale: &Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
) -> Array1<f64> {
    Array1::from_iter(
        x.iter()
            .zip(scale)
            .zip(origin)
            .zip(lower.iter().zip(upper))
            .map(|(((&x, &s), &p), (&l, &u))| x.mul_add(s, p).max(l).min(u)),
    )
}

fn kkt_gradient(price: f64, gradient: f64, lower: f64, upper: f64) -> f64 {
    if (price <= lower && gradient >= 0.0) || (price >= upper && gradient <= 0.0) {
        0.0
    } else {
        gradient
    }
}

fn residuals(
    price: &Array1<f64>,
    evaluation: &RouteEvaluation,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    options: &SolveOptions,
) -> (f64, f64, f64) {
    let mut kkt_ratio = 0.0_f64;
    let mut deficit = 0.0_f64;
    let mut deficit_ratio = 0.0_f64;
    for i in 0..price.len() {
        let allowance = options.absolute_residual_value
            + options.relative_residual_tolerance * evaluation.turnover[i] * price[i];
        let g = evaluation.gradient[i];
        kkt_ratio = kkt_ratio
            .max(kkt_gradient(price[i], g, lower[i], upper[i]).abs() * price[i] / allowance);
        // A finite utility upper bound permits a negative derivative there.
        let negative_value = (-kkt_gradient(price[i], g, lower[i], upper[i]) * price[i]).max(0.0);
        deficit = deficit.max(negative_value);
        deficit_ratio = deficit_ratio.max(negative_value / allowance);
    }
    (kkt_ratio, deficit, deficit_ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{market::UniswapV2, utility::NonnegativeLinear};
    use ndarray::arr1;

    #[test]
    fn coordinate_recovery_finds_root_or_active_bound_from_either_side() {
        let route = Route {
            objective: NonnegativeLinear {
                c: arr1(&[0.0, 1.0]),
            },
            markets: vec![(UniswapV2::new(2.0, 5.0, 1.0), (0, 1))],
            tokens: 2,
        };
        for (initial, lower0, upper0, expected) in [
            (10.0, 0.01, f64::INFINITY, 2.5),
            (0.1, 0.01, f64::INFINITY, 2.5),
            (10.0, 3.0, f64::INFINITY, 3.0),
            (0.1, 0.01, 1.5, 1.5),
        ] {
            let mut price = arr1(&[initial, 1.0]);
            let mut evaluations = 0;
            let options = SolveOptions::default();
            let before = route.utility(&price);
            coordinate_sweep(
                &route,
                &mut price,
                &arr1(&[lower0, 1.0]),
                &arr1(&[upper0, 1.0]),
                &options,
                &mut evaluations,
            );
            assert!((price[0] - expected).abs() < 1e-7, "{price:?}");
            assert_eq!(price[1], 1.0);
            assert!(route.utility(&price) <= before);
            assert!(evaluations > 0 && evaluations < 128);
        }
    }

    #[test]
    fn coordinate_recovery_reserves_final_verification_budget() {
        let route = Route {
            objective: NonnegativeLinear {
                c: arr1(&[0.0, 1.0]),
            },
            markets: vec![(UniswapV2::new(2.0, 5.0, 1.0), (0, 1))],
            tokens: 2,
        };
        for budget in 1..12 {
            let mut price = arr1(&[10.0, 1.0]);
            let mut evaluations = 0;
            let mut options = SolveOptions::default();
            options.optimizer.max_evaluations = budget;
            coordinate_sweep(
                &route,
                &mut price,
                &arr1(&[0.01, 1.0]),
                &arr1(&[f64::INFINITY, 1.0]),
                &options,
                &mut evaluations,
            );
            assert!(evaluations < budget);
            assert!(price[0] >= 0.01 && price[0].is_finite());
            assert_eq!(price[1], 1.0);
        }
    }
}

pub fn solve_price<U: UtilityConjugate, M: Market>(route: Route<U, M>) -> Array1<f64> {
    let tokens = route.tokens;
    solve_price_with_init_p(route, Array1::from_elem(tokens, 1.0))
}
