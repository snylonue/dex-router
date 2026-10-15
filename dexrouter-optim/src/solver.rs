//! Routing-specific adaptation and acceptance; the optimizer stays task-agnostic.
use dexrouter_lbfgsb::{
    Bound, LineSearchFailure, Options, Status, minimize, minimize_with_differences,
};
use ndarray::Array1;

use crate::{Route, RouteEvaluation, market::Market, utility::UtilityConjugate};

#[path = "precise_solver.rs"]
mod precise;
pub use precise::{PreciseSolveDiagnostics, PreciseStatus, solve_price_with_precise_newton};

#[derive(Debug, Clone)]
pub struct SolveOptions {
    /// Generic optimizer options; applied to the normalized objective.
    pub optimizer: Options<f64>,
    /// Dimensionless positive guard relative to the initial price, not a utility constraint.
    pub relative_price_floor: f64,
    pub normalize_objective: bool,
    /// Recenter every line search using stable per-market value differences.
    pub stable_differences: bool,
    /// Exact convex square-root-price reformulation. Requires explicit promises
    /// from every market and the utility (built-in V2 and nonnegative baskets).
    /// Unsupported models return an error, never silently become nonconvex.
    pub sqrt_prices: bool,
    /// Enable diagonal curvature scaling and bounded coordinate recovery.
    pub precondition: bool,
    /// Reset curvature history and recompute the affine coordinate scales.
    pub restart_interval: usize,
    pub use_utility_anchor: bool,
    /// Split only when the utility explicitly promises separability.
    pub decompose: bool,
    /// Prefer exact regional pool curvature; custom implementations fall back to probes.
    pub analytic_curvature: bool,
    /// Try graph-Newton recovery before coordinate recovery.
    /// Damping affects the step matrix, never the economic objective.
    pub graph_precondition: bool,
    /// Continue successful sparse Newton recovery instead of rebuilding L-BFGS history.
    pub newton_polish: bool,
    /// Use a curvature spanning-forest inverse metric for the standalone Newton PCG.
    /// It changes neither prices/bounds nor the objective; false keeps Jacobi.
    pub newton_tree_precondition: bool,
    /// Exact degree-two graph elimination for standalone/warm f64 Newton.
    /// The separate hi+lo refinement always attempts this metric.
    /// Uses a bounded equilibrated dense core.
    /// A failed/oversized factorization falls back to the configured metric.
    pub newton_schur_precondition: bool,
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
            stable_differences: true,
            sqrt_prices: false,
            precondition: true,
            restart_interval: 500,
            use_utility_anchor: true,
            decompose: true,
            analytic_curvature: true,
            graph_precondition: true,
            newton_polish: true,
            newton_tree_precondition: false,
            newton_schur_precondition: false,
            relative_residual_tolerance: 1e-8,
            absolute_residual_value: 1e-8,
        }
    }
}

impl SolveOptions {
    /// Bound a diagnostic warm solve before handing its prices to a stricter
    /// downstream solver (for example DD). Preserve all economic tolerances,
    /// bounds and optimizer settings, and never enlarge a caller's budget.
    /// Budget exhaustion or stagnation is not economic acceptance: callers may
    /// reuse the returned prices, but must independently verify the final solve.
    pub fn with_warm_start_budget(mut self) -> Self {
        self.optimizer.max_iterations = self.optimizer.max_iterations.min(2000);
        self.optimizer.max_evaluations = self.optimizer.max_evaluations.min(2000);
        self
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
pub struct SolveDiagnostics<S = Status> {
    pub price: Array1<f64>,
    pub cost: f64,
    pub gradient: Array1<f64>,
    /// All joint and coordinate evaluations, including probes and final verification.
    pub evaluations: usize,
    pub iterations: usize,
    pub restarts: usize,
    /// Generic optimizer termination is preserved, independently of acceptance.
    pub status: S,
    /// Newton backend work only, separate from objective/gradient evaluations.
    pub hessian_products: usize,
    pub hessian_evaluations: usize,
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
    /// Fixed prices are absent from every optimizer vector.
    pub optimizer_variables: usize,
    pub optimization_blocks: usize,
}

impl<S> SolveDiagnostics<S> {
    fn map_status<T>(self, map: impl FnOnce(S) -> T) -> SolveDiagnostics<T> {
        SolveDiagnostics {
            price: self.price,
            cost: self.cost,
            gradient: self.gradient,
            evaluations: self.evaluations,
            iterations: self.iterations,
            restarts: self.restarts,
            status: map(self.status),
            line_search_failure: self.line_search_failure,
            hessian_products: self.hessian_products,
            hessian_evaluations: self.hessian_evaluations,
            assessment: self.assessment,
            max_abs_gradient: self.max_abs_gradient,
            max_projected_gradient: self.max_projected_gradient,
            max_scaled_projected_gradient: self.max_scaled_projected_gradient,
            max_kkt_residual_ratio: self.max_kkt_residual_ratio,
            max_deficit_value: self.max_deficit_value,
            max_deficit_ratio: self.max_deficit_ratio,
            numerical_lower_bounds: self.numerical_lower_bounds,
            objective_scale: self.objective_scale,
            optimizer_variables: self.optimizer_variables,
            optimization_blocks: self.optimization_blocks,
        }
    }

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
    Ok(
        solve_impl(&route, initial, options, None)?.map_status(|s| match s {
            BackendStatus::Lbfgsb(s) => s,
            _ => unreachable!(),
        }),
    )
}

/// Explicit, opt-in matrix-free Newton backend. Status is the Newton termination,
/// never relabeled as an L-BFGS-B task. Shares preprocessing and original economic
/// acceptance, but does not run the legacy graph/coordinate recovery after Newton.
/// Requires analytic curvature and affine prices. Both supplied FG/iteration
/// budgets are honored by taking their minimum; HVP has its own global budget.
pub fn solve_price_with_newton<U: UtilityConjugate, M: Market>(
    route: Route<U, M>,
    initial: Array1<f64>,
    options: SolveOptions,
    newton: dexrouter_newton::Options,
) -> anyhow::Result<SolveDiagnostics<dexrouter_newton::Status>> {
    Ok(
        solve_impl(&route, initial, options, Some(newton))?.map_status(|s| match s {
            BackendStatus::Newton(s) => s,
            _ => unreachable!(),
        }),
    )
}

struct PhaseResult {
    x: Vec<f64>,
    iterations: usize,
    evaluations: usize,
    status: Status,
    line_search_failure: Option<LineSearchFailure>,
}
impl From<dexrouter_lbfgsb::OptimizeResult<f64>> for PhaseResult {
    fn from(r: dexrouter_lbfgsb::OptimizeResult<f64>) -> Self {
        Self {
            x: r.x,
            iterations: r.iterations,
            evaluations: r.evaluations,
            status: r.status,
            line_search_failure: r.line_search_failure,
        }
    }
}

enum BackendStatus {
    Lbfgsb(Status),
    Newton(dexrouter_newton::Status),
}

// Only internal driver categories are shared. Public diagnostics retain the
// native backend status, including HVP exhaustion and numerical stagnation.
fn newton_control_status(s: dexrouter_newton::Status) -> Status {
    use dexrouter_newton::Status as N;
    match s {
        N::ConvergedProjectedGradient => Status::ConvergedProjectedGradient,
        N::MaxIterations => Status::MaxIterations,
        N::MaxEvaluations | N::MaxHessianProducts => Status::MaxEvaluations,
        N::LineSearchFailed | N::NumericalStagnation => Status::LineSearchFailed,
    }
}

fn solve_impl<U: UtilityConjugate, M: Market>(
    route: &Route<U, M>,
    initial: Array1<f64>,
    mut options: SolveOptions,
    newton_options: Option<dexrouter_newton::Options>,
) -> anyhow::Result<SolveDiagnostics<BackendStatus>> {
    options.optimizer.validate()?;
    if let Some(n) = &newton_options {
        n.validate()?;
        anyhow::ensure!(
            !options.sqrt_prices && options.analytic_curvature,
            "Newton requires affine prices and analytic curvature"
        );
        options.optimizer.max_evaluations =
            options.optimizer.max_evaluations.min(n.max_evaluations);
        options.optimizer.max_iterations = options.optimizer.max_iterations.min(n.max_iterations);
    }
    anyhow::ensure!(
        !options.sqrt_prices
            || (route.objective.convex_in_sqrt_prices()
                && route.markets.iter().all(|(m, _)| m.convex_in_sqrt_prices())),
        "square-root price coordinates require convexity promises from utility and markets"
    );
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
    let blocks = optimization_blocks(route, &lower, &upper, options.decompose);
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
    let mut pgtol = newton_options
        .as_ref()
        .map_or(options.optimizer.pgtol, |n| n.pgtol);
    let mut newton_status = None;
    let mut hessian_products = 0;
    let mut hessian_evaluations = 0;
    let mut progress = ResidualProgress::new(
        (0..route.tokens)
            .map(|i| token_residual_ratio(i, &price, &first, &math_lower, &math_upper, &options)),
    );
    let mut numerical_stagnation = false;
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
        if newton_options
            .as_ref()
            .is_some_and(|n| hessian_products >= n.max_hessian_products)
        {
            status = Status::MaxEvaluations;
            newton_status = Some(dexrouter_newton::Status::MaxHessianProducts);
            break;
        }
        // No extra anchors are introduced. Eliminated-price residuals still
        // participate in the independent, original-problem acceptance test.
        let Some(block) = blocks.iter().max_by(|a, b| {
            block_residual(a, &price, &current, &math_lower, &math_upper, &options).total_cmp(
                &block_residual(b, &price, &current, &math_lower, &math_upper, &options),
            )
        }) else {
            break;
        };
        if block_residual(block, &price, &current, &math_lower, &math_upper, &options) <= 1.0 {
            break;
        }
        let curvature = if options.precondition && options.analytic_curvature {
            relative_curvature(route, &price)
        } else {
            None
        };
        if options.precondition {
            if let Some(model) = &curvature {
                for &i in &block.tokens {
                    let diagonal = model.diagonal[i];
                    if diagonal > 0.0 {
                        let candidate = price[i] * (objective_scale / diagonal).sqrt();
                        if candidate.is_finite() && candidate > 0.0 {
                            scale[i] = candidate.clamp(initial[i] * 1e-6, initial[i] * 1e6);
                        }
                    }
                }
            } else {
                for &i in &block.tokens {
                    // Keep one verification and at least one optimizer evaluation.
                    if evaluations + 4 >= options.optimizer.max_evaluations {
                        break;
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
                    let diagonal = (gp - gm) / (plus[i] - minus[i]);
                    if diagonal.is_finite() && diagonal > 0.0 {
                        scale[i] = (objective_scale / diagonal)
                            .sqrt()
                            .clamp(initial[i] * 1e-6, initial[i] * 1e6);
                    }
                }
            }
        }
        let bounds: Vec<_> = block
            .tokens
            .iter()
            .map(|&i| {
                Bound::new(
                    Some(coordinate_bound(
                        lower[i],
                        price[i],
                        scale[i],
                        options.sqrt_prices,
                    )),
                    upper[i].is_finite().then(|| {
                        coordinate_bound(upper[i], price[i], scale[i], options.sqrt_prices)
                    }),
                )
            })
            .collect();
        let mut optimizer = options.optimizer.clone();
        optimizer.max_iterations =
            (options.optimizer.max_iterations - iterations).min(options.restart_interval);
        optimizer.max_evaluations = options.optimizer.max_evaluations - evaluations - 1;
        optimizer.pgtol = pgtol;
        // Start at zero around the current physical price. Multiplicative
        // p / scale -> x * scale round trips can move a price by one ulp.
        let zero = vec![0.0; block.tokens.len()];
        let original_trades: Vec<_> = block
            .markets
            .iter()
            .map(|&i| current.market_evaluations[i])
            .collect();
        let mut reference_x = zero.clone();
        let mut reference_price = price.clone();
        let mut reference_trades = original_trades.clone();
        let mut last_x = zero.clone();
        let mut last_price = price.clone();
        let mut last_trades = original_trades.clone();
        let mut previous_trial = (zero.clone(), price.clone(), original_trades.clone());
        let mut fg = |x: &[f64], g: &mut [f64], reference: Option<&[f64]>| {
            if let Some(reference) = reference
                && reference != reference_x
            {
                // Expansion may reject the trial after the accepted one. Keep
                // both quote caches: the search reference is the accepted point,
                // not necessarily the last FG. No uncounted quote is needed.
                if reference == last_x {
                    reference_x.clone_from(&last_x);
                    reference_price.clone_from(&last_price);
                    reference_trades.clone_from(&last_trades);
                } else {
                    debug_assert_eq!(reference, previous_trial.0);
                    reference_x.clone_from(&previous_trial.0);
                    reference_price.clone_from(&previous_trial.1);
                    reference_trades.clone_from(&previous_trial.2);
                }
            }
            let p = physical_price(
                x,
                &block.tokens,
                &price,
                &scale,
                &lower,
                &upper,
                options.sqrt_prices,
            );
            let (gradient, cost_change, trades) = evaluate_block(
                route,
                block,
                &p,
                &price,
                &original_trades,
                Some((&last_price, &last_trades)),
                !options.stable_differences,
            );
            for (local, &i) in block.tokens.iter().enumerate() {
                let jacobian = if options.sqrt_prices {
                    scale[i] * (1.0 + 0.5 * x[local] * (scale[i] / price[i]))
                } else {
                    scale[i]
                };
                g[local] = gradient[local] * jacobian / objective_scale;
            }
            let difference = if reference.is_some() {
                block_value_change(
                    route,
                    block,
                    &p,
                    &reference_price,
                    &trades,
                    &reference_trades,
                )
            } else {
                0.0
            };
            previous_trial = (
                std::mem::replace(&mut last_x, x.to_vec()),
                std::mem::replace(&mut last_price, p),
                std::mem::replace(&mut last_trades, trades),
            );
            Ok((cost_change / objective_scale, difference / objective_scale))
        };
        let result = if let Some(n) = &newton_options {
            let mut n = n.clone();
            n.max_evaluations = optimizer.max_evaluations;
            n.max_iterations = optimizer.max_iterations;
            n.max_hessian_products -= hessian_products;
            n.pgtol = pgtol;
            let newton_bounds = bounds
                .iter()
                .map(|b| {
                    dexrouter_newton::Bound::new(
                        b.lower.unwrap_or(f64::NEG_INFINITY),
                        b.upper.unwrap_or(f64::INFINITY),
                    )
                })
                .collect();
            let solved = dexrouter_newton::minimize(
                zero,
                newton_bounds,
                n,
                |x, g, reference| {
                    let (value, change) = fg(x, g, reference)?;
                    Ok(if options.stable_differences {
                        dexrouter_newton::Evaluation::with_difference(value, change)
                    } else {
                        dexrouter_newton::Evaluation::value(value)
                    })
                },
                |x| {
                    let p = physical_price(x, &block.tokens, &price, &scale, &lower, &upper, false);
                    let curvature =
                        relative_curvature_for_markets(route, &p, block.markets.iter().copied())
                            .ok_or_else(|| {
                                anyhow::anyhow!("analytic route curvature unavailable")
                            })?;
                    Ok(BlockHessian {
                        factors: block.tokens.iter().map(|&i| scale[i] / p[i]).collect(),
                        tokens: block.tokens.clone(),
                        curvature,
                        objective_scale,
                        tree_precondition: options.newton_tree_precondition,
                        schur_precondition: options.newton_schur_precondition,
                    })
                },
            )?;
            hessian_products += solved.hessian_products;
            hessian_evaluations += solved.hessian_evaluations;
            newton_status = Some(solved.status);
            // Internal phase record only; the public status below is native.
            Ok(PhaseResult {
                x: solved.x,
                iterations: solved.iterations,
                evaluations: solved.evaluations,
                status: newton_control_status(solved.status),
                line_search_failure: None,
            })
        } else if options.stable_differences {
            minimize_with_differences(zero, bounds, optimizer, &mut fg).map(PhaseResult::from)
        } else {
            minimize(zero, bounds, optimizer, |x, g| fg(x, g, None).map(|v| v.0))
                .map(PhaseResult::from)
        }?;
        evaluations += result.evaluations;
        iterations += result.iterations;
        let before_round = price.clone();
        price = physical_price(
            &result.x,
            &block.tokens,
            &price,
            &scale,
            &lower,
            &upper,
            options.sqrt_prices,
        );
        status = result.status;
        line_search_failure = result.line_search_failure;
        current = route.evaluate(&price);
        evaluations += 1;
        let ratios = residuals(&price, &current, &math_lower, &math_upper, &options);
        if ratios.0 <= 1.0 && ratios.2 <= 1.0 {
            break;
        }
        if let Some(newton) = &newton_options
            && status == Status::LineSearchFailed
            && iterations < options.optimizer.max_iterations
            && evaluations + 2 <= options.optimizer.max_evaluations
            && let Some(repaired) = rounding_repair(
                route,
                block,
                &price,
                &current,
                &lower,
                &upper,
                &math_lower,
                &math_upper,
                &options,
                newton.armijo,
                &mut evaluations,
                &mut hessian_evaluations,
            )
        {
            price = repaired;
            current = route.evaluate(&price);
            evaluations += 1;
            iterations += 1;
            let ratios = residuals(&price, &current, &math_lower, &math_upper, &options);
            if ratios.0 <= 1.0 && ratios.2 <= 1.0 {
                break;
            }
        }
        if newton_options.is_none()
            && matches!(
                status,
                Status::LineSearchFailed
                    | Status::MaxIterations
                    | Status::ConvergedProjectedGradient
                    | Status::ConvergedFunctionReduction
            )
            && block_residual(block, &price, &current, &math_lower, &math_upper, &options) > 1.0
            && options.precondition
            && iterations < options.optimizer.max_iterations
            && evaluations + 2 < options.optimizer.max_evaluations
        {
            let before = price.clone();
            let graph_recovered = options.graph_precondition
                && options.analytic_curvature
                && graph_recovery(
                    route,
                    block,
                    &mut price,
                    &current,
                    &lower,
                    &upper,
                    &options,
                    &mut evaluations,
                );
            if !graph_recovered {
                coordinate_sweep(
                    route,
                    &mut price,
                    &lower,
                    &upper,
                    &options,
                    &mut evaluations,
                );
            }
            let (recovered, change) = route.evaluate_centered(&price, Some((&before, &current)));
            if change.is_finite() && change <= 0.0 {
                current = recovered;
            } else {
                price = before.clone();
            }
            evaluations += 1;
            iterations += 1;
            // Once the active fee/tick region is useful, solve the reduced KKT
            // equations directly. Every accepted step still decreases the true
            // dual and contracts the block's independently scaled KKT residual.
            while graph_recovered
                && options.newton_polish
                && iterations < options.optimizer.max_iterations
                && evaluations + 2 < options.optimizer.max_evaluations
                && block_residual(block, &price, &current, &math_lower, &math_upper, &options) > 1.0
            {
                let before = price.clone();
                if !graph_recovery(
                    route,
                    block,
                    &mut price,
                    &current,
                    &lower,
                    &upper,
                    &options,
                    &mut evaluations,
                ) {
                    break;
                }
                let (next, change) = route.evaluate_centered(&price, Some((&before, &current)));
                evaluations += 1;
                iterations += 1;
                if !change.is_finite() || change > 0.0 {
                    price = before;
                    break;
                }
                current = next;
            }
            let ratios = residuals(&price, &current, &math_lower, &math_upper, &options);
            if ratios.0 <= 1.0 && ratios.2 <= 1.0 {
                break;
            }
            if evaluations >= options.optimizer.max_evaluations {
                status = Status::MaxEvaluations;
                break;
            }
        }
        // Assess the complete optimizer + recovery round. Neither a lower
        // objective nor a changed price/positive iteration count proves that
        // an economically violating token made meaningful progress.
        if progress.stagnated(
            (0..route.tokens).map(|i| {
                token_residual_ratio(i, &price, &current, &math_lower, &math_upper, &options)
            }),
        ) {
            numerical_stagnation = true;
            break;
        }
        if status == Status::MaxEvaluations
            || (status == Status::LineSearchFailed
                && result.iterations == 0
                && price == before_round)
            || (price == before_round && status != Status::ConvergedProjectedGradient)
        {
            break;
        }
        if matches!(
            status,
            Status::ConvergedProjectedGradient | Status::ConvergedFunctionReduction
        ) && block_residual(block, &price, &current, &math_lower, &math_upper, &options) > 1.0
        {
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
    } else if numerical_stagnation {
        SolveAssessment::NumericalStagnation
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
        status: if newton_options.is_some() {
            let native = newton_status
                .filter(|&s| newton_control_status(s) == status)
                .unwrap_or(match status {
                    Status::ConvergedProjectedGradient => {
                        dexrouter_newton::Status::ConvergedProjectedGradient
                    }
                    Status::MaxIterations => dexrouter_newton::Status::MaxIterations,
                    Status::MaxEvaluations => dexrouter_newton::Status::MaxEvaluations,
                    _ => dexrouter_newton::Status::NumericalStagnation,
                });
            BackendStatus::Newton(native)
        } else {
            BackendStatus::Lbfgsb(status)
        },
        hessian_products,
        hessian_evaluations,
        line_search_failure,
        assessment,
        max_abs_gradient,
        max_kkt_residual_ratio: kkt_ratio,
        max_deficit_value: deficit_value,
        max_deficit_ratio: deficit_ratio,
        numerical_lower_bounds: lower,
        objective_scale,
        optimizer_variables: blocks.iter().map(|b| b.tokens.len()).sum(),
        optimization_blocks: blocks.len(),
    })
}

#[derive(Debug)]
struct OptimizationBlock {
    tokens: Vec<usize>,
    markets: Vec<usize>,
    // Per-market local gradient indices; usize::MAX denotes a fixed endpoint.
    // Precompute once rather than binary-searching twice per pool per FG.
    market_tokens: Vec<[usize; 2]>,
}

fn optimization_blocks<U: UtilityConjugate, M: Market>(
    route: &Route<U, M>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    decompose: bool,
) -> Vec<OptimizationBlock> {
    let mut incident = vec![false; route.tokens];
    for (_, (a, b)) in &route.markets {
        incident[*a] = true;
        incident[*b] = true;
    }
    let retained: Vec<_> = (0..route.tokens)
        .map(|i| {
            lower[i] != upper[i] && (incident[i] || !route.objective.is_constant_coordinate(i))
        })
        .collect();
    let free: Vec<_> = (0..route.tokens).filter(|&i| retained[i]).collect();
    let groups = if decompose && route.objective.is_separable() {
        // Removing fixed vertices is exact: incident pools are functions of just
        // their remaining free endpoint, not couplings between its neighbors.
        crate::graph::components(
            route.tokens,
            route
                .markets
                .iter()
                .filter_map(|(_, (a, b))| (retained[*a] && retained[*b]).then_some((*a, *b))),
        )
        .into_iter()
        .map(|group| {
            group
                .into_iter()
                .filter(|&i| retained[i])
                .collect::<Vec<_>>()
        })
        .filter(|group| !group.is_empty())
        .collect::<Vec<_>>()
    } else if free.is_empty() {
        Vec::new()
    } else {
        vec![free]
    };
    let mut owner = vec![usize::MAX; route.tokens];
    let mut blocks: Vec<_> = groups
        .into_iter()
        .enumerate()
        .map(|(id, tokens)| {
            for &i in &tokens {
                owner[i] = id;
            }
            OptimizationBlock {
                tokens,
                markets: Vec::new(),
                market_tokens: Vec::new(),
            }
        })
        .collect();
    for (index, (_, (a, b))) in route.markets.iter().enumerate() {
        let id = if owner[*a] != usize::MAX {
            owner[*a]
        } else {
            owner[*b]
        };
        if id != usize::MAX {
            blocks[id].markets.push(index);
        }
    }
    for block in &mut blocks {
        block.market_tokens = block
            .markets
            .iter()
            .map(|&index| {
                let (_, (a, b)) = &route.markets[index];
                [*a, *b].map(|token| block.tokens.binary_search(&token).unwrap_or(usize::MAX))
            })
            .collect();
    }
    blocks
}

// A per-token high-water mark prevents oscillation or rotating the worst token
// from resetting the watchdog. Ratios below acceptance are all equivalent.
// Tiny reductions accumulate against the last meaningful mark (0.1%).
struct ResidualProgress {
    best: Vec<f64>,
    stagnant_rounds: usize,
}

impl ResidualProgress {
    fn new(ratios: impl Iterator<Item = f64>) -> Self {
        Self {
            best: ratios.map(|r| r.max(1.0)).collect(),
            stagnant_rounds: 0,
        }
    }

    fn stagnated(&mut self, ratios: impl Iterator<Item = f64>) -> bool {
        let mut improved = false;
        for (best, ratio) in self.best.iter_mut().zip(ratios) {
            let ratio = ratio.max(1.0);
            if ratio < *best && (ratio <= 1.0 || ratio <= *best * (1.0 - 1e-3)) {
                *best = ratio;
                improved = true;
            }
        }
        self.stagnant_rounds = if improved {
            0
        } else {
            self.stagnant_rounds + 1
        };
        self.stagnant_rounds >= 8
    }
}

fn token_residual_ratio(
    i: usize,
    price: &Array1<f64>,
    current: &RouteEvaluation,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    options: &SolveOptions,
) -> f64 {
    let allowance = options.absolute_residual_value
        + options.relative_residual_tolerance * current.turnover[i] * price[i];
    kkt_gradient(price[i], current.gradient[i], lower[i], upper[i]).abs() * price[i] / allowance
}

fn block_residual(
    block: &OptimizationBlock,
    price: &Array1<f64>,
    current: &RouteEvaluation,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    options: &SolveOptions,
) -> f64 {
    block
        .tokens
        .iter()
        .map(|&i| token_residual_ratio(i, price, current, lower, upper, options))
        .fold(0.0, f64::max)
}

// Resolve representability stalls without allowing uphill Newton steps. A
// diagonal correction of other coordinates can fund a coupled lattice move;
// rounding one coordinate alone can be an inferior local discrete minimum.
// Every probe uses the original centered objective and economic tolerances.
#[allow(clippy::too_many_arguments)]
fn rounding_repair<U: UtilityConjugate, M: Market>(
    route: &Route<U, M>,
    block: &OptimizationBlock,
    price: &Array1<f64>,
    current: &RouteEvaluation,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    math_lower: &Array1<f64>,
    math_upper: &Array1<f64>,
    options: &SolveOptions,
    armijo: f64,
    evaluations: &mut usize,
    models: &mut usize,
) -> Option<Array1<f64>> {
    if *evaluations + 2 > options.optimizer.max_evaluations {
        return None;
    }
    let model = relative_curvature_for_markets(route, price, block.markets.iter().copied())?;
    *models += 1;
    let i = *block.tokens.iter().max_by(|&&a, &&b| {
        token_residual_ratio(a, price, current, math_lower, math_upper, options).total_cmp(
            &token_residual_ratio(b, price, current, math_lower, math_upper, options),
        )
    })?;
    let before = block_residual(block, price, current, math_lower, math_upper, options);
    let mut free = vec![false; route.tokens];
    for &k in &block.tokens {
        free[k] = true;
    }
    let components = crate::graph::components(
        route.tokens,
        model
            .edges
            .iter()
            .filter_map(|&(a, b, _)| (free[a] && free[b]).then_some((a, b))),
    );
    let correlated = components.iter().find(|c| c.contains(&i))?;
    // Avoid simultaneous Jacobi corrections of strongly coupled prices. Use
    // one independent, representably descending coordinate as the background.
    let mut base = price.clone();
    let mut background = None;
    let mut best_decrease = 0.0;
    for &k in &block.tokens {
        if model.diagonal[k] > 0.0 && !correlated.contains(&k) {
            let relative = (-current.gradient[k] * price[k] / model.diagonal[k]).clamp(-0.5, 0.5);
            let candidate = relative
                .mul_add(price[k], price[k])
                .max(lower[k])
                .min(upper[k]);
            let step = (candidate - price[k]) / price[k];
            let decrease =
                -(current.gradient[k] * price[k] * step + 0.5 * model.diagonal[k] * step * step);
            if decrease.is_finite() && decrease > best_decrease {
                background = Some((k, candidate));
                best_decrease = decrease;
            }
        }
    }
    if let Some((k, p)) = background {
        base[k] = p;
    }
    let parent = model
        .edges
        .iter()
        .filter_map(|&(a, b, w)| {
            let j = if a == i {
                b
            } else if b == i {
                a
            } else {
                return None;
            };
            block.tokens.binary_search(&j).is_ok().then_some((j, w))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(j, _)| j);
    let mut candidates = vec![base.clone()];
    if model.diagonal[i] > 0.0 {
        let relative = (-current.gradient[i] * price[i] / model.diagonal[i]).clamp(-0.5, 0.5);
        let center = relative.mul_add(price[i], price[i]);
        for value in [center, center.next_down(), center.next_up()] {
            let mut p = base.clone();
            p[i] = value.max(lower[i]).min(upper[i]);
            candidates.push(p);
        }
    }
    for neighbor in [price[i].next_down(), price[i].next_up()] {
        let mut p = base.clone();
        p[i] = neighbor.max(lower[i]).min(upper[i]);
        candidates.push(p);
    }
    if let Some(j) = parent {
        for offset in [-1_i32, 1, -2, 2, -4, 4, -8, 8, -16, 16, -32, 32] {
            let mut p = base.clone();
            p[j] = price[j];
            for _ in 0..offset.abs() {
                p[j] = if offset < 0 {
                    p[j].next_down()
                } else {
                    p[j].next_up()
                };
            }
            p[j] = p[j].max(lower[j]).min(upper[j]);
            let mut rhs = crate::numerics::Sum::default();
            rhs.add(-current.gradient[i] * price[i]);
            for &(a, b, w) in &model.edges {
                let neighbor = if a == i {
                    b
                } else if b == i {
                    a
                } else {
                    continue;
                };
                rhs.add(w * ((p[neighbor] - price[neighbor]) / price[neighbor]));
            }
            let center = (rhs.total() / model.diagonal[i]).mul_add(price[i], price[i]);
            for value in [center, center.next_down(), center.next_up()] {
                p[i] = value.max(lower[i]).min(upper[i]);
                candidates.push(p.clone());
            }
        }
    }
    let mut best = None;
    let mut best_ratio = before;
    for candidate in candidates {
        // Reserve independent full-route verification of the selected candidate.
        if *evaluations + 2 > options.optimizer.max_evaluations {
            break;
        }
        if candidate == *price || candidate.iter().any(|p| !p.is_finite()) {
            continue;
        }
        let mut slope = crate::numerics::Sum::default();
        for (g, (p, r)) in current.gradient.iter().zip(candidate.iter().zip(price)) {
            slope.add(g * (p - r));
        }
        let slope = slope.total();
        if !slope.is_finite() || slope >= 0.0 {
            continue;
        }
        let (evaluation, change) = route.evaluate_centered(&candidate, Some((price, current)));
        *evaluations += 1;
        if !evaluation.cost.is_finite()
            || evaluation.gradient.iter().any(|g| !g.is_finite())
            || evaluation.turnover.iter().any(|t| !t.is_finite())
            || !change.is_finite()
            || change > armijo * slope
        {
            continue;
        }
        let ratio = block_residual(
            block,
            &candidate,
            &evaluation,
            math_lower,
            math_upper,
            options,
        );
        if ratio < best_ratio && (ratio <= 1.0 || ratio < 0.9 * before) {
            best_ratio = ratio;
            best = Some(candidate);
            if ratio <= 1.0 {
                break;
            }
        }
    }
    best
}

// Outside-block prices are unchanged. For an opted-in separable utility this
// is exactly the original centered objective with constant pool terms omitted.
fn evaluate_block<U: UtilityConjugate, M: Market>(
    route: &Route<U, M>,
    block: &OptimizationBlock,
    price: &Array1<f64>,
    reference_price: &Array1<f64>,
    reference: &[crate::market::MarketEvaluation],
    previous: Option<(&Array1<f64>, &[crate::market::MarketEvaluation])>,
    accurate_origin_change: bool,
) -> (Vec<f64>, f64, Vec<crate::market::MarketEvaluation>) {
    let utility_gradient = route.objective.grad(price);
    let mut gradient = vec![crate::numerics::Sum::default(); block.tokens.len()];
    for (local, &i) in block.tokens.iter().enumerate() {
        gradient[local].add(utility_gradient[i]);
    }
    let mut change = crate::numerics::Sum::default();
    change.add(route.objective.value_change(price, reference_price));
    let mut trades = Vec::with_capacity(block.markets.len());
    for (local_market, &index) in block.markets.iter().enumerate() {
        let (market, (a, b)) = &route.markets[index];
        let v = [price[*a], price[*b]];
        // Active bounds and rounded backtracking steps often leave most pools
        // unchanged. Quotes are pure functions of their two physical prices;
        // even a rejected trial is a valid cache, never an accepted iterate.
        let trade = if let Some((p, trades)) = previous
            && v == [p[*a], p[*b]]
        {
            trades[local_market]
        } else if v == [reference_price[*a], reference_price[*b]] {
            reference[local_market]
        } else {
            market.evaluate(v)
        };
        trades.push(trade);
        change.add(if v == [reference_price[*a], reference_price[*b]] {
            0.0
        } else if accurate_origin_change {
            market.value_change(
                v,
                [reference_price[*a], reference_price[*b]],
                trade,
                reference[local_market],
            )
        } else {
            // With per-search differences this value is only the absolute
            // function-reduction scale, never a Wolfe comparison or a decrease.
            // Do not calculate two expensive stable remainders per pool.
            trade.value - reference[local_market].value
        });
        for (local, index) in block.market_tokens[local_market].into_iter().enumerate() {
            if index != usize::MAX {
                gradient[index].add(trade.output[local] - trade.input[local]);
            }
        }
    }
    (
        gradient
            .into_iter()
            .map(crate::numerics::Sum::total)
            .collect(),
        change.total(),
        trades,
    )
}

fn block_value_change<U: UtilityConjugate, M: Market>(
    route: &Route<U, M>,
    block: &OptimizationBlock,
    price: &Array1<f64>,
    reference_price: &Array1<f64>,
    trades: &[crate::market::MarketEvaluation],
    reference: &[crate::market::MarketEvaluation],
) -> f64 {
    let mut change = crate::numerics::Sum::default();
    change.add(route.objective.value_change(price, reference_price));
    for (local, &index) in block.markets.iter().enumerate() {
        let (market, (a, b)) = &route.markets[index];
        if price[*a] == reference_price[*a] && price[*b] == reference_price[*b] {
            continue;
        }
        change.add(market.value_change(
            [price[*a], price[*b]],
            [reference_price[*a], reference_price[*b]],
            trades[local],
            reference[local],
        ));
    }
    change.total()
}

struct BlockHessian {
    schur_precondition: bool,
    tree_precondition: bool,
    curvature: RelativeCurvature,
    tokens: Vec<usize>,
    factors: Vec<f64>,
    objective_scale: f64,
}
impl dexrouter_newton::Hessian for BlockHessian {
    fn apply(&self, vector: &[f64], product: &mut [f64]) -> anyhow::Result<()> {
        let mut relative = vec![0.0; self.curvature.diagonal.len()];
        for (local, &token) in self.tokens.iter().enumerate() {
            relative[token] = vector[local] * self.factors[local];
        }
        let applied = self.curvature.apply(&relative);
        for (local, &token) in self.tokens.iter().enumerate() {
            product[local] = applied[token] * self.factors[local] / self.objective_scale;
        }
        Ok(())
    }
    fn diagonal(&self, diagonal: &mut [f64]) -> anyhow::Result<()> {
        for (local, &token) in self.tokens.iter().enumerate() {
            diagonal[local] =
                self.curvature.diagonal[token] * self.factors[local] * self.factors[local]
                    / self.objective_scale;
        }
        Ok(())
    }
    fn preconditioner(
        &self,
        metric: &[f64],
        free: &[bool],
        damping: f64,
    ) -> anyhow::Result<Option<Box<dyn dexrouter_newton::Preconditioner>>> {
        if !self.tree_precondition && !self.schur_precondition {
            return Ok(None);
        }
        let mut local = vec![usize::MAX; self.curvature.diagonal.len()];
        for (i, &token) in self.tokens.iter().enumerate() {
            local[token] = i;
        }
        let edges: Vec<_> = self
            .curvature
            .edges
            .iter()
            .map(|&(a, b, w)| (local[a], local[b], w))
            .collect();
        let utility: Vec<_> = self
            .tokens
            .iter()
            .map(|&i| self.curvature.utility_diagonal[i])
            .collect();
        if self.schur_precondition
            && let Some(p) = crate::schur_preconditioner::SchurPreconditioner::new(
                &edges,
                &utility,
                &self.factors,
                self.objective_scale,
                metric,
                free,
                damping,
            )
        {
            return Ok(Some(Box::new(p)));
        }
        if !self.tree_precondition {
            return Ok(None);
        }
        Ok(crate::tree_preconditioner::TreePreconditioner::new(
            &edges,
            &utility,
            &self.factors,
            self.objective_scale,
            metric,
            free,
            damping,
        )
        .map(|p| Box::new(p) as Box<dyn dexrouter_newton::Preconditioner>))
    }
}

struct RelativeCurvature {
    utility_diagonal: Vec<f64>,
    diagonal: Vec<f64>,
    edges: Vec<(usize, usize, f64)>,
}

impl RelativeCurvature {
    fn apply(&self, x: &[f64]) -> Vec<f64> {
        let mut out: Vec<_> = x
            .iter()
            .zip(&self.utility_diagonal)
            .map(|(x, d)| x * d)
            .collect();
        for &(a, b, w) in &self.edges {
            let delta = w * (x[a] - x[b]);
            out[a] += delta;
            out[b] -= delta;
        }
        out
    }
}

fn relative_curvature<U: UtilityConjugate, M: Market>(
    route: &Route<U, M>,
    price: &Array1<f64>,
) -> Option<RelativeCurvature> {
    relative_curvature_for_markets(route, price, 0..route.markets.len())
}

fn relative_curvature_for_markets<U: UtilityConjugate, M: Market>(
    route: &Route<U, M>,
    price: &Array1<f64>,
    markets: impl Iterator<Item = usize>,
) -> Option<RelativeCurvature> {
    let utility = route.objective.diagonal_hessian(price)?;
    if utility.len() != route.tokens {
        return None;
    }
    let utility_diagonal: Vec<_> = utility
        .iter()
        .zip(price)
        .map(|(&h, &p)| h * p * p)
        .collect();
    let mut diagonal = utility_diagonal.clone();
    let mut edges = Vec::with_capacity(markets.size_hint().0);
    for index in markets {
        let (market, (a, b)) = &route.markets[index];
        let weight = market.relative_curvature([price[*a], price[*b]])?;
        if !weight.is_finite() || weight < 0.0 {
            return None;
        }
        if a != b && weight > 0.0 {
            diagonal[*a] += weight;
            diagonal[*b] += weight;
            edges.push((*a, *b, weight));
        }
    }
    if utility_diagonal
        .iter()
        .chain(&diagonal)
        .any(|d| !d.is_finite() || *d < 0.0)
    {
        return None;
    }
    Some(RelativeCurvature {
        utility_diagonal,
        diagonal,
        edges,
    })
}

// Sparse Newton-CG in relative increments. Bound-active coordinates are masked;
// box projection and an Armijo test on g (not the damped model) safeguard the step.
#[allow(clippy::too_many_arguments)]
fn graph_recovery<U: UtilityConjugate, M: Market>(
    route: &Route<U, M>,
    block: &OptimizationBlock,
    price: &mut Array1<f64>,
    current: &RouteEvaluation,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    options: &SolveOptions,
    evaluations: &mut usize,
) -> bool {
    let Some(model) = relative_curvature(route, price) else {
        return false;
    };
    let mut movable = vec![false; route.tokens];
    for &i in &block.tokens {
        movable[i] = !((price[i] <= lower[i] && current.gradient[i] >= 0.0)
            || (price[i] >= upper[i] && current.gradient[i] <= 0.0));
    }
    let normalization = block
        .tokens
        .iter()
        .map(|&i| model.diagonal[i])
        .fold(0.0, f64::max);
    if normalization <= 0.0 {
        return false;
    }
    // Homogeneity gives H_relative = a weighted graph Laplacian. Grounded
    // components (fixed/active price or positive utility curvature) have no
    // scale null mode: solve their Newton equations without biasing weak modes.
    let grounded =
        crate::graph::components(route.tokens, model.edges.iter().map(|&(a, b, _)| (a, b)))
            .iter()
            .all(|component| {
                component
                    .iter()
                    .any(|&i| !movable[i] || model.utility_diagonal[i] > 0.0)
            });
    let damping = if grounded { 0.0 } else { 1e-8 };
    let regularizer: Vec<_> = model
        .diagonal
        .iter()
        .map(|&d| {
            // Relative (Jacobi metric) damping must not swamp low-liquidity pools
            // merely because an unrelated edge has enormous liquidity.
            damping * if d > 0.0 { d / normalization } else { 1.0 }
        })
        .collect();
    let math_lower = route.objective.lower_bounds();
    let math_upper = route.objective.upper_bounds();
    let before_ratio = block_residual(block, price, current, &math_lower, &math_upper, options);
    let diagonal: Vec<_> = (0..route.tokens)
        .map(|i| {
            if movable[i] {
                model.diagonal[i] / normalization + regularizer[i]
            } else {
                1.0
            }
        })
        .collect();
    let rhs: Vec<_> = (0..route.tokens)
        .map(|i| {
            if movable[i] {
                -current.gradient[i] * price[i] / normalization
            } else {
                0.0
            }
        })
        .collect();
    let Some(direction) = crate::graph::pcg(&diagonal, &rhs, |x| {
        let masked: Vec<_> = x
            .iter()
            .enumerate()
            .map(|(i, &x)| if movable[i] { x } else { 0.0 })
            .collect();
        let product = model.apply(&masked);
        product
            .iter()
            .enumerate()
            .map(|(i, &h)| {
                if movable[i] {
                    h / normalization + regularizer[i] * x[i]
                } else {
                    x[i]
                }
            })
            .collect()
    }) else {
        return false;
    };
    let norm = direction.iter().map(|x| x.abs()).fold(0.0, f64::max);
    if norm == 0.0 {
        return false;
    }
    let mut alpha = (0.5 / norm).min(1.0);
    for _ in 0..20 {
        // Reserve the caller's original-objective verification evaluation.
        if *evaluations + 1 >= options.optimizer.max_evaluations {
            break;
        }
        let mut candidate = price.clone();
        let mut slope = crate::numerics::Sum::default();
        for &i in &block.tokens {
            candidate[i] = (alpha * direction[i])
                .mul_add(price[i], price[i])
                .max(lower[i])
                .min(upper[i]);
            slope.add(current.gradient[i] * (candidate[i] - price[i]));
        }
        let slope = slope.total();
        if slope.is_finite() && slope < 0.0 && candidate != *price {
            let (evaluation, change) = route.evaluate_centered(&candidate, Some((price, current)));
            *evaluations += 1;
            let after_ratio = block_residual(
                block,
                &candidate,
                &evaluation,
                &math_lower,
                &math_upper,
                options,
            );
            if change.is_finite()
                && change <= 1e-4 * slope
                && evaluation.gradient.iter().all(|g| g.is_finite())
                && (after_ratio <= 1.0 || after_ratio < 0.9 * before_ratio)
            {
                *price = candidate;
                return true;
            }
        }
        alpha *= 0.5;
    }
    false
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

fn coordinate_bound(bound: f64, origin: f64, scale: f64, sqrt: bool) -> f64 {
    let affine = (bound - origin) / scale;
    if sqrt {
        affine * (2.0 * (origin.sqrt() / (bound.sqrt() + origin.sqrt())))
    } else {
        affine
    }
}

fn physical_price(
    x: &[f64],
    tokens: &[usize],
    origin: &Array1<f64>,
    scale: &Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    sqrt: bool,
) -> Array1<f64> {
    let mut price = origin.clone();
    for (&x, &i) in x.iter().zip(tokens) {
        price[i] = if sqrt {
            // p = p_ref (1+t)^2, t = x scale / (2 p_ref). Expanded around
            // p_ref so zero maps back exactly, including prices whose square
            // root would otherwise lose an ulp on a sqrt/square round trip.
            let t = 0.5 * x * (scale[i] / origin[i]);
            if t.abs() <= 0.5 {
                (t * (2.0 + t)).mul_add(origin[i], origin[i])
            } else {
                let root = 1.0 + t;
                (origin[i] * root) * root
            }
        } else {
            x.mul_add(scale[i], origin[i])
        }
        .max(lower[i])
        .min(upper[i]);
    }
    price
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
    fn rounding_repair_counts_quotes_and_reserves_independent_verification() {
        use crate::{
            market::{Market, MarketEvaluation},
            utility::BasketLiquidation,
        };
        use std::{cell::Cell, rc::Rc};
        struct Counted {
            inner: UniswapV2,
            quotes: Rc<Cell<usize>>,
        }
        impl Market for Counted {
            fn arbitrage(&self, v: [f64; 2]) -> ([f64; 2], [f64; 2]) {
                let t = self.evaluate(v);
                (t.input, t.output)
            }
            fn evaluate(&self, v: [f64; 2]) -> MarketEvaluation {
                self.quotes.set(self.quotes.get() + 1);
                self.inner.evaluate(v)
            }
            fn value_change(
                &self,
                v: [f64; 2],
                r: [f64; 2],
                t: MarketEvaluation,
                old: MarketEvaluation,
            ) -> f64 {
                self.inner.value_change(v, r, t, old)
            }
            fn relative_curvature(&self, v: [f64; 2]) -> Option<f64> {
                self.inner.relative_curvature(v)
            }
        }
        let quotes = Rc::new(Cell::new(0));
        let route = Route {
            objective: BasketLiquidation {
                out: 0,
                inputs: arr1(&[0.0, 1.0]),
            },
            tokens: 2,
            markets: vec![(
                Counted {
                    inner: UniswapV2::new(1000.0, 1000.0, 1.0),
                    quotes: quotes.clone(),
                },
                (1, 0),
            )],
        };
        let price = arr1(&[1.0, 1.0]);
        let current = route.evaluate(&price);
        let lower = arr1(&[1.0, 1e-12]);
        let upper = arr1(&[1.0, f64::INFINITY]);
        let ml = route.objective.lower_bounds();
        let mu = route.objective.upper_bounds();
        let block = OptimizationBlock {
            tokens: vec![1],
            markets: vec![0],
            market_tokens: vec![[0, usize::MAX]],
        };
        for cap in [1, 2, 3, 10] {
            let mut options = SolveOptions::default();
            options.optimizer.max_evaluations = cap;
            let mut evaluations = 0;
            let mut models = 0;
            quotes.set(0);
            let repaired = rounding_repair(
                &route,
                &block,
                &price,
                &current,
                &lower,
                &upper,
                &ml,
                &mu,
                &options,
                1e-4,
                &mut evaluations,
                &mut models,
            );
            assert_eq!(quotes.get(), evaluations);
            if cap == 1 {
                assert!(repaired.is_none());
                assert_eq!(evaluations, 0);
                assert_eq!(models, 0);
                continue;
            }
            assert!(evaluations + 1 <= cap);
            assert_eq!(models, 1);
            let repaired = repaired.expect("a rounded scalar Newton correction must descend");
            let (after, change) = route.evaluate_centered(&repaired, Some((&price, &current)));
            let slope = current.gradient[1] * (repaired[1] - price[1]);
            assert!(slope < 0.0 && change <= 1e-4 * slope);
            assert!(after.cost < current.cost);
            assert!(
                block_residual(&block, &repaired, &after, &ml, &mu, &options)
                    < 0.9 * block_residual(&block, &price, &current, &ml, &mu, &options)
            );
        }
    }

    #[test]
    fn residual_watchdog_ignores_tiny_changes_and_accepted_tokens() {
        let mut progress = ResidualProgress::new([4986.0, 0.8].into_iter());
        for round in 1..=8 {
            assert_eq!(
                progress.stagnated([4986.0 - round as f64 * 1e-8, 0.8 / round as f64].into_iter()),
                round == 8
            );
        }
    }

    #[test]
    fn residual_watchdog_tracks_each_token_and_does_not_reward_oscillation() {
        let mut progress = ResidualProgress::new([5000.0, 100.0].into_iter());
        for _ in 0..7 {
            assert!(!progress.stagnated([5000.0, 100.0].into_iter()));
        }
        // The worst token is unchanged; progress on another violating token counts.
        assert!(!progress.stagnated([5000.0, 90.0].into_iter()));
        for round in 1..=8 {
            let ratio = if round % 2 == 0 { 90.0 } else { 100.0 };
            assert_eq!(progress.stagnated([5000.0, ratio].into_iter()), round == 8);
        }
    }

    #[test]
    fn residual_watchdog_accumulates_small_improvements_and_accepts_boundary() {
        let mut progress = ResidualProgress::new([100.0, 1.00001].into_iter());
        for round in 1..=7 {
            assert!(!progress.stagnated([100.0 - round as f64 * 0.02, 1.00001].into_iter()));
        }
        assert_eq!(progress.stagnant_rounds, 2);
        assert!(!progress.stagnated([99.86, 1.0].into_iter()));
        assert_eq!(progress.stagnant_rounds, 0);
    }

    #[test]
    fn block_kernel_is_the_original_centered_objective_and_gradient() {
        let route = Route {
            objective: crate::utility::BasketLiquidation {
                out: 0,
                inputs: arr1(&[0.0, 10.0, 20.0, 0.0]),
            },
            markets: vec![
                (UniswapV2::new(1000.0, 2000.0, 0.997), (0, 1)),
                (UniswapV2::new(700.0, 3000.0, 0.997), (0, 2)),
            ],
            tokens: 4,
        };
        let reference_price = arr1(&[1.0, 0.4, 0.3, 9.0]);
        let reference = route.evaluate(&reference_price);
        let lower = arr1(&[1.0, 0.0, 0.0, 0.0]);
        let upper = arr1(&[1.0, f64::INFINITY, f64::INFINITY, f64::INFINITY]);
        let blocks = optimization_blocks(&route, &lower, &upper, true);
        assert_eq!(blocks.len(), 2);
        for block in blocks {
            let mut price = reference_price.clone();
            for &i in &block.tokens {
                price[i] *= 1.1;
            }
            let reference_trades: Vec<_> = block
                .markets
                .iter()
                .map(|&i| reference.market_evaluations[i])
                .collect();
            let (gradient, change, _) = evaluate_block(
                &route,
                &block,
                &price,
                &reference_price,
                &reference_trades,
                None,
                true,
            );
            let (full, full_change) =
                route.evaluate_centered(&price, Some((&reference_price, &reference)));
            assert!((change - full_change).abs() < 1e-12);
            for (local, &i) in block.tokens.iter().enumerate() {
                assert_eq!(gradient[local], full.gradient[i]);
            }
        }
    }

    #[test]
    fn block_quotes_reuse_only_identical_physical_market_prices() {
        use std::cell::Cell;
        struct Counted {
            pool: UniswapV2,
            quotes: Cell<usize>,
            differences: Cell<usize>,
        }
        impl Market for Counted {
            fn arbitrage(&self, v: [f64; 2]) -> ([f64; 2], [f64; 2]) {
                self.pool.arbitrage(v)
            }
            fn evaluate(&self, v: [f64; 2]) -> crate::market::MarketEvaluation {
                self.quotes.set(self.quotes.get() + 1);
                self.pool.evaluate(v)
            }
            fn value_change(
                &self,
                v: [f64; 2],
                r: [f64; 2],
                t: crate::market::MarketEvaluation,
                rt: crate::market::MarketEvaluation,
            ) -> f64 {
                self.differences.set(self.differences.get() + 1);
                self.pool.value_change(v, r, t, rt)
            }
        }
        let route = Route {
            objective: NonnegativeLinear {
                c: arr1(&[1.0, 0.0, 0.0]),
            },
            tokens: 3,
            markets: [1, 2]
                .into_iter()
                .map(|i| {
                    (
                        Counted {
                            pool: UniswapV2::new(1000.0, 2000.0, 0.997),
                            quotes: Cell::new(0),
                            differences: Cell::new(0),
                        },
                        (0, i),
                    )
                })
                .collect(),
        };
        let block = OptimizationBlock {
            tokens: vec![1, 2],
            markets: vec![0, 1],
            market_tokens: vec![[usize::MAX, 0], [usize::MAX, 1]],
        };
        let origin = arr1(&[1.0, 0.4, 0.3]);
        let reference = route.evaluate(&origin);
        let first = arr1(&[1.0, 0.45, 0.3]);
        let (_, _, first_trades) = evaluate_block(
            &route,
            &block,
            &first,
            &origin,
            &reference.market_evaluations,
            None,
            false,
        );
        assert_eq!(route.markets[0].0.quotes.get(), 2);
        assert_eq!(route.markets[1].0.quotes.get(), 1);
        for (m, _) in &route.markets {
            m.quotes.set(0);
        }
        // The previous quote may be a rejected trial: reuse its values only for
        // exactly matching pool prices, while keeping the search origin intact.
        let second = arr1(&[1.0, 0.45, 0.35]);
        let (gradient, change, trades) = evaluate_block(
            &route,
            &block,
            &second,
            &origin,
            &reference.market_evaluations,
            Some((&first, &first_trades)),
            false,
        );
        assert_eq!(route.markets[0].0.quotes.get(), 0);
        assert_eq!(route.markets[1].0.quotes.get(), 1);
        let delta = block_value_change(&route, &block, &second, &first, &trades, &first_trades);
        assert_eq!(route.markets[0].0.differences.get(), 0);
        assert_eq!(route.markets[1].0.differences.get(), 1);
        let full = route.evaluate(&second);
        for (local, &i) in block.tokens.iter().enumerate() {
            assert_eq!(gradient[local], full.gradient[i]);
        }
        assert!((change - (full.cost - reference.cost)).abs() < 1e-12);
        let first_evaluation = route.evaluate(&first);
        let (_, full_delta) = route.evaluate_centered(&second, Some((&first, &first_evaluation)));
        assert_eq!(delta, full_delta);
        for (m, _) in &route.markets {
            m.quotes.set(0);
        }
        let (_, _, at_origin) = evaluate_block(
            &route,
            &block,
            &origin,
            &origin,
            &reference.market_evaluations,
            Some((&second, &trades)),
            true,
        );
        for (m, _) in &route.markets {
            assert_eq!(m.quotes.get(), 0);
        }
        for (t, r) in at_origin.iter().zip(&reference.market_evaluations) {
            assert_eq!(t.input, r.input);
            assert_eq!(t.output, r.output);
            assert_eq!(t.value, r.value);
        }
    }

    #[test]
    fn forest_metric_recovers_a_weak_mode_hidden_by_strong_liquidity() {
        let solve = |tree| {
            let w = 1e12;
            let weak = 1e-12;
            dexrouter_newton::minimize(
                vec![0.0, 0.0],
                vec![dexrouter_newton::Bound::unbounded(); 2],
                dexrouter_newton::Options {
                    pgtol: 1e-24,
                    max_iterations: 50,
                    max_evaluations: 500,
                    max_hessian_products: 500,
                    ..Default::default()
                },
                |x, g, r| {
                    let edge = x[0] - x[1];
                    g[0] = w * edge + weak * (x[0] - 1.0);
                    g[1] = -w * edge;
                    let change = r.map_or(0.0, |r| {
                        0.5 * w * (edge - (r[0] - r[1])) * (edge + r[0] - r[1])
                            + 0.5 * weak * (x[0] - r[0]) * (x[0] + r[0] - 2.0)
                    });
                    Ok(dexrouter_newton::Evaluation::with_difference(
                        0.5 * w * edge * edge + 0.5 * weak * (x[0] - 1.0).powi(2),
                        change,
                    ))
                },
                |_| {
                    Ok(BlockHessian {
                        tree_precondition: tree,
                        schur_precondition: false,
                        curvature: RelativeCurvature {
                            utility_diagonal: vec![weak, 0.0],
                            diagonal: vec![w + weak, w],
                            edges: vec![(0, 1, w)],
                        },
                        tokens: vec![0, 1],
                        factors: vec![1.0, 1.0],
                        objective_scale: 1.0,
                    })
                },
            )
            .unwrap()
        };
        let jacobi = solve(false);
        let tree = solve(true);
        println!(
            "weak-mode jacobi={:?} HVP={} PG={} tree={:?} HVP={} PG={}",
            jacobi.status,
            jacobi.hessian_products,
            jacobi.projected_gradient_inf_norm,
            tree.status,
            tree.hessian_products,
            tree.projected_gradient_inf_norm
        );
        assert_eq!(
            tree.status,
            dexrouter_newton::Status::ConvergedProjectedGradient
        );
        assert!(tree.hessian_products < jacobi.hessian_products);
        for p in tree.x {
            assert!((p - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn block_hessian_matches_gradient_differences_and_masks_fixed_prices() {
        use crate::market::UniswapV3;
        use dexrouter_newton::Hessian;
        fn check<M: Market>(market: M) {
            let route = Route {
                objective: NonnegativeLinear {
                    c: arr1(&[1.0, 0.0, 0.0]),
                },
                markets: vec![(market, (0, 1))],
                tokens: 3,
            };
            let p = arr1(&[1.0, 0.65, 7.0]);
            let scale = arr1(&[1.0, 2.0, 3.0]);
            let h = BlockHessian {
                curvature: relative_curvature(&route, &p).unwrap(),
                tokens: vec![1, 2],
                factors: vec![scale[1] / p[1], scale[2] / p[2]],
                objective_scale: 11.0,
                tree_precondition: true,
                schur_precondition: false,
            };
            let v = [0.3, -0.2];
            let mut hv = [0.0; 2];
            h.apply(&v, &mut hv).unwrap();
            let mut plus = p.clone();
            let mut minus = p.clone();
            let eps = 1e-6;
            for (i, &t) in h.tokens.iter().enumerate() {
                plus[t] += eps * scale[t] * v[i];
                minus[t] -= eps * scale[t] * v[i];
            }
            let gp = route.evaluate(&plus).gradient;
            let gm = route.evaluate(&minus).gradient;
            for (i, &t) in h.tokens.iter().enumerate() {
                let fd = (gp[t] - gm[t]) * scale[t] / (2.0 * eps * 11.0);
                assert!(
                    (fd - hv[i]).abs() < 1e-6 * fd.abs().max(1.0),
                    "{fd} != {}",
                    hv[i]
                );
            }
            assert_eq!(hv[1], 0.0);
        }
        check(UniswapV2::new(100.0, 120.0, 0.997));
        check(UniswapV3::new(
            1.0,
            vec![2.0, 1.1, 0.8, 0.5],
            vec![50.0, 100.0, 150.0],
            0.997,
        ));
    }

    #[test]
    fn grounded_newton_does_not_damp_a_weak_pool_by_unrelated_liquidity() {
        let route = Route {
            objective: crate::utility::BasketLiquidation {
                out: 0,
                inputs: arr1(&[0.0, 0.0, 10.0]),
            },
            markets: vec![
                (UniswapV2::new(1e12, 1e12, 1.0), (0, 1)),
                (UniswapV2::new(1000.0, 1000.0, 1.0), (0, 2)),
            ],
            tokens: 3,
        };
        let lower = arr1(&[1.0, 1e-12, 1e-12]);
        let upper = arr1(&[1.0, f64::INFINITY, f64::INFINITY]);
        let block = optimization_blocks(&route, &lower, &upper, false).remove(0);
        let mut price = arr1(&[1.0, 1.0, 0.99]);
        let current = route.evaluate(&price);
        let mut evaluations = 1;
        assert!(graph_recovery(
            &route,
            &block,
            &mut price,
            &current,
            &lower,
            &upper,
            &SolveOptions::default(),
            &mut evaluations
        ));
        let next = route.evaluate(&price);
        assert_eq!(price[0], 1.0);
        assert_eq!(price[1], 1.0);
        assert!(next.gradient[2].abs() < 0.05 * current.gradient[2].abs());
        assert!(next.cost < current.cost);
        assert_eq!(evaluations, 2);
    }

    #[test]
    fn sqrt_coordinate_map_preserves_origin_bounds_and_chain_rule() {
        let origin = arr1(&[1.7, 1e-12, 1e12]);
        let scale = origin.clone();
        let lower = origin.mapv(|p| p * 1e-12);
        let upper = origin.mapv(|p| p * 100.0);
        let tokens = [0, 1, 2];
        assert_eq!(
            physical_price(&[0.0; 3], &tokens, &origin, &scale, &lower, &upper, true),
            origin
        );
        for multiplier in [1e-12, 0.25, 1.0, 100.0] {
            let x: Vec<_> = (0..3)
                .map(|i| coordinate_bound(origin[i] * multiplier, origin[i], scale[i], true))
                .collect();
            let p = physical_price(&x, &tokens, &origin, &scale, &lower, &upper, true);
            for i in 0..3 {
                assert!((p[i] / (origin[i] * multiplier) - 1.0).abs() < 1e-8);
            }
        }
        let x = [0.2, -0.3, 0.8];
        for i in 0..3 {
            let mut plus = x;
            plus[i] += 1e-5;
            let mut minus = x;
            minus[i] -= 1e-5;
            let fp = physical_price(&plus, &tokens, &origin, &scale, &lower, &upper, true)[i];
            let fm = physical_price(&minus, &tokens, &origin, &scale, &lower, &upper, true)[i];
            let jacobian = scale[i] * (1.0 + 0.5 * x[i] * scale[i] / origin[i]);
            assert!(((fp - fm) / (2e-5 * jacobian) - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn graph_recovery_preserves_bounds_objective_and_budget() {
        let route = Route {
            objective: crate::utility::BasketLiquidation {
                out: 0,
                inputs: arr1(&[0.0, 10.0]),
            },
            markets: vec![(UniswapV2::new(1000.0, 2000.0, 0.997), (0, 1))],
            tokens: 2,
        };
        let lower = arr1(&[1.0, 0.01]);
        let upper = arr1(&[1.0, f64::INFINITY]);
        let blocks = optimization_blocks(&route, &lower, &upper, true);
        let mut price = arr1(&[1.0, 0.4]);
        let current = route.evaluate(&price);
        let mut evaluations = 1;
        let options = SolveOptions::default();
        assert!(graph_recovery(
            &route,
            &blocks[0],
            &mut price,
            &current,
            &lower,
            &upper,
            &options,
            &mut evaluations
        ));
        assert_eq!(price[0], 1.0);
        assert!(price[1] >= lower[1]);
        let next = route.evaluate(&price);
        assert!(next.cost < current.cost);
        let math_lower = route.objective.lower_bounds();
        let math_upper = route.objective.upper_bounds();
        assert!(
            block_residual(
                &blocks[0],
                &price,
                &next,
                &math_lower,
                &math_upper,
                &options
            ) < block_residual(
                &blocks[0],
                &arr1(&[1.0, 0.4]),
                &current,
                &math_lower,
                &math_upper,
                &options
            )
        );
        let mut no_budget = options;
        no_budget.optimizer.max_evaluations = evaluations + 1;
        let before = price.clone();
        assert!(!graph_recovery(
            &route,
            &blocks[0],
            &mut price,
            &next,
            &lower,
            &upper,
            &no_budget,
            &mut evaluations
        ));
        assert_eq!(before, price);
    }

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
