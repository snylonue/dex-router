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
    /// Split only when the utility explicitly promises separability.
    pub decompose: bool,
    /// Prefer exact regional pool curvature; custom implementations fall back to probes.
    pub analytic_curvature: bool,
    /// Try graph-Newton recovery before coordinate recovery.
    /// Damping affects the step matrix, never the economic objective.
    pub graph_precondition: bool,
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
            decompose: true,
            analytic_curvature: true,
            graph_precondition: true,
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
    /// Fixed prices are absent from every optimizer vector.
    pub optimizer_variables: usize,
    pub optimization_blocks: usize,
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
    let blocks = optimization_blocks(&route, &lower, &upper, options.decompose);
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
    let mut zero_progress_restarts = 0;
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
            relative_curvature(&route, &price)
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
                    Some((lower[i] - price[i]) / scale[i]),
                    upper[i]
                        .is_finite()
                        .then_some((upper[i] - price[i]) / scale[i]),
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
        let result = minimize(vec![0.0; block.tokens.len()], bounds, optimizer, |x, g| {
            let p = physical_price(x, &block.tokens, &price, &scale, &lower, &upper);
            let (gradient, cost_change) = evaluate_block(&route, block, &p, &price, &current);
            for (local, &i) in block.tokens.iter().enumerate() {
                g[local] = gradient[local] * scale[i] / objective_scale;
            }
            Ok(cost_change / objective_scale)
        })?;
        evaluations += result.evaluations;
        iterations += result.iterations;
        let next_price = physical_price(&result.x, &block.tokens, &price, &scale, &lower, &upper);
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
        if matches!(
            status,
            Status::LineSearchFailed
                | Status::MaxIterations
                | Status::ConvergedProjectedGradient
                | Status::ConvergedFunctionReduction
        ) && block_residual(block, &price, &current, &math_lower, &math_upper, &options) > 1.0
            && options.precondition
            && iterations < options.optimizer.max_iterations
            && evaluations + 2 < options.optimizer.max_evaluations
        {
            let before = price.clone();
            let graph_recovered = options.graph_precondition
                && options.analytic_curvature
                && graph_recovery(
                    &route,
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
                    &route,
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
            let ratios = residuals(&price, &current, &math_lower, &math_upper, &options);
            if ratios.0 <= 1.0 && ratios.2 <= 1.0 {
                break;
            }
            if evaluations >= options.optimizer.max_evaluations {
                status = Status::MaxEvaluations;
                break;
            }
            if price != before {
                zero_progress_restarts = 0;
                restarts += 1;
                continue;
            }
        }
        if result.iterations == 0 {
            zero_progress_restarts += 1;
        } else {
            zero_progress_restarts = 0;
        }
        if status == Status::MaxEvaluations
            || (status == Status::LineSearchFailed && result.iterations == 0)
            || (unchanged && status != Status::ConvergedProjectedGradient)
            || zero_progress_restarts >= 8
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
        optimizer_variables: blocks.iter().map(|b| b.tokens.len()).sum(),
        optimization_blocks: blocks.len(),
    })
}

#[derive(Debug)]
struct OptimizationBlock {
    tokens: Vec<usize>,
    markets: Vec<usize>,
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
    blocks
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
        .map(|&i| {
            let allowance = options.absolute_residual_value
                + options.relative_residual_tolerance * current.turnover[i] * price[i];
            kkt_gradient(price[i], current.gradient[i], lower[i], upper[i]).abs() * price[i]
                / allowance
        })
        .fold(0.0, f64::max)
}

// Outside-block prices are unchanged. For an opted-in separable utility this
// is exactly the original centered objective with constant pool terms omitted.
fn evaluate_block<U: UtilityConjugate, M: Market>(
    route: &Route<U, M>,
    block: &OptimizationBlock,
    price: &Array1<f64>,
    reference_price: &Array1<f64>,
    reference: &RouteEvaluation,
) -> (Vec<f64>, f64) {
    let utility_gradient = route.objective.grad(price);
    let mut gradient = vec![crate::numerics::Sum::default(); block.tokens.len()];
    for (local, &i) in block.tokens.iter().enumerate() {
        gradient[local].add(utility_gradient[i]);
    }
    let mut change = crate::numerics::Sum::default();
    change.add(route.objective.value_change(price, reference_price));
    for &index in &block.markets {
        let (market, (a, b)) = &route.markets[index];
        let v = [price[*a], price[*b]];
        let trade = market.evaluate(v);
        change.add(market.value_change(
            v,
            [reference_price[*a], reference_price[*b]],
            trade,
            reference.market_evaluations[index],
        ));
        for (local, token) in [*a, *b].into_iter().enumerate() {
            if let Ok(index) = block.tokens.binary_search(&token) {
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
    )
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
    let mut edges = Vec::with_capacity(route.markets.len());
    for (market, (a, b)) in &route.markets {
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
    let damping = 1e-8;
    let math_lower = route.objective.lower_bounds();
    let math_upper = route.objective.upper_bounds();
    let before_ratio = block_residual(block, price, current, &math_lower, &math_upper, options);
    let diagonal: Vec<_> = (0..route.tokens)
        .map(|i| {
            if movable[i] {
                model.diagonal[i] / normalization + damping
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
                    h / normalization + damping * x[i]
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

fn physical_price(
    x: &[f64],
    tokens: &[usize],
    origin: &Array1<f64>,
    scale: &Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
) -> Array1<f64> {
    let mut price = origin.clone();
    for (&x, &i) in x.iter().zip(tokens) {
        price[i] = x.mul_add(scale[i], origin[i]).max(lower[i]).min(upper[i]);
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
            let (gradient, change) =
                evaluate_block(&route, &block, &price, &reference_price, &reference);
            let (full, full_change) =
                route.evaluate_centered(&price, Some((&reference_price, &reference)));
            assert!((change - full_change).abs() < 1e-12);
            for (local, &i) in block.tokens.iter().enumerate() {
                assert_eq!(gradient[local], full.gradient[i]);
            }
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
