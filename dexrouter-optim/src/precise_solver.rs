//! Recentered affine Newton refinement; all work shares the warm solve budget.
use super::*;
use crate::{market::PreciseMarket, precision::Price, utility::PreciseUtility};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreciseStatus {
    EconomicallyConverged,
    MaxIterations,
    MaxEvaluations,
    MaxHessianProducts,
    LineSearchFailed,
    NumericalStagnation,
}

/// A distinct result contract: rounding price to hi alone voids its certificate.
#[derive(Debug, Clone)]
pub struct PreciseSolveDiagnostics {
    pub price: Vec<Price>,
    pub evaluation: RouteEvaluation,
    /// Original f64 backend status and diagnostics, never relabeled as precise success.
    pub warm_start: SolveDiagnostics<dexrouter_newton::Status>,
    pub status: PreciseStatus,
    pub evaluations: usize,
    pub iterations: usize,
    pub hessian_products: usize,
    pub hessian_evaluations: usize,
    pub max_kkt_residual_ratio: f64,
    pub max_deficit_ratio: f64,
    /// Sum of accepted, stable original-objective differences after the warm solve.
    pub refinement_objective_change: f64,
}
impl PreciseSolveDiagnostics {
    pub fn is_usable(&self) -> bool {
        self.status == PreciseStatus::EconomicallyConverged
            && self.max_kkt_residual_ratio <= 1.0
            && self.max_deficit_ratio <= 1.0
    }
}

fn ratios<U: PreciseUtility, M: PreciseMarket>(
    route: &Route<U, M>,
    p: &[Price],
    e: &RouteEvaluation,
    o: &SolveOptions,
) -> (f64, f64) {
    let l = route.objective.lower_bounds();
    let u = route.objective.upper_bounds();
    let mut kkt = 0.0_f64;
    let mut deficit = 0.0_f64;
    for i in 0..p.len() {
        let g = if (p[i] <= l[i].into() && e.gradient[i] >= 0.0)
            || (p[i] >= u[i].into() && e.gradient[i] <= 0.0)
        {
            0.0
        } else {
            e.gradient[i]
        };
        let allowed =
            o.absolute_residual_value + o.relative_residual_tolerance * e.turnover[i] * p[i].hi();
        kkt = kkt.max(g.abs() * p[i].hi() / allowed);
        deficit = deficit.max((-g * p[i].hi()).max(0.0) / allowed);
    }
    if !e.cost.is_finite()
        || e.gradient
            .iter()
            .chain(e.turnover.iter())
            .any(|g| !g.is_finite())
        || p.iter()
            .enumerate()
            .any(|(i, &p)| !p.is_finite() || p < l[i].into() || p > u[i].into())
    {
        (f64::INFINITY, f64::INFINITY)
    } else {
        (kkt, deficit)
    }
}
fn model<U: PreciseUtility, M: PreciseMarket>(
    route: &Route<U, M>,
    p: &[Price],
) -> anyhow::Result<RelativeCurvature> {
    let h = route.objective.precise_diagonal_hessian(p);
    anyhow::ensure!(h.len() == p.len(), "precise utility Hessian dimensions");
    let utility_diagonal: Vec<_> = h.iter().zip(p).map(|(&h, p)| h * p.hi() * p.hi()).collect();
    let mut diagonal = utility_diagonal.clone();
    let mut edges = Vec::new();
    for (m, (a, b)) in &route.markets {
        let w = m
            .relative_curvature_precise([p[*a], p[*b]])
            .ok_or_else(|| anyhow::anyhow!("precise curvature unavailable"))?;
        anyhow::ensure!(w.is_finite() && w >= 0.0, "invalid precise curvature");
        if a != b && w > 0.0 {
            diagonal[*a] += w;
            diagonal[*b] += w;
            edges.push((*a, *b, w));
        }
    }
    anyhow::ensure!(
        diagonal
            .iter()
            .chain(utility_diagonal.iter())
            .all(|d| d.is_finite() && *d >= 0.0),
        "invalid precise diagonal"
    );
    Ok(RelativeCurvature {
        utility_diagonal,
        diagonal,
        edges,
    })
}

/// Original-objective Newton with a twofold physical-price result. Uses the same
/// global FG, iteration and HVP limits as the f64 warm solve. No uphill polishing.
/// Custom markets/utilities must explicitly implement precise-price evaluation.
/// Refinement always attempts Schur; newton_schur_precondition controls the warm stage.
pub fn solve_price_with_precise_newton<U: PreciseUtility, M: PreciseMarket>(
    route: &Route<U, M>,
    initial: Array1<f64>,
    options: SolveOptions,
    newton: dexrouter_newton::Options,
) -> anyhow::Result<PreciseSolveDiagnostics> {
    newton.validate()?;
    options.optimizer.validate()?;
    anyhow::ensure!(
        options.stable_differences && !options.sqrt_prices,
        "precise Newton requires affine prices and stable differences"
    );
    let max_e = options
        .optimizer
        .max_evaluations
        .min(newton.max_evaluations);
    let max_i = options.optimizer.max_iterations.min(newton.max_iterations);
    anyhow::ensure!(
        max_e >= 2,
        "precise Newton needs warm evaluation and independent verification"
    );
    // Do not spend the entire shared budget searching an inadequate f64 lattice.
    // Only the warm WORK cap changes; original economic tolerances are untouched.
    let warm_cap = ((max_e - 1) / 2).clamp(1, 2000);
    let mut warm_options = options.clone().with_warm_start_budget();
    warm_options.optimizer.max_evaluations = warm_cap;
    let mut warm_newton = newton.clone();
    warm_newton.max_evaluations = warm_cap;
    let warm =
        solve_impl(route, initial, warm_options, Some(warm_newton))?.map_status(|s| match s {
            BackendStatus::Newton(s) => s,
            _ => unreachable!(),
        });
    let mut p: Vec<Price> = warm.price.iter().map(|&v| v.into()).collect();
    let mut current = route.evaluate_precise(&p);
    let mut evaluations = warm.evaluations + 1;
    let mut iterations = warm.iterations;
    let mut hvp = warm.hessian_products;
    let mut models = warm.hessian_evaluations;
    let lower = warm.numerical_lower_bounds.clone();
    let mut upper = route.objective.upper_bounds();
    if options.use_utility_anchor
        && let Some((i, a)) = route.objective.price_anchor()
    {
        upper[i] = a;
    }
    let tokens: Vec<_> = (0..p.len()).filter(|&i| lower[i] != upper[i]).collect();
    let objective_scale = warm.objective_scale;
    let mut total_change = crate::numerics::Sum::default();
    let status;
    loop {
        let r = ratios(route, &p, &current, &options);
        if r.0 <= 1.0 && r.1 <= 1.0 {
            status = PreciseStatus::EconomicallyConverged;
            break;
        }
        if iterations >= max_i {
            status = PreciseStatus::MaxIterations;
            break;
        }
        if evaluations + 3 > max_e {
            status = PreciseStatus::MaxEvaluations;
            break;
        }
        if hvp >= newton.max_hessian_products {
            status = PreciseStatus::MaxHessianProducts;
            break;
        }
        let scales: Vec<_> = tokens.iter().map(|&i| p[i].hi()).collect();
        let physical = |x: &[f64]| {
            let mut trial = p.clone();
            for (k, &i) in tokens.iter().enumerate() {
                let v = p[i] + Price::from(x[k]) * scales[k].into();
                trial[i] = if v < lower[i].into() {
                    lower[i].into()
                } else if v > upper[i].into() {
                    upper[i].into()
                } else {
                    v
                };
            }
            trial
        };
        let bounds: Vec<_> = tokens
            .iter()
            .enumerate()
            .map(|(k, &i)| {
                dexrouter_newton::Bound::new(
                    ((Price::from(lower[i]) - p[i]) / scales[k].into()).rounded(),
                    if upper[i].is_finite() {
                        ((Price::from(upper[i]) - p[i]) / scales[k].into()).rounded()
                    } else {
                        f64::INFINITY
                    },
                )
            })
            .collect();
        let mut n = newton.clone();
        n.max_iterations = 1;
        n.max_evaluations = max_e - evaluations - 1;
        n.max_hessian_products -= hvp;
        n.pgtol = 0.0;
        let solved = dexrouter_newton::minimize(
            vec![0.0; tokens.len()],
            bounds,
            n,
            |x, g, reference| {
                // One accepted step per phase: every search reference is exactly x=0.
                debug_assert!(reference.is_none_or(|r| r.iter().all(|x| *x == 0.0)));
                let trial = physical(x);
                let e = route.evaluate_precise(&trial);
                let change = route.precise_value_change(&trial, &p, &e, &current);
                for (k, &i) in tokens.iter().enumerate() {
                    g[k] = e.gradient[i] * scales[k] / objective_scale;
                }
                Ok(dexrouter_newton::Evaluation::with_difference(
                    change / objective_scale,
                    change / objective_scale,
                ))
            },
            |x| {
                let point = physical(x);
                Ok(BlockHessian {
                    curvature: model(route, &point)?,
                    tokens: tokens.clone(),
                    factors: tokens
                        .iter()
                        .enumerate()
                        .map(|(k, &i)| scales[k] / point[i].hi())
                        .collect(),
                    objective_scale,
                    tree_precondition: options.newton_tree_precondition,
                    // The precision stage must not reintroduce a poorly solved
                    // stiff direction. Keep the warm-stage metric independently selectable.
                    schur_precondition: true,
                })
            },
        )?;
        evaluations += solved.evaluations;
        iterations += solved.iterations;
        hvp += solved.hessian_products;
        models += solved.hessian_evaluations;
        if solved.x.iter().all(|x| *x == 0.0) {
            status = match solved.status {
                dexrouter_newton::Status::MaxEvaluations => PreciseStatus::MaxEvaluations,
                dexrouter_newton::Status::MaxHessianProducts => PreciseStatus::MaxHessianProducts,
                dexrouter_newton::Status::LineSearchFailed => PreciseStatus::LineSearchFailed,
                _ => PreciseStatus::NumericalStagnation,
            };
            break;
        }
        let candidate = physical(&solved.x);
        let evaluation = route.evaluate_precise(&candidate);
        evaluations += 1;
        let change = route.precise_value_change(&candidate, &p, &evaluation, &current);
        let mut slope = crate::numerics::Sum::default();
        for &i in &tokens {
            slope.add(current.gradient[i] * (candidate[i] - p[i]).rounded());
        }
        let slope = slope.total();
        if !change.is_finite()
            || !slope.is_finite()
            || slope >= 0.0
            || change > newton.armijo * slope
        {
            status = PreciseStatus::LineSearchFailed;
            break;
        }
        total_change.add(change);
        p = candidate;
        current = evaluation;
    }
    let r = ratios(route, &p, &current, &options);
    Ok(PreciseSolveDiagnostics {
        price: p,
        evaluation: current,
        warm_start: warm,
        status,
        evaluations,
        iterations,
        hessian_products: hvp,
        hessian_evaluations: models,
        max_kkt_residual_ratio: r.0,
        max_deficit_ratio: r.1,
        refinement_objective_change: total_change.total(),
    })
}
