use crate::{
    Bound, Evaluation, Hessian, OptimizeResult, Options, Status, dot, norm, projected_gradient_norm,
};

/// Minimize a general differentiable objective in a box. The callback supplies
/// f and every gradient component. A reference, when present, is the current
/// accepted point and is fixed during a search (including the gradient retry).
/// A new reference is the last accepted trial. Model construction is separate
/// from FG callbacks and counted, as are all Hessian-vector products.
///
/// Only projected-gradient convergence is reported. Small steps, function ties,
/// exhausted CG/search budgets and a bad model are never convergence certificates.
pub fn minimize<F, M, H>(
    mut x: Vec<f64>,
    bounds: Vec<Bound>,
    options: Options,
    mut fg: F,
    mut model: M,
) -> anyhow::Result<OptimizeResult>
where
    F: FnMut(&[f64], &mut [f64], Option<&[f64]>) -> anyhow::Result<Evaluation>,
    M: FnMut(&[f64]) -> anyhow::Result<H>,
    H: Hessian,
{
    options.validate()?;
    anyhow::ensure!(
        !x.is_empty() && x.len() == bounds.len(),
        "invalid dimensions"
    );
    anyhow::ensure!(x.iter().all(|v| v.is_finite()), "nonfinite initial point");
    for (x, b) in x.iter_mut().zip(&bounds) {
        anyhow::ensure!(
            !b.lower.is_nan()
                && !b.upper.is_nan()
                && b.lower <= b.upper
                && b.lower != f64::INFINITY
                && b.upper != f64::NEG_INFINITY,
            "invalid bound"
        );
        *x = b.project(*x);
    }
    let mut gradient = vec![f64::NAN; x.len()];
    let first = fg(&x, &mut gradient, None)?;
    anyhow::ensure!(
        first.value.is_finite() && gradient.iter().all(|g| g.is_finite()),
        "nonfinite initial objective/gradient"
    );
    let initial_pg = projected_gradient_norm(&x, &gradient, &bounds);
    let mut result = OptimizeResult {
        x,
        f: first.value,
        gradient,
        projected_gradient_inf_norm: initial_pg,
        status: Status::ConvergedProjectedGradient,
        iterations: 0,
        evaluations: 1,
        hessian_evaluations: 0,
        hessian_products: 0,
        gradient_fallbacks: 0,
        curvature_truncations: 0,
    };
    let mut damping = options.initial_damping;
    let mut previous_change: Option<f64> = None;
    loop {
        if result.projected_gradient_inf_norm <= options.pgtol {
            result.status = Status::ConvergedProjectedGradient;
            break;
        }
        if result.iterations >= options.max_iterations {
            result.status = Status::MaxIterations;
            break;
        }
        if result.evaluations >= options.max_evaluations {
            result.status = Status::MaxEvaluations;
            break;
        }
        if result.hessian_products >= options.max_hessian_products {
            result.status = Status::MaxHessianProducts;
            break;
        }
        let hessian = model(&result.x)?;
        result.hessian_evaluations += 1;
        let n = result.x.len();
        let mut metric = vec![f64::NAN; n];
        hessian.diagonal(&mut metric)?;
        anyhow::ensure!(
            metric.iter().all(|v| v.is_finite()),
            "nonfinite Hessian diagonal"
        );
        let flat: Vec<_> = metric.iter().map(|d| *d == 0.0).collect();
        for d in &mut metric {
            *d = if *d == 0.0 {
                1.0
            } else {
                d.abs().max(f64::MIN_POSITIVE)
            };
        }
        let free: Vec<_> = result
            .x
            .iter()
            .zip(&result.gradient)
            .zip(&bounds)
            .map(|((&x, &g), b)| {
                b.lower != b.upper && !((x <= b.lower && g >= 0.0) || (x >= b.upper && g <= 0.0))
            })
            .collect();
        let eta = (result.projected_gradient_inf_norm / initial_pg)
            .sqrt()
            .min(options.cg_relative_tolerance)
            .max(1e-10);
        let cg = pcg(
            &hessian,
            &result.gradient,
            &metric,
            &free,
            damping,
            eta,
            &options,
            &mut result.hessian_products,
        )?;
        let Cg::Direction(mut direction, truncated) = cg else {
            result.status = Status::MaxHessianProducts;
            break;
        };
        if truncated {
            result.curvature_truncations += 1;
        }
        // Restrict the Newton direction to the feasible tangent cone. Coupling
        // can make this restriction non-descent; actual projected slopes and
        // the independently safeguarded gradient direction handle that case.
        for i in 0..n {
            if (result.x[i] <= bounds[i].lower && direction[i] < 0.0)
                || (result.x[i] >= bounds[i].upper && direction[i] > 0.0)
            {
                direction[i] = 0.0;
            }
        }
        let mut search = Search::Stagnation;
        if direction.iter().all(|v| v.is_finite()) && dot(&result.gradient, &direction) < 0.0 {
            let slope = dot(&result.gradient, &direction);
            let initial_alpha = previous_change.map_or(1.0, |change| {
                let estimate = 2.0 * (change / slope);
                if estimate.is_finite() && estimate > 0.0 {
                    estimate.min(1.0)
                } else {
                    1.0
                }
            });
            search = line_search_from(
                &result.x,
                result.f,
                &result.gradient,
                &direction,
                &bounds,
                &options,
                &mut result.evaluations,
                &mut fg,
                initial_alpha,
            )?;
        }
        let mut fallback = false;
        if !matches!(search, Search::Accepted(..) | Search::Budget) {
            fallback = true;
            result.gradient_fallbacks += 1;
            // Keep Jacobi scaling when Newton is unusable: raw steepest
            // descent can reintroduce precisely the conditioning PCG removed.
            // Flat model coordinates have no natural Newton length; use a unit
            // downhill direction there instead of an arbitrarily tiny gradient.
            let mut direction: Vec<_> = (0..n)
                .map(|i| {
                    if !free[i] || result.gradient[i] == 0.0 {
                        0.0
                    } else if flat[i] {
                        -result.gradient[i].signum()
                    } else {
                        -result.gradient[i] / metric[i]
                    }
                })
                .collect();
            if !direction.iter().all(|d| d.is_finite()) || norm(&direction) == 0.0 {
                let magnitude = result
                    .gradient
                    .iter()
                    .zip(&free)
                    .filter_map(|(&g, &f)| f.then_some(g.abs()))
                    .fold(0.0, f64::max);
                direction = result
                    .gradient
                    .iter()
                    .zip(&free)
                    .map(|(&g, &f)| if f { -g / magnitude } else { 0.0 })
                    .collect();
            }
            let magnitude = norm(&direction).max(1.0);
            for d in &mut direction {
                *d /= magnitude;
            }
            search = line_search(
                &result.x,
                result.f,
                &result.gradient,
                &direction,
                &bounds,
                &options,
                &mut result.evaluations,
                &mut fg,
            )?;
        }
        match search {
            Search::Accepted(x, value, gradient, _alpha, change) => {
                previous_change = Some(change);
                result.x = x;
                result.f = value;
                result.gradient = gradient;
                result.iterations += 1;
                result.projected_gradient_inf_norm =
                    projected_gradient_norm(&result.x, &result.gradient, &bounds);
                damping = if fallback || truncated {
                    (damping.max(1e-8) * 10.0).min(1e12)
                } else if damping < 1e-12 {
                    0.0
                } else {
                    damping * 0.2
                };
            }
            Search::Budget => {
                result.status = Status::MaxEvaluations;
                break;
            }
            Search::Stagnation => {
                result.status = Status::NumericalStagnation;
                break;
            }
            Search::Failed => {
                result.status = Status::LineSearchFailed;
                break;
            }
        }
    }
    Ok(result)
}

enum Cg {
    Direction(Vec<f64>, bool),
    Budget,
}
#[allow(clippy::too_many_arguments)]
fn pcg<H: Hessian>(
    h: &H,
    g: &[f64],
    metric: &[f64],
    free: &[bool],
    damping: f64,
    eta: f64,
    options: &Options,
    products: &mut usize,
) -> anyhow::Result<Cg> {
    let n = g.len();
    let mut x = vec![0.0; n];
    let mut r: Vec<_> = g
        .iter()
        .zip(free)
        .map(|(&g, &free)| if free { -g } else { 0.0 })
        .collect();
    let mut z: Vec<_> = r
        .iter()
        .zip(metric)
        .map(|(&r, &d)| (r / d) / (1.0 + damping))
        .collect();
    let mut direction = z.clone();
    let mut rz = dot(&r, &z);
    let initial = rz;
    if !rz.is_finite() || rz <= 0.0 {
        return Ok(Cg::Direction(x, true));
    }
    // Some extra iterations allow finite-precision loss of CG conjugacy, but
    // both local and global product budgets remain hard limits.
    for _ in 0..options.max_cg_iterations {
        if *products >= options.max_hessian_products {
            return Ok(Cg::Budget);
        }
        let magnitude = norm(&direction);
        if !direction.iter().all(|v| v.is_finite()) || magnitude == 0.0 {
            return Ok(Cg::Direction(x, true));
        }
        // H is linear: normalize its input, then compensate in the step length.
        // Null/near-null modes can grow CG vectors enormously even when the
        // current iterate and Hessian entries are finite. Never pass those
        // unscaled vectors to an otherwise well-behaved sparse operator.
        let unit: Vec<_> = direction.iter().map(|v| v / magnitude).collect();
        let mut product = vec![f64::NAN; n];
        h.apply(&unit, &mut product)?;
        *products += 1;
        anyhow::ensure!(
            product.iter().all(|v| v.is_finite()),
            "nonfinite Hessian product"
        );
        for i in 0..n {
            product[i] = if free[i] {
                product[i] + (damping * unit[i]) * metric[i]
            } else {
                0.0
            };
        }
        let curvature = dot(&unit, &product);
        let roundoff = 32.0
            * f64::EPSILON
            * unit
                .iter()
                .zip(&product)
                .map(|(a, b)| (a * b).abs())
                .sum::<f64>();
        if !curvature.is_finite() || curvature <= roundoff {
            if norm(&x) == 0.0 {
                x = z;
            }
            return Ok(Cg::Direction(x, true));
        }
        let alpha = (rz / magnitude) / curvature;
        if !alpha.is_finite() {
            return Ok(Cg::Direction(x, true));
        }
        let candidate: Vec<_> = x
            .iter()
            .zip(&unit)
            .map(|(&x, &p)| alpha.mul_add(p, x))
            .collect();
        let residual: Vec<_> = r
            .iter()
            .zip(&product)
            .map(|(&r, &a)| (-alpha).mul_add(a, r))
            .collect();
        if !candidate.iter().chain(&residual).all(|v| v.is_finite()) {
            return Ok(Cg::Direction(x, true));
        }
        x = candidate;
        r = residual;
        for i in 0..n {
            z[i] = (r[i] / metric[i]) / (1.0 + damping);
        }
        let next = dot(&r, &z);
        if !next.is_finite() || next < 0.0 {
            return Ok(Cg::Direction(x, true));
        }
        if next == 0.0 || next / initial <= eta * eta {
            return Ok(Cg::Direction(x, false));
        }
        let beta = next / rz;
        for i in 0..n {
            direction[i] = beta.mul_add(direction[i], z[i]);
        }
        rz = next;
    }
    Ok(Cg::Direction(x, false))
}

enum Search {
    Accepted(Vec<f64>, f64, Vec<f64>, f64, f64),
    Budget,
    Stagnation,
    Failed,
}
#[allow(clippy::too_many_arguments)]
fn line_search<F>(
    x: &[f64],
    value: f64,
    g: &[f64],
    direction: &[f64],
    bounds: &[Bound],
    options: &Options,
    evaluations: &mut usize,
    fg: &mut F,
) -> anyhow::Result<Search>
where
    F: FnMut(&[f64], &mut [f64], Option<&[f64]>) -> anyhow::Result<Evaluation>,
{
    line_search_from(
        x,
        value,
        g,
        direction,
        bounds,
        options,
        evaluations,
        fg,
        1.0,
    )
}
#[allow(clippy::too_many_arguments)]
fn line_search_from<F>(
    x: &[f64],
    value: f64,
    g: &[f64],
    direction: &[f64],
    bounds: &[Bound],
    options: &Options,
    evaluations: &mut usize,
    fg: &mut F,
    initial_alpha: f64,
) -> anyhow::Result<Search>
where
    F: FnMut(&[f64], &mut [f64], Option<&[f64]>) -> anyhow::Result<Evaluation>,
{
    let mut alpha = initial_alpha;
    let mut tried_full_step = initial_alpha == 1.0;
    let mut best = None;
    let project = |alpha: f64| {
        x.iter()
            .zip(direction)
            .zip(bounds)
            .map(|((&x, &d), b)| b.project(alpha.mul_add(d, x)))
            .collect::<Vec<_>>()
    };
    for _ in 0..options.max_line_search_steps {
        let mut contraction = options.backtrack;
        if *evaluations >= options.max_evaluations {
            return Ok(best.unwrap_or(Search::Budget));
        }
        let mut trial = project(alpha);
        if trial == x && !tried_full_step && best.is_none() {
            alpha = 1.0;
            tried_full_step = true;
            trial = project(alpha);
        }
        if trial == x {
            return Ok(Search::Stagnation);
        }
        let step: Vec<_> = trial.iter().zip(x).map(|(t, x)| t - x).collect();
        let slope = dot(g, &step);
        if trial.iter().all(|v| v.is_finite()) && slope.is_finite() && slope < 0.0 {
            let mut gradient = vec![f64::NAN; x.len()];
            let evaluation = fg(&trial, &mut gradient, Some(x))?;
            *evaluations += 1;
            let change = evaluation.difference.unwrap_or(evaluation.value - value);
            let finite = evaluation.value.is_finite()
                && change.is_finite()
                && gradient.iter().all(|v| v.is_finite());
            if finite && change <= options.armijo * slope {
                let tangent: Vec<_> = direction
                    .iter()
                    .zip(&trial)
                    .zip(bounds)
                    .map(|((&d, &t), b)| {
                        if (t <= b.lower && d < 0.0) || (t >= b.upper && d > 0.0) {
                            0.0
                        } else {
                            d
                        }
                    })
                    .collect();
                let short = dot(&gradient, &tangent) < 0.9 * dot(g, direction);
                let accepted = Search::Accepted(trial, evaluation.value, gradient, alpha, change);
                if alpha == 1.0 || !short {
                    return Ok(accepted);
                }
                best = Some(accepted);
                alpha = (alpha * 2.0).min(1.0);
                continue;
            }
            if let Some(accepted) = best {
                return Ok(accepted);
            }
            // A tiny warm start can round to a function tie while still being
            // downhill. It is not evidence that the full Newton step failed.
            if finite && change == 0.0 && !tried_full_step && dot(&gradient, direction) < 0.0 {
                alpha = 1.0;
                tried_full_step = true;
                continue;
            }
            let remainder = change - slope;
            let measured_curvature = 0.5 * (dot(&gradient, &step) - slope);
            let value_scale = if evaluation.difference.is_some() {
                change.abs()
            } else {
                evaluation.value.abs().max(value.abs())
            };
            let resolved = remainder > 64.0 * f64::EPSILON * value_scale.max(slope.abs());
            let agreement = remainder / measured_curvature;
            if finite && resolved && (0.5..=2.0).contains(&agreement) {
                // Minimize the quadratic through (0, f), its downhill slope,
                // and this rejected value. Use the actual projected slope;
                // projection need not be linear, so this is only a proposal.
                // Normalize by -slope before adding to avoid overflow in
                // 2*(change-slope). Every proposal still needs a fresh Armijo
                // check; nonfinite/domain-invalid trials use plain backtracking.
                let ratio = 0.5 / (1.0 + change / (-slope));
                if ratio.is_finite() && ratio >= 0.0 {
                    contraction = ratio.clamp(0.2 * options.backtrack, options.backtrack);
                }
            }
        }
        alpha *= contraction;
    }
    Ok(best.unwrap_or(Search::Failed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warm_starts_expand_but_keep_the_last_accepted_trial_on_domain_failure() {
        let mut calls = Vec::new();
        let mut evaluations = 0;
        let options = Options {
            max_evaluations: 2,
            ..Default::default()
        };
        let search = line_search_from(
            &[1.0],
            1.0,
            &[1.0],
            &[-1.0],
            &[Bound::unbounded()],
            &options,
            &mut evaluations,
            &mut |x, g, r| {
                assert_eq!(r, Some([1.0].as_slice()));
                calls.push(x[0]);
                g[0] = 1.0;
                Ok(Evaluation::value(if x[0] >= 0.85 {
                    x[0]
                } else {
                    f64::NAN
                }))
            },
            0.1,
        )
        .unwrap();
        let Search::Accepted(x, value, g, alpha, change) = search else {
            panic!("lost accepted trial");
        };
        assert_eq!(calls, vec![0.9, 0.8]);
        assert_eq!(evaluations, 2);
        assert_eq!(x, vec![0.9]);
        assert_eq!(value, 0.9);
        assert_eq!(g, vec![1.0]);
        assert_eq!(alpha, 0.1);
        assert!(change < 0.0);
    }

    #[test]
    fn warm_start_rounding_does_not_falsely_exhaust_the_direction() {
        let mut evaluations = 0;
        let options = Options {
            max_line_search_steps: 1,
            ..Default::default()
        };
        let search = line_search_from(
            &[1.0],
            0.5,
            &[1.0],
            &[-1e-8],
            &[Bound::unbounded()],
            &options,
            &mut evaluations,
            &mut |x, g, _| {
                g[0] = x[0];
                Ok(Evaluation::value(0.5 * x[0] * x[0]))
            },
            1e-20,
        )
        .unwrap();
        assert!(matches!(search, Search::Accepted(..)));
        assert_eq!(evaluations, 1);
        let mut evaluations = 0;
        let search = line_search_from(
            &[100.0],
            1e16 + 5000.0,
            &[100.0],
            &[-10.0],
            &[Bound::unbounded()],
            &Options::default(),
            &mut evaluations,
            &mut |x, g, _| {
                g[0] = x[0];
                Ok(Evaluation::value(1e16 + 0.5 * x[0] * x[0]))
            },
            1e-3,
        )
        .unwrap();
        let Search::Accepted(x, _, _, _, _) = search else {
            panic!("function tie blocked full step");
        };
        assert_eq!(x, vec![90.0]);
        assert_eq!(evaluations, 2);
    }

    #[test]
    fn expanded_search_rebases_on_accepted_not_last_evaluated_points() {
        struct Scalar(f64);
        impl Hessian for Scalar {
            fn apply(&self, v: &[f64], out: &mut [f64]) -> anyhow::Result<()> {
                out[0] = self.0 * v[0];
                Ok(())
            }
            fn diagonal(&self, out: &mut [f64]) -> anyhow::Result<()> {
                out[0] = self.0;
                Ok(())
            }
        }
        let mut saw_rejected_expansion = false;
        for b in [0.005, 0.01, 0.02] {
            let mut last = 0.1;
            let mut origin = 0.1;
            let result = minimize(
                vec![0.1],
                vec![Bound::unbounded()],
                Options::default(),
                |x, g, r| {
                    if let Some(r) = r {
                        if r[0] != origin && r[0] != last {
                            saw_rejected_expansion = true;
                        }
                        origin = r[0];
                    }
                    last = x[0];
                    let (f, gradient) = if x[0] >= 0.0 {
                        (0.5 * x[0] * x[0] + b * x[0], x[0] + b)
                    } else if x[0] >= -1.0 {
                        (0.5e-5 * x[0] * x[0] + b * x[0], 1e-5 * x[0] + b)
                    } else {
                        let d = x[0] + 1.0;
                        (
                            0.5e-5 - b + (b - 1e-5) * d + 5.0 * d * d,
                            b - 1e-5 + 10.0 * d,
                        )
                    };
                    g[0] = gradient;
                    Ok(Evaluation::value(f))
                },
                |x| {
                    Ok(Scalar(if x[0] >= 0.0 {
                        1.0
                    } else if x[0] >= -1.0 {
                        1e-5
                    } else {
                        10.0
                    }))
                },
            )
            .unwrap();
            assert_eq!(
                result.status,
                Status::ConvergedProjectedGradient,
                "{result:?}"
            );
            assert!((result.x[0] - (-1.0 - (b - 1e-5) / 10.0)).abs() < 1e-8);
        }
        assert!(saw_rejected_expansion);
    }

    #[test]
    fn interpolation_resolves_oversized_steps_across_objective_scales() {
        for scale in [1e-200, 1.0, 1e200] {
            let options = Options::default();
            let mut evaluations = 0;
            let search = line_search(
                &[1.0],
                0.5 * scale,
                &[scale],
                &[-1e18],
                &[Bound::unbounded()],
                &options,
                &mut evaluations,
                &mut |x, g, reference| {
                    assert_eq!(reference, Some([1.0].as_slice()));
                    g[0] = scale * x[0];
                    let step = x[0] - 1.0;
                    Ok(Evaluation::with_difference(
                        0.5 * scale * x[0] * x[0],
                        (0.5 * scale * step) * (x[0] + 1.0),
                    ))
                },
            )
            .unwrap();
            let Search::Accepted(x, value, gradient, _, _) = search else {
                panic!("no accepted step");
            };
            assert!(evaluations < 25, "FG={evaluations} scale={scale}");
            assert!(x[0].abs() < 1e-12);
            assert!(value <= 0.5 * scale);
            assert_eq!(gradient[0], scale * x[0]);
        }
    }

    #[test]
    fn projected_interpolation_checks_true_decrease_and_fixed_coordinates() {
        let mut evaluations = 0;
        let search = line_search(
            &[0.0, 7.0],
            0.005,
            &[-0.1, 100.0],
            &[1e18, 1e30],
            &[Bound::new(0.0, 1.0), Bound::new(7.0, 7.0)],
            &Options::default(),
            &mut evaluations,
            &mut |x, g, reference| {
                assert_eq!(reference, Some([0.0, 7.0].as_slice()));
                assert!((0.0..=1.0).contains(&x[0]));
                assert_eq!(x[1], 7.0);
                g[0] = x[0] - 0.1;
                g[1] = 100.0;
                Ok(Evaluation::with_difference(
                    0.5 * g[0] * g[0],
                    0.5 * x[0] * (x[0] - 0.2),
                ))
            },
        )
        .unwrap();
        let Search::Accepted(x, value, _, _, _) = search else {
            panic!("no accepted step");
        };
        assert!(evaluations < 30);
        let slope = -0.1 * x[0];
        assert!(value - 0.005 <= Options::default().armijo * slope);
    }

    #[test]
    fn unresolved_function_values_are_not_used_for_interpolation() {
        for accurate in [false, true] {
            let mut trials = Vec::new();
            let mut evaluations = 0;
            let options = Options {
                max_line_search_steps: 3,
                ..Default::default()
            };
            let search = line_search(
                &[1.0],
                1e100,
                &[0.05],
                &[-1.0],
                &[Bound::unbounded()],
                &options,
                &mut evaluations,
                &mut |x, g, _| {
                    trials.push(x[0]);
                    g[0] = x[0] - 0.95;
                    Ok(if accurate {
                        Evaluation::with_difference(
                            1e100,
                            (x[0] - 1.0) * (0.5 * (x[0] + 1.0) - 0.95),
                        )
                    } else {
                        Evaluation::value(1e100)
                    })
                },
            )
            .unwrap();
            assert_eq!(evaluations, 3);
            if accurate {
                let Search::Accepted(x, _, _, _, _) = search else {
                    panic!("no accepted step");
                };
                assert!((x[0] - 0.95).abs() < 1e-15);
            } else {
                assert!(matches!(search, Search::Failed));
                assert_eq!(trials, vec![0.0, 0.5, 0.75]);
            }
        }
    }

    #[test]
    fn invalid_or_inconsistent_trials_keep_geometric_backtracking() {
        for invalid in [false, true] {
            let mut trials = Vec::new();
            let mut evaluations = 0;
            let options = Options {
                max_line_search_steps: 3,
                ..Default::default()
            };
            let search = line_search(
                &[1.0],
                1.0,
                &[1.0],
                &[-1.0],
                &[Bound::unbounded()],
                &options,
                &mut evaluations,
                &mut |x, g, _| {
                    trials.push(x[0]);
                    g[0] = 1.0;
                    // Constant gradient disagrees with the alleged positive
                    // curvature. A domain-invalid value must not be fitted either.
                    Ok(Evaluation::with_difference(
                        if invalid { f64::NAN } else { 2.0 },
                        1.0,
                    ))
                },
            )
            .unwrap();
            assert!(matches!(search, Search::Failed));
            assert_eq!(trials, vec![0.0, 0.5, 0.75]);
            assert_eq!(evaluations, 3);
        }
    }

    #[test]
    fn interpolation_respects_callback_budget_and_propagates_errors() {
        let options = Options {
            max_evaluations: 2,
            ..Default::default()
        };
        let mut calls = 0;
        let mut evaluations = 0;
        let search = line_search(
            &[1.0],
            0.5,
            &[1.0],
            &[-1e18],
            &[Bound::unbounded()],
            &options,
            &mut evaluations,
            &mut |x, g, _| {
                calls += 1;
                g[0] = x[0];
                Ok(Evaluation::value(0.5 * x[0] * x[0]))
            },
        )
        .unwrap();
        assert!(matches!(search, Search::Budget));
        assert_eq!(evaluations, 2);
        assert_eq!(calls, 2);
        assert!(
            line_search(
                &[1.0],
                0.5,
                &[1.0],
                &[-1e18],
                &[Bound::unbounded()],
                &options,
                &mut 0,
                &mut |_, _, _| anyhow::bail!("oracle failed"),
            )
            .is_err()
        );
    }
}
