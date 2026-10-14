use num_traits::{Float, FromPrimitive};

use crate::{
    all_finite,
    bounds::{Bound, project, projected_gradient, projected_steepest_direction},
    cast,
    cauchy::generalized_cauchy_point,
    dot,
    line_search::{LineSearchFailure, LineSearchResult, strong_wolfe},
    memory::LbfgsMemory,
    norm_inf, sub,
    subspace::subspace_minimize,
};

#[derive(Debug, Clone)]
pub struct Options<T> {
    pub memory: usize,
    pub factr: T,
    pub pgtol: T,
    pub max_iterations: usize,
    pub max_evaluations: usize,
    pub max_line_search_steps: usize,
    pub wolfe_c1: T,
    pub wolfe_c2: T,
}

impl<T: Float + FromPrimitive> Default for Options<T> {
    fn default() -> Self {
        Self {
            memory: 17,
            factr: T::zero(),
            pgtol: cast(1e-5),
            max_iterations: 1000,
            max_evaluations: 10_000,
            max_line_search_steps: 20,
            wolfe_c1: cast(1e-4),
            wolfe_c2: cast(0.9),
        }
    }
}

impl<T: Float> Options<T> {
    /// Validate task-independent parameters without evaluating a problem.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.memory > 0, "memory must be positive");
        anyhow::ensure!(
            self.factr.is_finite() && self.factr >= T::zero(),
            "factr must be non-negative"
        );
        anyhow::ensure!(
            self.pgtol.is_finite() && self.pgtol >= T::zero(),
            "pgtol must be non-negative"
        );
        anyhow::ensure!(self.max_iterations > 0, "max_iterations must be positive");
        anyhow::ensure!(self.max_evaluations > 0, "max_evaluations must be positive");
        anyhow::ensure!(
            self.max_line_search_steps > 0,
            "max_line_search_steps must be positive"
        );
        anyhow::ensure!(
            self.wolfe_c1 > T::zero() && self.wolfe_c1 < self.wolfe_c2,
            "wolfe_c1 must satisfy 0 < c1 < c2"
        );
        anyhow::ensure!(self.wolfe_c2 < T::one(), "wolfe_c2 must be less than 1");
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    ConvergedProjectedGradient,
    ConvergedFunctionReduction,
    MaxIterations,
    MaxEvaluations,
    LineSearchFailed,
}

#[derive(Debug, Clone)]
pub struct OptimizeResult<T> {
    pub x: Vec<T>,
    pub f: T,
    pub gradient: Vec<T>,
    pub iterations: usize,
    pub evaluations: usize,
    pub status: Status,
    pub projected_gradient_inf_norm: T,
    pub line_search_failure: Option<LineSearchFailure>,
}

/// Minimize a box-constrained objective. Fixed coordinates are excluded from
/// search directions and BFGS history, but callbacks/results retain the full
/// original dimension and gradient. No extra callback is needed for expansion.
pub fn minimize<T, F>(
    mut x: Vec<T>,
    bounds: Vec<Bound<T>>,
    options: Options<T>,
    mut fg: F,
) -> anyhow::Result<OptimizeResult<T>>
where
    T: Float + FromPrimitive,
    F: FnMut(&[T], &mut [T]) -> anyhow::Result<T>,
{
    validate_problem(&x, &bounds, &options)?;
    project(&mut x, &bounds);
    let variables: Vec<_> = bounds
        .iter()
        .enumerate()
        .filter_map(|(i, bound)| (!bound.is_fixed()).then_some(i))
        .collect();
    let variable_bounds: Vec<_> = variables.iter().map(|&i| bounds[i]).collect();

    let mut gradient = vec![T::zero(); x.len()];
    let mut f = fg(&x, &mut gradient)?;
    anyhow::ensure!(f.is_finite(), "objective returned non-finite value");
    anyhow::ensure!(all_finite(&gradient), "gradient returned non-finite value");

    let mut evaluations = 1usize;
    let mut memory = LbfgsMemory::new(options.memory);
    let mut projected_gradient_inf_norm = norm_inf(&projected_gradient(&x, &gradient, &bounds));

    if projected_gradient_inf_norm <= options.pgtol {
        return Ok(OptimizeResult {
            x,
            f,
            gradient,
            iterations: 0,
            evaluations,
            status: Status::ConvergedProjectedGradient,
            projected_gradient_inf_norm,
            line_search_failure: None,
        });
    }

    for iteration in 0..options.max_iterations {
        if evaluations >= options.max_evaluations {
            return Ok(OptimizeResult {
                x,
                f,
                gradient,
                iterations: iteration,
                evaluations,
                status: Status::MaxEvaluations,
                projected_gradient_inf_norm,
                line_search_failure: None,
            });
        }

        let direction =
            reduced_search_direction(&x, &gradient, &variables, &variable_bounds, &memory);
        let mut direction = if is_descent_direction(&gradient, &direction) {
            direction
        } else {
            memory.clear();
            let mut fallback = projected_steepest_direction(&x, &gradient, &bounds);
            safeguard_steepest_direction(&x, &gradient, &bounds, &mut fallback);
            fallback
        };

        let normalize_model = memory.is_empty() && norm_inf(&direction) < T::epsilon().sqrt();
        safeguard_direction(&mut direction, &gradient, normalize_model);

        if norm_inf(&direction) == T::zero() {
            projected_gradient_inf_norm = norm_inf(&projected_gradient(&x, &gradient, &bounds));
            return Ok(OptimizeResult {
                x,
                f,
                gradient,
                iterations: iteration,
                evaluations,
                status: Status::LineSearchFailed,
                projected_gradient_inf_norm,
                line_search_failure: Some(LineSearchFailure::NonDescentDirection),
            });
        }

        if !is_descent_direction(&gradient, &direction) {
            direction = projected_steepest_direction(&x, &gradient, &bounds);
            safeguard_steepest_direction(&x, &gradient, &bounds, &mut direction);
        }
        if !is_descent_direction(&gradient, &direction) {
            projected_gradient_inf_norm = norm_inf(&projected_gradient(&x, &gradient, &bounds));
            return Ok(OptimizeResult {
                x,
                f,
                gradient,
                iterations: iteration,
                evaluations,
                status: Status::LineSearchFailed,
                projected_gradient_inf_norm,
                line_search_failure: Some(LineSearchFailure::NonDescentDirection),
            });
        }

        // As in the classic solver, keep the first displacement bounded when
        // there are unbounded coordinates and no curvature scale is known yet.
        let initial_step = if memory.is_empty()
            && bounds
                .iter()
                .any(|b| b.lower.is_none() || b.upper.is_none())
        {
            T::one() / norm_inf(&direction).max(T::one())
        } else {
            T::one()
        };
        let mut search = strong_wolfe(
            &x,
            f,
            &gradient,
            &direction,
            &bounds,
            &options,
            options.max_evaluations - evaluations,
            initial_step,
            &mut fg,
        )?;
        evaluations += search.evaluations;
        // Restart only if there was curvature history to discard. With no history,
        // repeating the same steepest search only wastes the evaluation budget.
        if search.step.is_none() && !search.evaluation_limit && !memory.is_empty() {
            let diagonal = if search.failure == Some(LineSearchFailure::NumericalStagnation) {
                rounded_diagonal_direction(&x, &gradient, &bounds, &variables, &memory)
            } else {
                None
            };
            memory.clear();
            let mut fallback =
                diagonal.unwrap_or_else(|| projected_steepest_direction(&x, &gradient, &bounds));
            safeguard_steepest_direction(&x, &gradient, &bounds, &mut fallback);
            search = strong_wolfe(
                &x,
                f,
                &gradient,
                &fallback,
                &bounds,
                &options,
                options.max_evaluations - evaluations,
                initial_step,
                &mut fg,
            )?;
            evaluations += search.evaluations;
        }
        let Some(line_search) = search.step else {
            return Ok(OptimizeResult {
                x,
                f,
                gradient,
                iterations: iteration,
                evaluations,
                status: if search.evaluation_limit {
                    Status::MaxEvaluations
                } else {
                    Status::LineSearchFailed
                },
                projected_gradient_inf_norm,
                line_search_failure: search.failure,
            });
        };

        let old_f = f;
        accept_step(
            &mut x,
            &mut f,
            &mut gradient,
            &mut memory,
            &variables,
            line_search,
        );
        projected_gradient_inf_norm = norm_inf(&projected_gradient(&x, &gradient, &bounds));

        if projected_gradient_inf_norm <= options.pgtol {
            return Ok(OptimizeResult {
                x,
                f,
                gradient,
                iterations: iteration + 1,
                evaluations,
                status: Status::ConvergedProjectedGradient,
                projected_gradient_inf_norm,
                line_search_failure: None,
            });
        }

        if function_reduction_converged(old_f, f, options.factr) {
            return Ok(OptimizeResult {
                x,
                f,
                gradient,
                iterations: iteration + 1,
                evaluations,
                status: Status::ConvergedFunctionReduction,
                projected_gradient_inf_norm,
                line_search_failure: None,
            });
        }
    }

    Ok(OptimizeResult {
        x,
        f,
        gradient,
        iterations: options.max_iterations,
        evaluations,
        status: Status::MaxIterations,
        projected_gradient_inf_norm,
        line_search_failure: None,
    })
}

fn validate_problem<T: Float>(
    x: &[T],
    bounds: &[Bound<T>],
    options: &Options<T>,
) -> anyhow::Result<()> {
    anyhow::ensure!(!x.is_empty(), "x0 must be non-empty");
    anyhow::ensure!(
        x.len() == bounds.len(),
        "x0 and bounds must have the same length"
    );
    anyhow::ensure!(all_finite(x), "x0 must be finite");
    for bound in bounds {
        bound.validate()?;
    }
    options.validate()
}

fn reduced_search_direction<T: Float>(
    x: &[T],
    gradient: &[T],
    variables: &[usize],
    bounds: &[Bound<T>],
    memory: &LbfgsMemory<T>,
) -> Vec<T> {
    if variables.len() == x.len() {
        return search_direction(x, gradient, bounds, memory);
    }
    let reduced_x: Vec<_> = variables.iter().map(|&i| x[i]).collect();
    let reduced_g: Vec<_> = variables.iter().map(|&i| gradient[i]).collect();
    let reduced = search_direction(&reduced_x, &reduced_g, bounds, memory);
    let mut direction = vec![T::zero(); x.len()];
    for (&i, &value) in variables.iter().zip(&reduced) {
        direction[i] = value;
    }
    direction
}

fn search_direction<T: Float>(
    x: &[T],
    gradient: &[T],
    bounds: &[Bound<T>],
    memory: &LbfgsMemory<T>,
) -> Vec<T> {
    if !memory.is_empty() && memory.theta().is_finite() && memory.theta() > T::zero() {
        // A feasible unconstrained minimizer also minimizes the box-constrained
        // positive-definite BFGS model. Use its two-loop inverse directly, before
        // building a Cauchy point or the forward Hessian representation.
        // Check the rounded displacement as well; actual-objective Wolfe tests
        // still decide whether the solver accepts this search direction.
        let target = sub(x, &memory.inverse_hessian_product(gradient));
        if all_finite(&target)
            && target
                .iter()
                .zip(bounds)
                .all(|(&value, bound)| bound.project(value) == value)
        {
            let direction = sub(&target, x);
            let slope = dot(gradient, &direction);
            if all_finite(&direction) && slope.is_finite() && slope < T::zero() {
                return direction;
            }
        }
    }
    let cauchy = generalized_cauchy_point(x, gradient, bounds, memory);
    let target = if memory.is_empty() {
        cauchy
    } else {
        subspace_minimize(x, gradient, &cauchy, bounds, memory)
    };
    let direction = sub(&target, x);
    let direction_norm = norm_inf(&direction);
    if memory.is_empty() && direction_norm > T::one() {
        let mut steepest = projected_steepest_direction(x, gradient, bounds);
        let gradient_norm = norm_inf(&steepest);
        if gradient_norm > T::one() {
            // The identity model can clip a high-gradient coordinate at its bound
            // while leaving an enormous free displacement. Unit displacement
            // limiting can suppress that coordinate to roundoff and follow an almost
            // orthogonal ray. Compare in scaled units so neither dot overflows.
            let slope = gradient
                .iter()
                .zip(&direction)
                .filter(|(_, d)| **d != T::zero())
                .fold(T::zero(), |sum, (&g, &d)| {
                    sum + (g / gradient_norm) * (d / direction_norm)
                });
            let roundoff = T::epsilon() * T::from(32.).unwrap();
            if slope.is_finite() && slope >= -roundoff {
                safeguard_steepest_direction(x, gradient, bounds, &mut steepest);
                return steepest;
            }
        }
    }
    if direction_norm > T::zero() {
        direction
    } else {
        let mut fallback = projected_steepest_direction(x, gradient, bounds);
        safeguard_steepest_direction(x, gradient, bounds, &mut fallback);
        fallback
    }
}

// Keep an already usable steepest ray at its natural scale. Unit normalization
// is needed for a stationary first point or an excessive restart displacement,
// not for every ordinary fallback: changing that scale also changes which
// probes fit into the finite line-search budget.
fn safeguard_steepest_direction<T: Float>(
    x: &[T],
    gradient: &[T],
    bounds: &[Bound<T>],
    direction: &mut [T],
) {
    let normalize = norm_inf(direction) > T::one()
        || x.iter()
            .zip(direction.iter())
            .zip(bounds)
            .all(|((&x, &d), bound)| bound.project(x + d) == x);
    safeguard_direction(direction, gradient, normalize);
}

// Near a floating-point lattice point, a steepest ray can move high-curvature
// coordinates by one harmful ULP before moving a weak-curvature coordinate at
// all. Before discarding a stalled model, retain its positive diagonal scaling
// and use the actual rounded displacement. This only proposes the one existing
// restart ray; it neither accepts an unchecked point nor spends extra FG calls.
fn rounded_diagonal_direction<T: Float>(
    x: &[T],
    gradient: &[T],
    bounds: &[Bound<T>],
    variables: &[usize],
    memory: &LbfgsMemory<T>,
) -> Option<Vec<T>> {
    let mut direction = vec![T::zero(); x.len()];
    for (j, &i) in variables.iter().enumerate() {
        if (gradient[i] > T::zero() && bounds[i].is_at_lower(x[i]))
            || (gradient[i] < T::zero() && bounds[i].is_at_upper(x[i]))
            || gradient[i] == T::zero()
        {
            continue;
        }
        let diagonal = memory.hessian_diagonal_at(j);
        if !diagonal.is_finite() || diagonal <= T::zero() {
            return None;
        }
        let target = x[i] - gradient[i] / diagonal;
        if !target.is_finite() {
            return None;
        }
        direction[i] = bounds[i].project(target) - x[i];
    }
    is_descent_direction(gradient, &direction).then_some(direction)
}

fn is_descent_direction<T: Float>(gradient: &[T], direction: &[T]) -> bool {
    let derivative = dot(gradient, direction);
    derivative.is_finite() && derivative < T::zero()
}

// Positive rescaling changes the line parameter, not the search ray. Without a
// curvature model, do not let a tiny gradient determine an unrepresentable first
// displacement. Division by the norm (not multiplication by its reciprocal)
// also works when that reciprocal would overflow. Every trial still needs the
// original feasibility and Wolfe checks.
fn safeguard_direction<T: Float>(direction: &mut [T], gradient: &[T], normalize: bool) {
    let norm = norm_inf(direction);
    if norm.is_finite()
        && norm > T::zero()
        && (normalize
            || !is_descent_direction(gradient, direction)
            // A finite initial dot can still leave no exponent headroom for
            // trial slopes. Reparameterize an excessive ray before zoom sees
            // infinite endpoint derivatives from otherwise finite callbacks.
            || (norm > T::one()
                && dot(gradient, direction).abs() > T::max_value().sqrt()))
    {
        for d in direction.iter_mut() {
            *d = *d / norm;
        }
        // A sum of finite g_i*d_i can still overflow after unit normalization.
        // This path has a steepest direction, so all nonzero terms are negative.
        let two = T::one() + T::one();
        while dot(gradient, direction) == T::neg_infinity() {
            for d in direction.iter_mut() {
                *d = *d / two;
            }
        }
    }
}

fn accept_step<T: Float>(
    x: &mut Vec<T>,
    f: &mut T,
    gradient: &mut Vec<T>,
    memory: &mut LbfgsMemory<T>,
    variables: &[usize],
    next: LineSearchResult<T>,
) {
    // In particular, fixed-coordinate gradient changes must not affect theta.
    let s = variables.iter().map(|&i| next.x[i] - x[i]).collect();
    let y = variables.iter().map(|&i| next.g[i] - gradient[i]).collect();
    memory.push(s, y);
    *x = next.x;
    *f = next.f;
    *gradient = next.g;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> LbfgsMemory<f64> {
        let mut memory = LbfgsMemory::new(3);
        // theta=5; one BFGS update gives B=[[1,2],[2,9]].
        assert!(memory.push(vec![1., 0.], vec![1., 2.]));
        memory
    }

    #[test]
    fn feasible_full_newton_direction_avoids_forward_hessian_reconstruction() {
        let memory = model();
        let direction = search_direction(&[0., 0.], &[1., 2.], &[Bound::both(-2., 2.); 2], &memory);
        assert_eq!(direction, [-1., 0.]);
        assert!(!memory.hessian_cache_is_initialized());
        // The independent dense model has its global minimum here.
        assert_eq!(
            1. * direction[0]
                + 2. * direction[1]
                + 0.5
                    * (direction[0].powi(2)
                        + 4. * direction[0] * direction[1]
                        + 9. * direction[1].powi(2)),
            -0.5
        );
    }

    #[test]
    fn steepest_fallback_still_safeguards_stationary_and_large_rays() {
        let mut memory = LbfgsMemory::new(1);
        assert!(memory.push(vec![1e-100_f64], vec![1.]));
        for gradient in [1e-100, 1e20] {
            let direction = search_direction(&[1.], &[gradient], &[Bound::unbounded()], &memory);
            assert_eq!(direction, [-1.]);
            assert!(is_descent_direction(&[gradient], &direction));
        }
    }

    #[test]
    fn usable_steepest_fallback_preserves_its_scale() {
        let mut memory = LbfgsMemory::new(1);
        assert!(memory.push(vec![1e-100_f64], vec![1.]));
        for gradient in [0.25, 1e-4] {
            for bound in [Bound::unbounded(), Bound::both(-2., 2.)] {
                // The model displacement rounds away, but the raw steepest
                // displacement is already finite, descending and representable.
                let direction = search_direction(&[1.], &[gradient], &[bound], &memory);
                assert_eq!(direction, [-gradient]);
            }
        }
    }

    #[test]
    fn invalid_model_scale_is_rejected_before_search() {
        let mut memory = LbfgsMemory::new(1);
        assert!(!memory.push(vec![1e-300_f64], vec![1e50]));
        assert_eq!(memory.theta(), 1.);
        assert!(memory.is_empty());
        assert!(!memory.hessian_cache_is_initialized());
        let direction = search_direction(&[1e-300], &[1e50], &[Bound::both(-1., 1.)], &memory);
        assert_eq!(direction, [-1.]);
        assert!(memory.hessian_cache_is_initialized());
    }

    fn check_empty_model_clipping<T: Float + std::fmt::Debug>(exponent: i32) {
        let c = |v| T::from(v).unwrap();
        let scale = c(2_f64.powi(exponent));
        for sign in [-T::one(), T::one()] {
            for swap in [false, true] {
                let mut x = vec![sign * c(0.5), sign * c(3.)];
                let mut gradient = vec![-sign * scale, sign * (scale * T::epsilon())];
                let mut bounds = vec![
                    if sign > T::zero() {
                        Bound::upper(c(0.875))
                    } else {
                        Bound::lower(c(-0.875))
                    },
                    Bound::unbounded(),
                ];
                if swap {
                    x.reverse();
                    gradient.reverse();
                    bounds.reverse();
                }
                let mut direction = search_direction(&x, &gradient, &bounds, &LbfgsMemory::new(1));
                assert!(is_descent_direction(&gradient, &direction), "{direction:?}");
                if swap {
                    direction.reverse();
                }
                assert!(sign * direction[0] > c(0.5), "{direction:?}");
                assert!(direction[1].abs() <= T::epsilon(), "{direction:?}");
            }
        }
    }

    #[test]
    fn empty_identity_model_must_not_erase_bounded_descent_f32() {
        check_empty_model_clipping::<f32>(60);
    }

    #[test]
    fn empty_identity_model_must_not_erase_bounded_descent_f64() {
        check_empty_model_clipping::<f64>(100);
    }

    #[test]
    fn finite_excessive_slopes_are_reparameterized_before_trials() {
        let mut d32 = [-2_f32.powi(30)];
        let g32 = [2_f32.powi(60)];
        assert!(dot(&g32, &d32).is_finite());
        safeguard_direction(&mut d32, &g32, false);
        assert_eq!(d32, [-1.]);
        let mut d64 = [-2_f64.powi(300)];
        let g64 = [2_f64.powi(600)];
        assert!(dot(&g64, &d64).is_finite());
        safeguard_direction(&mut d64, &g64, false);
        assert_eq!(d64, [-1.]);
    }

    fn check_diagonal<T: Float + std::fmt::Debug>() {
        let c = |v| T::from(v).unwrap();
        let mut memory = LbfgsMemory::new(3);
        for k in 0..12 {
            let s: Vec<_> = (0..5)
                .map(|i| c(((k + i * 3) % 7) as f64 - 3.) / c(4.))
                .collect();
            let sum = s.iter().copied().fold(T::zero(), |a, b| a + b);
            let y: Vec<_> = s
                .iter()
                .enumerate()
                .map(|(i, s)| c((i + 1) as f64) * *s + sum / c(16.))
                .collect();
            assert!(memory.push(s, y));
            for i in 0..5 {
                let mut basis = vec![T::zero(); 5];
                basis[i] = T::one();
                let reference = memory.hessian_product(&basis)[i];
                let diagonal = memory.hessian_diagonal_at(i);
                let error = (diagonal - reference).abs();
                assert!(
                    error <= c(64.) * T::epsilon() * reference.abs().max(T::one()),
                    "{diagonal:?} vs {reference:?}"
                );
            }
        }
    }

    #[test]
    fn cached_diagonal_tracks_hessian_products_across_history_eviction() {
        check_diagonal::<f32>();
        check_diagonal::<f64>();
    }

    #[test]
    fn diagonal_restart_uses_reduced_model_indices_and_keeps_fixed_coordinates() {
        let memory = model();
        let bounds = [
            Bound::both(-2., 2.),
            Bound::both(1., 1.),
            Bound::both(-2., 2.),
        ];
        let direction =
            rounded_diagonal_direction(&[0., 1., 0.], &[1., 1e200, 2.], &bounds, &[0, 2], &memory)
                .unwrap();
        assert!((direction[0] + 1.).abs() < 1e-12);
        assert_eq!(direction[1], 0.);
        assert!((direction[2] + 2. / 9.).abs() < 1e-12);
        assert!(
            rounded_diagonal_direction(
                &[-2., 1., -2.],
                &[1., 1e200, 2.],
                &bounds,
                &[0, 2],
                &memory
            )
            .is_none()
        );
    }

    #[test]
    fn ordinary_empty_identity_model_keeps_its_original_displacement() {
        let direction = search_direction(
            &[0.5, 0.],
            &[-1., 0.25],
            &[Bound::upper(0.875), Bound::unbounded()],
            &LbfgsMemory::new(1),
        );
        assert_eq!(direction, [0.375, -0.25]);
    }

    #[test]
    fn infeasible_full_newton_direction_keeps_cauchy_and_model_safeguards() {
        let memory = model();
        let bounds = [Bound::both(-0.25, 2.), Bound::both(-2., 2.)];
        let direction = search_direction(&[0., 0.], &[1., 2.], &bounds, &memory);
        assert!(memory.hessian_cache_is_initialized());
        assert!(
            direction
                .iter()
                .zip(bounds)
                .all(|(&x, bound)| bound.project(x) == x)
        );
        assert!(dot(&[1., 2.], &direction) < 0.);
    }
}

fn function_reduction_converged<T: Float>(old_f: T, new_f: T, factr: T) -> bool {
    let denom = old_f.abs().max(new_f.abs()).max(T::one());
    let relative = (old_f - new_f).abs() / denom;
    relative <= factr * T::epsilon()
}
