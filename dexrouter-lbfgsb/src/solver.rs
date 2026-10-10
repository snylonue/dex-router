use num_traits::{Float, FromPrimitive};

use crate::{
    all_finite,
    bounds::{Bound, project, projected_gradient, projected_steepest_direction},
    cast,
    cauchy::generalized_cauchy_point,
    dot,
    line_search::{LineSearchFailure, strong_wolfe},
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

        let direction = search_direction(&x, &gradient, &bounds, &memory);
        let mut direction = if descent_derivative(&gradient, &direction) < T::zero() {
            direction
        } else {
            memory.clear();
            projected_steepest_direction(&x, &gradient, &bounds)
        };

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

        if descent_derivative(&gradient, &direction) >= T::zero() {
            direction = projected_steepest_direction(&x, &gradient, &bounds);
        }
        if descent_derivative(&gradient, &direction) >= T::zero() {
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
        let initial_step = if iteration == 0
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
            memory.clear();
            let fallback = projected_steepest_direction(&x, &gradient, &bounds);
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
            line_search.x,
            line_search.f,
            line_search.g,
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

fn search_direction<T: Float>(
    x: &[T],
    gradient: &[T],
    bounds: &[Bound<T>],
    memory: &LbfgsMemory<T>,
) -> Vec<T> {
    let cauchy = generalized_cauchy_point(x, gradient, bounds, memory);
    let target = if memory.is_empty() {
        cauchy
    } else {
        subspace_minimize(x, gradient, &cauchy, bounds, memory)
    };
    let direction = sub(&target, x);
    if norm_inf(&direction) > T::zero() {
        direction
    } else {
        projected_steepest_direction(x, gradient, bounds)
    }
}

fn descent_derivative<T: Float>(gradient: &[T], direction: &[T]) -> T {
    dot(gradient, direction)
}

fn accept_step<T: Float>(
    x: &mut Vec<T>,
    f: &mut T,
    gradient: &mut Vec<T>,
    memory: &mut LbfgsMemory<T>,
    next_x: Vec<T>,
    next_f: T,
    next_gradient: Vec<T>,
) {
    let s = sub(&next_x, x);
    let y = sub(&next_gradient, gradient);
    memory.push(s, y);
    *x = next_x;
    *f = next_f;
    *gradient = next_gradient;
}

fn function_reduction_converged<T: Float>(old_f: T, new_f: T, factr: T) -> bool {
    let denom = old_f.abs().max(new_f.abs()).max(T::one());
    let relative = (old_f - new_f).abs() / denom;
    relative <= factr * T::epsilon()
}
