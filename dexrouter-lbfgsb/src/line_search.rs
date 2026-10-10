use num_traits::{Float, FromPrimitive};

use crate::{
    add_scaled,
    bounds::{Bound, feasible_step_max, project},
    dot,
    solver::Options,
};

#[derive(Debug, Clone)]
pub(crate) struct LineSearchResult<T> {
    pub(crate) x: Vec<T>,
    pub(crate) f: T,
    pub(crate) g: Vec<T>,
}

pub(crate) struct LineSearchOutcome<T> {
    pub(crate) step: Option<LineSearchResult<T>>,
    pub(crate) evaluations: usize,
    pub(crate) evaluation_limit: bool,
}

/// Strong Wolfe search, with Armijo acceptance at a finite feasible endpoint.
/// The trial limit covers both bracketing and zoom; rejected trials are counted too.
pub(crate) fn strong_wolfe<T, F>(
    x0: &[T],
    f0: T,
    g0: &[T],
    direction: &[T],
    bounds: &[Bound<T>],
    options: &Options<T>,
    evaluation_budget: usize,
    initial_step: T,
    fg: &mut F,
) -> anyhow::Result<LineSearchOutcome<T>>
where
    T: Float + FromPrimitive,
    F: FnMut(&[T], &mut [T]) -> anyhow::Result<T>,
{
    let mut outcome = LineSearchOutcome {
        step: None,
        evaluations: 0,
        evaluation_limit: false,
    };
    let derphi0 = dot(g0, direction);
    let step_max = feasible_step_max(x0, direction, bounds);
    if !derphi0.is_finite() || derphi0 >= T::zero() || step_max <= T::zero() {
        return Ok(outcome);
    }

    let mut previous_step = T::zero();
    let mut previous_phi = f0;
    let mut step = initial_step.min(step_max);
    let mut bracket: Option<ZoomBracket<T>> = None;
    for iteration in 0..options.max_line_search_steps {
        if outcome.evaluations >= evaluation_budget {
            outcome.evaluation_limit = true;
            return Ok(outcome);
        }
        if let Some(b) = &bracket {
            step = b.low_step / (T::one() + T::one()) + b.high_step / (T::one() + T::one());
            // Adjacent representable endpoints: never accept an unchecked "best" trial.
            if step == b.low_step || step == b.high_step {
                return Ok(outcome);
            }
        }
        let eval = evaluate(x0, direction, step, bounds, fg)?;
        outcome.evaluations += 1;
        let derphi = dot(&eval.g, direction);
        let armijo = eval.f <= f0 + options.wolfe_c1 * step * derphi0;
        // Test both Wolfe conditions before deciding how to tighten a bracket.
        // Near a nonzero minimum, a valid decrease can round to the same value.
        // A constrained minimum may still have a negative directional derivative.
        if armijo
            && (derphi.abs() <= -options.wolfe_c2 * derphi0
                || (step_max.is_finite() && step == step_max && derphi < T::zero()))
        {
            outcome.step = Some(eval);
            return Ok(outcome);
        }
        if let Some(b) = &mut bracket {
            if !armijo || eval.f >= b.low_phi {
                b.high_step = step;
            } else {
                if derphi * (b.high_step - b.low_step) >= T::zero() {
                    b.high_step = b.low_step;
                }
                b.low_step = step;
                b.low_phi = eval.f;
            }
        } else {
            if !armijo || (iteration > 0 && eval.f >= previous_phi) {
                bracket = Some(ZoomBracket {
                    low_step: previous_step,
                    low_phi: previous_phi,
                    high_step: step,
                });
            } else if derphi >= T::zero() {
                bracket = Some(ZoomBracket {
                    low_step: step,
                    low_phi: eval.f,
                    high_step: previous_step,
                });
            } else {
                previous_step = step;
                previous_phi = eval.f;
                let next = (step + step).min(step_max);
                if !next.is_finite() || next == step {
                    return Ok(outcome);
                }
                step = next;
            }
        }
    }
    outcome.evaluation_limit = outcome.evaluations >= evaluation_budget;
    Ok(outcome)
}

struct ZoomBracket<T> {
    low_step: T,
    low_phi: T,
    high_step: T,
}

fn evaluate<T, F>(
    x0: &[T],
    direction: &[T],
    step: T,
    bounds: &[Bound<T>],
    fg: &mut F,
) -> anyhow::Result<LineSearchResult<T>>
where
    T: Float + FromPrimitive,
    F: FnMut(&[T], &mut [T]) -> anyhow::Result<T>,
{
    let mut x = add_scaled(x0, step, direction);
    project(&mut x, bounds);
    anyhow::ensure!(
        x.iter().all(|value| value.is_finite()),
        "line search produced non-finite point"
    );
    let mut g = vec![T::zero(); x.len()];
    let f = fg(&x, &mut g)?;
    anyhow::ensure!(f.is_finite(), "objective returned non-finite value");
    anyhow::ensure!(
        g.iter().all(|value| value.is_finite()),
        "gradient returned non-finite value"
    );
    Ok(LineSearchResult { x, f, g })
}
