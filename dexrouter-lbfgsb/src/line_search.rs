use num_traits::{Float, FromPrimitive};

use crate::{
    add_scaled,
    bounds::{Bound, feasible_step_max, project},
    cast, dot,
    solver::Options,
};

/// Task-independent reason a verified line-search step could not be found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineSearchFailure {
    NonDescentDirection,
    NoFeasibleStep,
    NumericalStagnation,
    TrialLimit,
    EvaluationLimit,
}

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
    pub(crate) failure: Option<LineSearchFailure>,
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
        failure: None,
    };
    let derphi0 = dot(g0, direction);
    let step_max = feasible_step_max(x0, direction, bounds);
    if !derphi0.is_finite() || derphi0 >= T::zero() {
        outcome.failure = Some(LineSearchFailure::NonDescentDirection);
        return Ok(outcome);
    }
    if step_max <= T::zero() {
        outcome.failure = Some(LineSearchFailure::NoFeasibleStep);
        return Ok(outcome);
    }

    let mut previous_step = T::zero();
    let mut previous_phi = f0;
    let mut previous_derivative = derphi0;
    let mut step = initial_step.min(step_max);
    let mut bracket: Option<ZoomBracket<T>> = None;
    for iteration in 0..options.max_line_search_steps {
        if outcome.evaluations >= evaluation_budget {
            outcome.evaluation_limit = true;
            outcome.failure = Some(LineSearchFailure::EvaluationLimit);
            return Ok(outcome);
        }
        if let Some(b) = &bracket {
            step = b.trial_step();
            // Adjacent representable endpoints: never accept an unchecked "best" trial.
            if step == b.low_step || step == b.high_step {
                outcome.failure = Some(LineSearchFailure::NumericalStagnation);
                return Ok(outcome);
            }
        }
        let eval = evaluate(x0, direction, step, bounds, fg)?;
        outcome.evaluations += 1;
        if eval.x == x0 {
            outcome.failure = Some(LineSearchFailure::NumericalStagnation);
            return Ok(outcome);
        }
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
                b.high_phi = eval.f;
                b.high_derivative = derphi;
            } else {
                if derphi * (b.high_step - b.low_step) >= T::zero() {
                    b.high_step = b.low_step;
                    b.high_phi = b.low_phi;
                    b.high_derivative = b.low_derivative;
                }
                b.low_step = step;
                b.low_phi = eval.f;
                b.low_derivative = derphi;
            }
        } else {
            if !armijo || (iteration > 0 && eval.f >= previous_phi) {
                bracket = Some(ZoomBracket {
                    low_step: previous_step,
                    low_phi: previous_phi,
                    low_derivative: previous_derivative,
                    high_step: step,
                    high_phi: eval.f,
                    high_derivative: derphi,
                });
            } else if derphi >= T::zero() {
                bracket = Some(ZoomBracket {
                    low_step: step,
                    low_phi: eval.f,
                    low_derivative: derphi,
                    high_step: previous_step,
                    high_phi: previous_phi,
                    high_derivative: previous_derivative,
                });
            } else {
                previous_step = step;
                previous_phi = eval.f;
                previous_derivative = derphi;
                let next = (step + step).min(step_max);
                if !next.is_finite() || next == step {
                    outcome.failure = Some(LineSearchFailure::NumericalStagnation);
                    return Ok(outcome);
                }
                step = next;
            }
        }
    }
    outcome.evaluation_limit = outcome.evaluations >= evaluation_budget;
    outcome.failure = Some(if outcome.evaluation_limit {
        LineSearchFailure::EvaluationLimit
    } else {
        LineSearchFailure::TrialLimit
    });
    Ok(outcome)
}

struct ZoomBracket<T> {
    low_step: T,
    low_phi: T,
    low_derivative: T,
    high_step: T,
    high_phi: T,
    high_derivative: T,
}

impl<T: Float + FromPrimitive> ZoomBracket<T> {
    fn trial_step(&self) -> T {
        let width = self.high_step - self.low_step;
        let opposite_slopes = (self.low_derivative < T::zero()
            && self.high_derivative >= T::zero())
            || (self.low_derivative > T::zero() && self.high_derivative <= T::zero());
        // Use the quadratic through both function values and the low-end slope.
        // If it is unusable, interpolate the derivative when it brackets a root.
        let secant = -self.low_derivative / (self.high_derivative - self.low_derivative);
        let slope = self.low_derivative * width;
        let denominator = (T::one() + T::one()) * (self.high_phi - self.low_phi - slope);
        let quadratic = -slope / denominator;
        let interior = |t: T| t.is_finite() && t > T::zero() && t < T::one();
        let fraction = if denominator > T::zero() && interior(quadratic) {
            quadratic
        } else if opposite_slopes && interior(secant) {
            secant
        } else {
            cast(0.5)
        };
        // Guarantee bracket contraction, without the slow 1/2 reduction on
        // narrow quadratics. This also works when low_step > high_step.
        let fraction = fraction.max(cast(0.1)).min(cast(0.9));
        let trial = self.low_step + fraction * width;
        if trial.is_finite()
            && trial > self.low_step.min(self.high_step)
            && trial < self.low_step.max(self.high_step)
        {
            trial
        } else {
            self.low_step / (T::one() + T::one()) + self.high_step / (T::one() + T::one())
        }
    }
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
#[cfg(test)]
mod tests {
    use super::ZoomBracket;

    #[test]
    fn zoom_interpolates_quadratic_with_either_endpoint_order() {
        for (low, high) in [(0_f64, 1.), (1., 0.)] {
            let bracket = ZoomBracket {
                low_step: low,
                low_phi: (low - 0.3).powi(2),
                low_derivative: 2. * (low - 0.3),
                high_step: high,
                high_phi: (high - 0.3).powi(2),
                high_derivative: 2. * (high - 0.3),
            };
            assert!((bracket.trial_step() - 0.3).abs() < 1e-15);
        }
    }

    #[test]
    fn zoom_safeguards_narrow_quadratic_with_either_endpoint_order() {
        for (low, high) in [(0_f64, 1.), (1., 0.)] {
            let bracket = ZoomBracket {
                low_step: low,
                low_phi: (low - 1e-8).powi(2),
                low_derivative: 2. * (low - 1e-8),
                high_step: high,
                high_phi: (high - 1e-8).powi(2),
                high_derivative: 2. * (high - 1e-8),
            };
            assert!((bracket.trial_step() - 0.1).abs() < 1e-15);
        }
    }

    #[test]
    fn zoom_uses_secant_or_bisection_when_quadratic_is_unusable() {
        let bracket = ZoomBracket {
            low_step: 0_f64,
            low_phi: 0.,
            low_derivative: -0.6,
            high_step: 1.,
            high_phi: -1., // Negative interpolated curvature.
            high_derivative: 1.4,
        };
        assert!((bracket.trial_step() - 0.3).abs() < 1e-15);
        let overflow = ZoomBracket {
            low_phi: f64::MAX,
            high_phi: f64::MAX,
            low_derivative: f64::MAX,
            high_derivative: f64::MAX,
            ..bracket
        };
        assert_eq!(overflow.trial_step(), 0.5);
    }
}
