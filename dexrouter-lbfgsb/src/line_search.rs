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
    let mut last_bracket_width = T::infinity();
    let mut older_bracket_width = T::infinity();
    for iteration in 0..options.max_line_search_steps {
        if outcome.evaluations >= evaluation_budget {
            outcome.evaluation_limit = true;
            outcome.failure = Some(LineSearchFailure::EvaluationLimit);
            return Ok(outcome);
        }
        if let Some(b) = &bracket {
            let width = (b.high_step - b.low_step).abs();
            // Clipping interpolation to 10% of the interval does not guarantee
            // useful progress: it can repeatedly move the same endpoint while
            // the slope barely changes. Require contraction over two zoom
            // updates before allowing further interpolation at the same end.
            let interpolated = b.trial_step();
            let signed_width = b.high_step - b.low_step;
            let clipped = interpolated == b.low_step + cast::<T>(0.1) * signed_width
                || interpolated == b.low_step + cast::<T>(0.9) * signed_width;
            step = if clipped && width >= cast::<T>(0.66) * older_bracket_width {
                b.low_step / cast(2.) + b.high_step / cast(2.)
            } else {
                interpolated
            };
            older_bracket_width = last_bracket_width;
            last_bracket_width = width;
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
            // A rounded tie does not establish which side contains the minimum.
            // Retain the derivative bracket instead of discarding its root.
            if !armijo || eval.f > b.low_phi {
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
            if !armijo || (iteration > 0 && eval.f > previous_phi) {
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
                let mut next = (step + step).min(step_max);
                if let Some(predicted) = quadratic_extrapolation(
                    previous_step,
                    previous_phi,
                    previous_derivative,
                    step,
                    eval.f,
                    derphi,
                ) {
                    let predicted = predicted.min(step_max);
                    if (!next.is_finite() || predicted > next)
                        && x0
                            .iter()
                            .zip(direction)
                            .all(|(&x, &d)| (x + predicted * d).is_finite())
                    {
                        next = predicted;
                    }
                }
                previous_step = step;
                previous_phi = eval.f;
                previous_derivative = derphi;
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

// A decreasing slope need not reach Wolfe curvature within twenty doublings.
// Extrapolate only when curvature is resolved above roundoff and the measured
// function change agrees with the integral of a linearly varying derivative.
// This predicts a trial, never an acceptance; feasibility and Wolfe still apply.
fn quadratic_extrapolation<T: Float + FromPrimitive>(
    previous_step: T,
    previous_phi: T,
    previous_derivative: T,
    step: T,
    phi: T,
    derivative: T,
) -> Option<T> {
    let width = step - previous_step;
    let change = derivative - previous_derivative;
    let derivative_scale = previous_derivative.abs().max(derivative.abs());
    if !change.is_finite()
        || change <= cast::<T>(32.) * T::epsilon() * derivative_scale
        || width <= T::zero()
    {
        return None;
    }
    let actual = phi - previous_phi;
    let previous_linear = previous_derivative * width;
    let current_linear = derivative * width;
    let expected = previous_linear / cast(2.) + current_linear / cast(2.);
    let scale = actual
        .abs()
        .max(previous_linear.abs())
        .max(current_linear.abs());
    if !actual.is_finite()
        || !expected.is_finite()
        || !scale.is_finite()
        || (actual - expected).abs() > cast::<T>(8.) * T::epsilon() * scale
    {
        return None;
    }
    let predicted = step - derivative * (width / change);
    (predicted.is_finite() && predicted > step).then_some(predicted)
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
        // Interpolate opposite derivatives at the endpoint with the smaller
        // slope, avoiding cancellation in 1-t for a root very near an endpoint.
        if opposite_slopes {
            let scale = self.low_derivative.abs().max(self.high_derivative.abs());
            if scale.is_finite() && scale > T::zero() {
                let low = self.low_derivative.abs() / scale;
                let high = self.high_derivative.abs() / scale;
                let trial = if low <= high {
                    self.low_step + (low / (low + high)) * width
                } else {
                    self.high_step - (high / (low + high)) * width
                };
                if trial.is_finite()
                    && trial > self.low_step.min(self.high_step)
                    && trial < self.low_step.max(self.high_step)
                {
                    // Equal rounded values contain no useful curvature. For
                    // unequal values, only bypass the 10% safeguard when an
                    // independently fitted function-value quadratic agrees.
                    // Otherwise false-position steps can hug an endpoint.
                    let near_endpoint = (trial - self.low_step)
                        .abs()
                        .min((trial - self.high_step).abs())
                        < cast::<T>(0.1) * width.abs();
                    if self.high_phi == self.low_phi
                        || (near_endpoint && self.quadratic_agrees(trial))
                    {
                        return trial;
                    }
                }
            }
        }
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

    fn quadratic_agrees(&self, derivative_trial: T) -> bool {
        // Fit from the endpoint with the smaller derivative as well. Fitting
        // from the distant endpoint can round a tiny displacement to zero.
        let (anchor, phi, derivative, other, other_phi) =
            if self.low_derivative.abs() <= self.high_derivative.abs() {
                (
                    self.low_step,
                    self.low_phi,
                    self.low_derivative,
                    self.high_step,
                    self.high_phi,
                )
            } else {
                (
                    self.high_step,
                    self.high_phi,
                    self.high_derivative,
                    self.low_step,
                    self.low_phi,
                )
            };
        let width = other - anchor;
        let slope = derivative * width;
        let denominator = (T::one() + T::one()) * (other_phi - phi - slope);
        let fraction = -slope / denominator;
        let quadratic_trial = anchor + fraction * width;
        let distance = (derivative_trial - anchor)
            .abs()
            .min((derivative_trial - other).abs());
        denominator > T::zero()
            && fraction.is_finite()
            && fraction > T::zero()
            && fraction < T::one()
            && quadratic_trial.is_finite()
            && (quadratic_trial - derivative_trial).abs()
                <= cast::<T>(8.) * T::epsilon().sqrt() * distance
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
    use super::{ZoomBracket, quadratic_extrapolation, strong_wolfe};
    use crate::{Bound, Options};

    #[test]
    fn plateau_secant_preserves_tiny_steps_and_endpoint_order() {
        for (low, high) in [(0_f64, 1.), (1., 0.)] {
            let root = 1e-24;
            let bracket = ZoomBracket {
                low_step: low,
                high_step: high,
                low_phi: 1e100,
                high_phi: 1e100,
                low_derivative: 1e24 * (low - root),
                high_derivative: 1e24 * (high - root),
            };
            assert!((bracket.trial_step() - root).abs() <= 1e-38);
        }
    }

    #[test]
    fn rounded_plateau_with_negative_slopes_keeps_expanding() {
        let mut fg = |x: &[f64], g: &mut [f64]| {
            g[0] = 0.01 * (x[0] - 100.);
            Ok(1e20 + 0.005 * (x[0] - 100.).powi(2))
        };
        let options = Options::default();
        let outcome = strong_wolfe(
            &[0.],
            1e20,
            &[-1.],
            &[1.],
            &[Bound::both(-1000., 1000.)],
            &options,
            20,
            1.,
            &mut fg,
        )
        .unwrap();
        let step = outcome
            .step
            .expect("equal rounded values must not stop expansion");
        assert!(step.x[0] >= 10.);
        assert!(step.g[0].abs() <= options.wolfe_c2);
        assert!(step.f <= 1e20 + options.wolfe_c1 * step.x[0] * -1.);
    }

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
    fn zoom_preserves_model_agreed_narrow_quadratic_with_either_endpoint_order() {
        for (low, high) in [(0_f64, 1.), (1., 0.)] {
            let bracket = ZoomBracket {
                low_step: low,
                low_phi: (low - 1e-8).powi(2),
                low_derivative: 2. * (low - 1e-8),
                high_step: high,
                high_phi: (high - 1e-8).powi(2),
                high_derivative: 2. * (high - 1e-8),
            };
            assert!((bracket.trial_step() - 1e-8).abs() < 1e-15);
        }
    }

    #[test]
    fn extrapolation_requires_resolved_and_consistent_curvature() {
        assert_eq!(
            quadratic_extrapolation(0_f64, 0., -1., 1., -0.875, -0.75),
            Some(4.)
        );
        // The change in slope is not resolved above the roundoff safeguard.
        let eps = 1e-15_f64;
        assert!(
            quadratic_extrapolation(
                0.,
                0.,
                eps - 1.,
                1.,
                eps * (1_f64.exp() - 1.) - 1.,
                eps * 1_f64.exp() - 1.
            )
            .is_none()
        );
        // Observable but non-quadratic curvature must not trigger a huge jump.
        let eps = 1e-6_f64;
        assert!(
            quadratic_extrapolation(
                0.,
                0.,
                eps - 1.,
                1.,
                eps * (1_f64.exp() - 1.) - 1.,
                eps * 1_f64.exp() - 1.
            )
            .is_none()
        );
        assert!(quadratic_extrapolation(0., 0., -1., 1., -1., -1.).is_none());
        assert!(quadratic_extrapolation(0., 0., -1., 1., 0., -0.5).is_none());
        assert!(quadratic_extrapolation(0., 0., -1., 1., -2., -2.).is_none());
    }

    #[test]
    fn disagreeing_models_keep_the_endpoint_safeguard() {
        for reversed in [false, true] {
            // f(t) = 1e24*t^4/4 - t: derivative secant predicts 1e-24,
            // the function-value quadratic predicts 2e-24, true root is 1e-8.
            let mut bracket = ZoomBracket {
                low_step: 0_f64,
                low_phi: 0.,
                low_derivative: -1.,
                high_step: 1.,
                high_phi: 0.25e24 - 1.,
                high_derivative: 1e24 - 1.,
            };
            if reversed {
                std::mem::swap(&mut bracket.low_step, &mut bracket.high_step);
                std::mem::swap(&mut bracket.low_phi, &mut bracket.high_phi);
                std::mem::swap(&mut bracket.low_derivative, &mut bracket.high_derivative);
            }
            let trial = bracket.trial_step();
            assert!((0.1..=0.9).contains(&trial), "{trial}");
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
