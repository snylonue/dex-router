#[path = "cubic.rs"]
mod cubic;

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

    let anchor = direction
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .expect("a finite descent ray has a nonzero coordinate")
        .0;
    let mut previous_step = T::zero();
    let mut previous_phi = f0;
    let mut previous_derivative = derphi0;
    let mut step = initial_step.min(step_max);
    let mut bracket: Option<ZoomBracket<T>> = None;
    let mut last_negative_trial: Option<(T, T, T)> = None;
    let mut last_positive_trial: Option<(T, T, T)> = None;
    let mut last_negative_prediction: Option<(T, T)> = None;
    let mut last_positive_prediction: Option<(T, T)> = None;
    let mut local_prediction: Option<T> = None;
    let mut last_bracket_width = T::infinity();
    let mut older_bracket_width = T::infinity();
    let mut last_low_derivative = T::infinity();
    let mut older_low_derivative = T::infinity();
    for iteration in 0..options.max_line_search_steps {
        if outcome.evaluations >= evaluation_budget {
            outcome.evaluation_limit = true;
            outcome.failure = Some(LineSearchFailure::EvaluationLimit);
            return Ok(outcome);
        }
        if let Some(b) = &bracket {
            if b.only_rejected_lattice_points(x0, direction, bounds, f0, derphi0, options.wolfe_c2)
            {
                outcome.failure = Some(LineSearchFailure::NumericalStagnation);
                return Ok(outcome);
            }
            let width = (b.high_step - b.low_step).abs();
            // Clipping interpolation to 10% of the interval does not guarantee
            // useful progress: it can repeatedly move the same endpoint while
            // the slope barely changes. Require contraction over two zoom
            // updates before allowing further interpolation at the same end.
            step = b.safeguarded_trial_step(older_bracket_width, older_low_derivative);
            if older_low_derivative.is_finite()
                && b.low_derivative.abs() >= cast::<T>(0.66) * older_low_derivative
                && let Some(local) = local_prediction
                && local > b.low_step.min(b.high_step)
                && local < b.low_step.max(b.high_step)
            {
                step = local;
            }
            older_bracket_width = last_bracket_width;
            last_bracket_width = width;
            older_low_derivative = last_low_derivative;
            last_low_derivative = b.low_derivative.abs();
            // Adjacent representable endpoints: never accept an unchecked "best" trial.
            if step == b.low_step || step == b.high_step {
                outcome.failure = Some(LineSearchFailure::NumericalStagnation);
                return Ok(outcome);
            }
        }
        if let Some(b) = &bracket {
            // An interpolation parameter can be distinct from both endpoints
            // while its rounded point is still x0. Probe farther along the same
            // ray before spending a callback; this is not a stagnating bracket.
            let upper = b.low_step.max(b.high_step);
            while x0
                .iter()
                .zip(direction)
                .zip(bounds)
                .all(|((&x, &d), bound)| bound.project(x + step * d) == x)
            {
                let doubled = step + step;
                let next = if doubled.is_finite() && doubled < upper {
                    doubled
                } else {
                    step / cast(2.) + upper / cast(2.)
                };
                if next == step || next == upper {
                    outcome.failure = Some(LineSearchFailure::NumericalStagnation);
                    return Ok(outcome);
                }
                step = next;
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
        let effective_step = (eval.x[anchor] - x0[anchor]) / direction[anchor];
        let (history, prediction_history) = if derphi > T::zero() {
            (&mut last_positive_trial, &mut last_positive_prediction)
        } else {
            (&mut last_negative_trial, &mut last_negative_prediction)
        };
        let prediction = history
            .and_then(|(a, f, g)| quadratic_stationary(a, f, g, effective_step, eval.f, derphi));
        local_prediction = match (prediction, *prediction_history) {
            (Some((root, error)), Some((older_root, older_error)))
                if (error + older_error).is_finite()
                    && (root - older_root).abs() <= error + older_error =>
            {
                Some(root)
            }
            _ => None,
        };
        *prediction_history = prediction;
        *history = Some((effective_step, eval.f, derphi));
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
                )
                .or_else(|| {
                    // Three same-side samples must corroborate this distant root.
                    // The two-point fast path can lose its integral test to function
                    // roundoff when curvature is small but still well resolved.
                    local_prediction.filter(|&root| {
                        derphi.abs() >= cast::<T>(0.66) * derphi0.abs() && root > step
                    })
                }) {
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

// Rejected trials on one side of the stationary point may share a quadratic
// segment even when the bracket crosses an activation wall. Check the integral
// of the measured slopes, including resolved function-value roundoff. The caller
// also requires overlapping root-error estimates from three same-side samples.
// This predicts a trial only: neither acceptance nor a stopping rule changes.
fn quadratic_stationary<T: Float + FromPrimitive>(
    previous_step: T,
    previous_phi: T,
    previous_derivative: T,
    step: T,
    phi: T,
    derivative: T,
) -> Option<(T, T)> {
    let width = step - previous_step;
    let change = derivative - previous_derivative;
    let derivative_scale = previous_derivative.abs().max(derivative.abs());
    if !change.is_finite()
        || change.abs() <= cast::<T>(32.) * T::epsilon() * derivative_scale
        || width == T::zero()
        || (change > T::zero()) != (width > T::zero())
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
    let value_roundoff = cast::<T>(4.) * T::epsilon() * phi.abs().max(previous_phi.abs());
    if !value_roundoff.is_finite()
        || value_roundoff > T::epsilon().sqrt() * scale
        || !actual.is_finite()
        || !expected.is_finite()
        || !scale.is_finite()
        || (actual - expected).abs() > cast::<T>(8.) * T::epsilon() * scale + value_roundoff
    {
        return None;
    }
    let predicted = if previous_derivative.abs() < derivative.abs() {
        previous_step - (previous_derivative / change) * width
    } else {
        step - (derivative / change) * width
    };
    let anchor = if previous_derivative.abs() < derivative.abs() {
        previous_step
    } else {
        step
    };
    let relative_error = cast::<T>(64.)
        * T::epsilon()
        * (derivative_scale / change.abs())
        * (T::one()
            + cast::<T>(2.) * (previous_derivative.abs().min(derivative.abs()) / change.abs()));
    let error = cast::<T>(64.) * T::epsilon() * anchor.abs() + relative_error * width.abs();
    (predicted.is_finite() && error.is_finite()).then_some((predicted, error))
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
    // Monotonic rounded ray coordinates leave no third point when the endpoint
    // vectors are identical or differ in just one adjacent coordinate. Only
    // terminate if both known points are unusable at EVERY interior parameter:
    // x0, a value above f0, or a slope that fails the original curvature test.
    // Equal points alone are insufficient: Armijo still depends on the parameter.
    fn only_rejected_lattice_points(
        &self,
        x0: &[T],
        direction: &[T],
        bounds: &[Bound<T>],
        f0: T,
        derphi0: T,
        c2: T,
    ) -> bool {
        let mut changes = 0;
        let mut low_is_origin = true;
        let mut high_is_origin = true;
        for ((&x, &d), b) in x0.iter().zip(direction).zip(bounds) {
            let low = b.project(x + self.low_step * d);
            let high = b.project(x + self.high_step * d);
            if !low.is_finite() || !high.is_finite() {
                return false;
            }
            low_is_origin &= low == x;
            high_is_origin &= high == x;
            if low != high {
                changes += 1;
                if changes > 1 {
                    return false;
                }
                let middle = low + (high - low) / cast(2.);
                if middle != low && middle != high {
                    return false;
                }
            }
        }
        let curvature = -c2 * derphi0;
        (low_is_origin || self.low_phi > f0 || self.low_derivative.abs() > curvature)
            && (high_is_origin || self.high_phi > f0 || self.high_derivative.abs() > curvature)
    }

    fn safeguarded_trial_step(&self, older_bracket_width: T, older_low_derivative: T) -> T {
        let width = (self.high_step - self.low_step).abs();
        let interpolated = self.trial_step();
        let signed_width = self.high_step - self.low_step;
        let clipped = interpolated == self.low_step + cast::<T>(0.1) * signed_width
            || interpolated == self.low_step + cast::<T>(0.9) * signed_width;
        // Model agreement is approximate, especially in f32. It must not
        // exempt an endpoint-hugging secant forever. Preserve accurate
        // narrow-quadratic trials while their slopes materially improve;
        // otherwise demand the same two-update interval contraction.
        let near_endpoint = (interpolated - self.low_step)
            .abs()
            .min((interpolated - self.high_step).abs())
            < cast::<T>(0.1) * width;
        let stalled_slope =
            near_endpoint && self.low_derivative.abs() >= cast::<T>(0.66) * older_low_derivative;
        if (clipped || stalled_slope) && width >= cast::<T>(0.66) * older_bracket_width {
            self.low_step / cast(2.) + self.high_step / cast(2.)
        } else {
            interpolated
        }
    }

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
        // A heavily clipped quadratic can repeatedly miss a narrow activation
        // region. Use both endpoint slopes in a safeguarded cubic proposal.
        // Ordinary interior and independently verified tiny-root predictions
        // above retain their original behavior. This never accepts a trial.
        if (fraction < cast::<T>(0.1) || fraction > cast::<T>(0.9))
            && let Some(cubic) = cubic::zoom(self)
        {
            return cubic;
        }
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
    use super::{ZoomBracket, quadratic_extrapolation, quadratic_stationary, strong_wolfe};
    use crate::{Bound, Options};

    #[test]
    fn local_curvature_requires_resolved_function_values_and_positive_curvature() {
        assert!(quadratic_stationary(0_f64, 1e24, -2., 1., 1e24, -1.).is_none());
        assert!(quadratic_stationary(0_f64, 0., -1., 1., -1.5, -2.).is_none());
        assert!(quadratic_stationary(0_f64, 0., -1., 1., -1., -1.).is_none());
    }

    #[test]
    fn local_curvature_uses_either_step_order_and_avoids_underflow_in_ratios() {
        for reverse in [false, true] {
            // Width/change would underflow in f32, although derivative/change
            // and the stationary point are representable.
            let a = 1e-12_f32;
            let b = 2e-12_f32;
            let h = 1e26_f32;
            let point = |x: f32| {
                let d = (x - 0.5e-12) * 1e12;
                (h * d * d / 2., h * d * 1e12)
            };
            let (fa, ga) = point(a);
            let (fb, gb) = point(b);
            assert_eq!(
                (b - a) / (gb - ga),
                0.,
                "the old intermediate really underflows"
            );
            let fit = if reverse {
                quadratic_stationary(b, fb, gb, a, fa, ga)
            } else {
                quadratic_stationary(a, fa, ga, b, fb, gb)
            };
            let (root, error) = fit.expect("resolved quadratic");
            assert!((root - 0.5e-12).abs() <= error);
            assert!((root - 0.5e-12).abs() < 1e-18);
        }
    }

    #[test]
    fn lattice_stagnation_requires_both_points_to_be_unusable_at_every_parameter() {
        use num_traits::{Float, FromPrimitive};
        fn check<T: Float + FromPrimitive>() {
            let c = |x| T::from(x).unwrap();
            let mut b = ZoomBracket {
                low_step: T::zero(),
                low_phi: T::one(),
                low_derivative: -T::one(),
                high_step: T::one(),
                high_phi: c(2.),
                high_derivative: T::one(),
            };
            for (x, d) in [
                (T::one(), T::epsilon()),
                (-T::one(), -T::epsilon()),
                (T::zero(), T::min_positive_value() * T::epsilon()),
            ] {
                assert!(b.only_rejected_lattice_points(
                    &[x],
                    &[d],
                    &[Bound::unbounded()],
                    T::one(),
                    -T::one(),
                    c(0.9)
                ));
            }
            assert!(!b.only_rejected_lattice_points(
                &[T::one(); 2],
                &[T::epsilon(); 2],
                &[Bound::unbounded(); 2],
                T::one(),
                -T::one(),
                c(0.9)
            ));
            assert!(!b.only_rejected_lattice_points(
                &[T::one()],
                &[c(2.) * T::epsilon()],
                &[Bound::unbounded()],
                T::one(),
                -T::one(),
                c(0.9)
            ));
            // Same point at a different parameter may now satisfy Armijo. Do not
            // call that stagnation when the original curvature check can pass.
            b.high_phi = c(0.999);
            b.high_derivative = T::zero();
            assert!(!b.only_rejected_lattice_points(
                &[T::one()],
                &[T::epsilon()],
                &[Bound::unbounded()],
                T::one(),
                -T::one(),
                c(0.9)
            ));
        }
        check::<f32>();
        check::<f64>();
    }

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
    fn slope_progress_guard_preserves_ordinary_interior_interpolation() {
        for reversed in [false, true] {
            let bracket = ZoomBracket {
                low_step: if reversed { 0_f64 } else { 1. },
                high_step: if reversed { 1. } else { 0. },
                low_phi: -1.,
                high_phi: 0.,
                low_derivative: if reversed { -2. } else { 2. },
                high_derivative: if reversed { 1. } else { -1. },
            };
            let ordinary = bracket.trial_step();
            assert!(ordinary > 0.1 && ordinary < 0.9);
            assert!((ordinary - 0.5).abs() > 0.1);
            assert_eq!(bracket.safeguarded_trial_step(1., 2.), ordinary);
        }
    }

    #[test]
    fn approximate_model_agreement_cannot_disable_zoom_progress_f32() {
        let scale = 2_f32.powi(80);
        let mut fg = |x: &[f32], g: &mut [f32]| {
            let d = x[0] + 0.625;
            let a = (-0.25 * d).max(0.);
            g[0] = scale * (d / 32. - 64. * a);
            Ok(scale * (d * d / 64. + 128. * a * a))
        };
        let x0 = [-0.6229999];
        let mut g0 = [0.];
        let f0 = fg(&x0, &mut g0).unwrap();
        let opts = Options::default();
        let result = strong_wolfe(
            &x0,
            f0,
            &g0,
            &[-1.],
            &[Bound::unbounded()],
            &opts,
            20,
            1.,
            &mut fg,
        )
        .unwrap();
        let step = result
            .step
            .expect("approximately agreeing models must not crawl along one endpoint");
        let displacement = x0[0] - step.x[0];
        assert!(step.f <= f0 - opts.wolfe_c1 * displacement * g0[0]);
        assert!(step.g[0].abs() <= opts.wolfe_c2 * g0[0].abs());
        assert!(result.evaluations <= 20);
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
