use num_traits::Float;

use crate::{
    all_finite,
    bounds::{Bound, feasible_step_max, is_free, project},
    dot,
    memory::LbfgsMemory,
    norm2_squared, sub,
};

pub(crate) fn subspace_minimize<T: Float>(
    x: &[T],
    g: &[T],
    cauchy: &[T],
    bounds: &[Bound<T>],
    memory: &LbfgsMemory<T>,
) -> Vec<T> {
    let free_mask: Vec<_> = cauchy
        .iter()
        .zip(bounds.iter())
        .map(|(&value, bound)| is_free(value, bound))
        .collect();
    let free: Vec<_> = free_mask
        .iter()
        .enumerate()
        .filter_map(|(idx, &is_free)| is_free.then_some(idx))
        .collect();
    if free.is_empty() {
        return cauchy.to_vec();
    }

    let cauchy_displacement = sub(cauchy, x);
    let b_displacement = memory.hessian_product(&cauchy_displacement);
    let model_gradient = g
        .iter()
        .zip(b_displacement.iter())
        .map(|(&gi, &bi)| gi + bi)
        .collect::<Vec<_>>();

    if free.len() == x.len() {
        // Compute the full-space Newton target without cancellation in g + B(z-x).
        let hg = memory.inverse_hessian_product(g);
        let target = sub(x, &hg);
        return feasible_model_step(cauchy, target, &model_gradient, bounds, memory);
    }

    let mut rhs = vec![T::zero(); x.len()];
    for &idx in &free {
        rhs[idx] = -model_gradient[idx];
    }
    // No size/CG-iteration threshold: use the at-most-2m direct system whenever
    // there is a genuinely restricted subspace. CG is only a numerical fallback.
    let step = memory
        .solve_reduced(&rhs, &free)
        .unwrap_or_else(|| cg_solve(&rhs, &free, &free_mask, memory));

    let mut target = cauchy.to_vec();
    for &idx in &free {
        target[idx] = cauchy[idx] + step[idx];
    }
    feasible_model_step(cauchy, target, &model_gradient, bounds, memory)
}

fn cg_solve<T: Float>(
    rhs: &[T],
    free: &[usize],
    free_mask: &[bool],
    memory: &LbfgsMemory<T>,
) -> Vec<T> {
    let mut step = vec![T::zero(); rhs.len()];
    let mut residual = rhs.to_vec();
    let mut direction = residual.clone();
    let mut residual_norm = norm2_squared(&residual);
    if residual_norm == T::zero() {
        return step;
    }
    let tolerance = T::epsilon().sqrt() * residual_norm.sqrt();
    for _ in 0..free.len().max(1) {
        let mut bd = memory.hessian_product(&direction);
        for (idx, value) in bd.iter_mut().enumerate() {
            if !free_mask[idx] {
                *value = T::zero();
            }
        }
        let curvature = dot(&direction, &bd);
        if !curvature.is_finite() || curvature <= T::zero() {
            break;
        }
        let alpha = residual_norm / curvature;
        for &idx in free {
            step[idx] = step[idx] + alpha * direction[idx];
            residual[idx] = residual[idx] - alpha * bd[idx];
        }
        let next_residual_norm = norm2_squared(&residual);
        if next_residual_norm.sqrt() <= tolerance {
            break;
        }
        let beta = next_residual_norm / residual_norm;
        for &idx in free {
            direction[idx] = residual[idx] + beta * direction[idx];
        }
        residual_norm = next_residual_norm;
    }
    step
}

fn feasible_model_step<T: Float>(
    cauchy: &[T],
    mut target: Vec<T>,
    model_gradient: &[T],
    bounds: &[Bound<T>],
    memory: &LbfgsMemory<T>,
) -> Vec<T> {
    let step = sub(&target, cauchy);
    if !all_finite(&step) {
        return cauchy.to_vec();
    }
    // Preserve the reduced Newton direction: coordinate-wise clipping can
    // increase a coupled quadratic even when the original direction is descent.
    let alpha = feasible_step_max(cauchy, &step, bounds).min(T::one());
    if alpha <= T::zero() {
        return cauchy.to_vec();
    }
    if alpha < T::one() {
        for ((value, &zi), &di) in target.iter_mut().zip(cauchy).zip(&step) {
            *value = zi + alpha * di;
        }
    }
    project(&mut target, bounds); // Only correct roundoff after uniform truncation.
    if !all_finite(&target) {
        return cauchy.to_vec();
    }

    // Check the actual displacement, including any roundoff at the bounds.
    // Computing a model difference avoids subtracting two large model values.
    let actual_step = sub(&target, cauchy);
    let b_step = memory.hessian_product(&actual_step);
    let change =
        dot(model_gradient, &actual_step) + dot(&actual_step, &b_step) / (T::one() + T::one());
    if change.is_finite() && change <= T::zero() {
        target
    } else {
        cauchy.to_vec()
    }
}
#[cfg(test)]
#[path = "subspace_benchmarks.rs"]
mod benchmarks;

#[cfg(test)]
mod tests {
    use super::feasible_model_step;
    use crate::{Bound, memory::LbfgsMemory};

    #[test]
    fn model_guard_rejects_increases_and_nonfinite_targets() {
        let memory = LbfgsMemory::<f64>::new(1);
        let bounds = [Bound::unbounded()];
        // B=I and model gradient at the Cauchy point is zero.
        for target in [1., f64::NAN, f64::INFINITY] {
            assert_eq!(
                feasible_model_step(&[0.], vec![target], &[0.], &bounds, &memory),
                vec![0.]
            );
        }
    }
}
