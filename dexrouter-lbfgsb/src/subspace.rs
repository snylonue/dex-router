use num_traits::Float;

use crate::{
    bounds::{Bound, is_free},
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
    let free = cauchy
        .iter()
        .zip(bounds.iter())
        .enumerate()
        .filter_map(|(idx, (&value, bound))| is_free(value, bound).then_some(idx))
        .collect::<Vec<_>>();
    if free.is_empty() {
        return cauchy.to_vec();
    }

    if free.len() == x.len() {
        // The full-space model minimizer is x - B^{-1} g. Avoid a truncated
        // CG solve (and cancellation in g + B(z-x)) when no variable is active.
        let hg = memory.inverse_hessian_product(g);
        return x
            .iter()
            .zip(&hg)
            .zip(bounds)
            .map(|((&xi, &hi), bound)| bound.project(xi - hi))
            .collect();
    }

    let cauchy_displacement = sub(cauchy, x);
    let b_displacement = memory.hessian_product(&cauchy_displacement);
    let model_gradient = g
        .iter()
        .zip(b_displacement.iter())
        .map(|(&gi, &bi)| gi + bi)
        .collect::<Vec<_>>();

    let mut step = vec![T::zero(); x.len()];
    let mut residual = vec![T::zero(); x.len()];
    for &idx in &free {
        residual[idx] = -model_gradient[idx];
    }
    let mut direction = residual.clone();
    let mut residual_norm = norm2_squared(&residual);
    let initial_residual_norm = residual_norm;
    if residual_norm == T::zero() {
        return cauchy.to_vec();
    }

    let tolerance = T::epsilon().sqrt() * initial_residual_norm.sqrt();
    for _ in 0..free.len().max(1) {
        let mut bd = memory.hessian_product(&direction);
        for (idx, value) in bd.iter_mut().enumerate() {
            if !free.contains(&idx) {
                *value = T::zero();
            }
        }
        let curvature = dot(&direction, &bd);
        if curvature <= T::zero() {
            break;
        }
        let alpha = residual_norm / curvature;
        for &idx in &free {
            step[idx] = step[idx] + alpha * direction[idx];
            residual[idx] = residual[idx] - alpha * bd[idx];
        }
        let next_residual_norm = norm2_squared(&residual);
        if next_residual_norm.sqrt() <= tolerance {
            break;
        }
        let beta = next_residual_norm / residual_norm;
        for &idx in &free {
            direction[idx] = residual[idx] + beta * direction[idx];
        }
        residual_norm = next_residual_norm;
    }

    let mut candidate = cauchy.to_vec();
    for &idx in &free {
        candidate[idx] = bounds[idx].project(cauchy[idx] + step[idx]);
    }
    candidate
}
