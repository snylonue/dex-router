use std::cmp::Ordering;

use num_traits::Float;

use crate::{
    bounds::{Bound, projected_steepest_direction},
    dot,
    memory::LbfgsMemory,
    sub,
};

#[derive(Debug, Clone, Copy)]
struct Breakpoint<T> {
    t: T,
    index: usize,
    value: T,
}

pub(crate) fn generalized_cauchy_point<T: Float>(
    x: &[T],
    g: &[T],
    bounds: &[Bound<T>],
    memory: &LbfgsMemory<T>,
) -> Vec<T> {
    let mut z = x.to_vec();
    let mut direction = projected_steepest_direction(x, g, bounds);
    let mut breakpoints = Vec::new();

    for (i, ((&xi, &di), bound)) in x
        .iter()
        .zip(direction.iter())
        .zip(bounds.iter())
        .enumerate()
    {
        if di > T::zero() {
            if let Some(upper) = bound.upper {
                let t = (upper - xi) / di;
                if t >= T::zero() && t.is_finite() {
                    breakpoints.push(Breakpoint {
                        t,
                        index: i,
                        value: upper,
                    });
                }
            }
        } else if di < T::zero() {
            if let Some(lower) = bound.lower {
                let t = (lower - xi) / di;
                if t >= T::zero() && t.is_finite() {
                    breakpoints.push(Breakpoint {
                        t,
                        index: i,
                        value: lower,
                    });
                }
            }
        }
    }

    breakpoints.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(Ordering::Equal));

    let mut previous_t = T::zero();
    let mut cursor = 0usize;
    loop {
        let next_t = breakpoints
            .get(cursor)
            .map(|breakpoint| breakpoint.t)
            .unwrap_or_else(T::infinity);
        let dt = next_t - previous_t;
        let displacement = sub(&z, x);
        let bd = memory.hessian_product(&direction);
        let derivative = dot(g, &direction) + dot(&displacement, &bd);
        let curvature = dot(&direction, &bd);

        // Removing a blocked coordinate can make the right derivative nonnegative.
        // The breakpoint is then the first local model minimum on the projected path.
        if derivative >= T::zero() {
            project_to_bounds(&mut z, bounds);
            return z;
        }

        if curvature > T::zero() {
            let minimizer_dt = -derivative / curvature;
            if minimizer_dt >= T::zero() && (next_t.is_infinite() || minimizer_dt <= dt) {
                for (zi, &di) in z.iter_mut().zip(direction.iter()) {
                    *zi = *zi + minimizer_dt * di;
                }
                project_to_bounds(&mut z, bounds);
                return z;
            }
        }

        if next_t.is_infinite() {
            let fallback_dt = if derivative < T::zero() {
                T::one()
            } else {
                T::zero()
            };
            for (zi, &di) in z.iter_mut().zip(direction.iter()) {
                *zi = *zi + fallback_dt * di;
            }
            project_to_bounds(&mut z, bounds);
            return z;
        }

        for (zi, &di) in z.iter_mut().zip(direction.iter()) {
            *zi = *zi + dt * di;
        }

        while cursor < breakpoints.len() && breakpoints[cursor].t <= next_t {
            let breakpoint = breakpoints[cursor];
            z[breakpoint.index] = breakpoint.value;
            direction[breakpoint.index] = T::zero();
            cursor += 1;
        }

        previous_t = next_t;
        if direction.iter().all(|value| value.is_zero()) {
            project_to_bounds(&mut z, bounds);
            return z;
        }
    }
}

fn project_to_bounds<T: Float>(x: &mut [T], bounds: &[Bound<T>]) {
    for (xi, bound) in x.iter_mut().zip(bounds.iter()) {
        *xi = bound.project(*xi);
    }
}
