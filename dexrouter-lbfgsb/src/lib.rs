//! Pure Rust L-BFGS-B optimizer.

mod bounds;
mod cauchy;
mod compact;
mod line_search;
mod memory;
mod solver;
mod subspace;

pub use bounds::Bound;
pub use line_search::LineSearchFailure;
pub use solver::{OptimizeResult, Options, Status, minimize};

use num_traits::{Float, FromPrimitive};

pub(crate) fn cast<T: Float + FromPrimitive>(value: f64) -> T {
    T::from_f64(value).expect("algorithm constant must be representable by scalar type")
}

pub(crate) fn dot<T: Float>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b.iter())
        .fold(T::zero(), |acc, (&x, &y)| acc + x * y)
}

pub(crate) fn axpy<T: Float>(y: &mut [T], alpha: T, x: &[T]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = *yi + alpha * xi;
    }
}

pub(crate) fn sub<T: Float>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

pub(crate) fn add_scaled<T: Float>(a: &[T], alpha: T, b: &[T]) -> Vec<T> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x + alpha * y)
        .collect()
}

pub(crate) fn norm_inf<T: Float>(values: &[T]) -> T {
    values
        .iter()
        .fold(T::zero(), |acc, &value| acc.max(value.abs()))
}

pub(crate) fn norm2_squared<T: Float>(values: &[T]) -> T {
    dot(values, values)
}

pub(crate) fn all_finite<T: Float>(values: &[T]) -> bool {
    values.iter().all(|value| value.is_finite())
}

#[cfg(test)]
mod regression_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        bounds::{projected_gradient, projected_steepest_direction},
        cauchy::generalized_cauchy_point,
        line_search::strong_wolfe,
        memory::LbfgsMemory,
    };

    fn quadratic<T: Float + FromPrimitive>(
        target: Vec<T>,
    ) -> impl FnMut(&[T], &mut [T]) -> anyhow::Result<T> {
        move |x, g| {
            let half = cast::<T>(0.5);
            let mut f = T::zero();
            for ((&xi, &ti), gi) in x.iter().zip(target.iter()).zip(g.iter_mut()) {
                let d = xi - ti;
                f = f + half * d * d;
                *gi = d;
            }
            Ok(f)
        }
    }

    #[test]
    fn minimizes_one_dimensional_quadratic_inside_bounds_f64() {
        let result = minimize(
            vec![0.0_f64],
            vec![Bound::new(Some(-10.0), Some(10.0))],
            Options::default(),
            quadratic(vec![3.0_f64]),
        )
        .unwrap();

        assert!((result.x[0] - 3.0).abs() < 1e-8);
        assert!(matches!(
            result.status,
            Status::ConvergedProjectedGradient | Status::ConvergedFunctionReduction
        ));
    }

    #[test]
    fn minimizes_one_dimensional_quadratic_at_lower_bound() {
        let result = minimize(
            vec![5.0_f64],
            vec![Bound::new(Some(1.0), Some(10.0))],
            Options::default(),
            quadratic(vec![-4.0_f64]),
        )
        .unwrap();

        assert!((result.x[0] - 1.0).abs() < 1e-8);
        assert!(result.projected_gradient_inf_norm < 1e-6);
    }

    #[test]
    fn minimizes_diagonal_quadratic_f32() {
        let result = minimize(
            vec![0.0_f32, 0.0, 0.0],
            vec![
                Bound::unbounded(),
                Bound::new(Some(-1.0), Some(3.0)),
                Bound::new(Some(0.0), Some(10.0)),
            ],
            Options {
                pgtol: 1e-4,
                ..Options::default()
            },
            quadratic(vec![2.0_f32, 5.0, -3.0]),
        )
        .unwrap();

        assert!((result.x[0] - 2.0).abs() < 1e-3);
        assert!((result.x[1] - 3.0).abs() < 1e-3);
        assert!((result.x[2] - 0.0).abs() < 1e-3);
    }

    #[test]
    fn minimizes_box_constrained_rosenbrock() {
        let mut options = Options::default();
        options.max_iterations = 2000;
        options.max_evaluations = 20_000;
        options.pgtol = 1e-6;

        let result = minimize(
            vec![-1.2_f64, 1.0],
            vec![
                Bound::new(Some(-2.0), Some(2.0)),
                Bound::new(Some(-1.0), Some(3.0)),
            ],
            options,
            |x, g| {
                let a = 1.0 - x[0];
                let b = x[1] - x[0] * x[0];
                g[0] = -2.0 * a - 400.0 * x[0] * b;
                g[1] = 200.0 * b;
                Ok(a * a + 100.0 * b * b)
            },
        )
        .unwrap();

        assert!((result.x[0] - 1.0).abs() < 1e-4, "{result:?}");
        assert!((result.x[1] - 1.0).abs() < 1e-4, "{result:?}");
        assert!(result.projected_gradient_inf_norm < 1e-4, "{result:?}");
    }

    #[test]
    fn projected_gradient_respects_active_bounds() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0];
        let g = vec![5.0_f64, -5.0, 7.0, -11.0];
        let bounds = vec![
            Bound::lower(1.0),
            Bound::upper(2.0),
            Bound::upper(3.0),
            Bound::lower(4.0),
        ];

        assert_eq!(
            projected_gradient(&x, &g, &bounds),
            vec![0.0, 0.0, 7.0, -11.0]
        );
        assert_eq!(
            projected_steepest_direction(&x, &g, &bounds),
            vec![0.0, 0.0, -7.0, 11.0]
        );
    }

    #[test]
    fn generalized_cauchy_point_stops_at_blocking_bound() {
        let memory = LbfgsMemory::new(5);
        let x = vec![0.0_f64];
        let g = vec![-2.0_f64];
        let bounds = vec![Bound::both(0.0, 1.0)];
        let cauchy = generalized_cauchy_point(&x, &g, &bounds, &memory);

        assert!((cauchy[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn strong_wolfe_line_search_accepts_quadratic_step() {
        let x = vec![0.0_f64];
        let f = 2.0_f64;
        let g = vec![-2.0_f64];
        let direction = vec![2.0_f64];
        let bounds = vec![Bound::unbounded()];
        let options = Options::default();
        let mut fg = quadratic(vec![2.0_f64]);

        let result = strong_wolfe(&x, f, &g, &direction, &bounds, &options, 100, 1.0, &mut fg)
            .unwrap()
            .step
            .unwrap();

        let derphi0 = dot(&g, &direction);
        let derphi = dot(&result.g, &direction);
        assert!(result.f <= f + options.wolfe_c1 * derphi0);
        assert!(derphi.abs() <= -options.wolfe_c2 * derphi0);
    }

    #[test]
    fn small_quadratic_matches_analytic_minimum() {
        let bounds = vec![Bound::both(-5.0_f64, 5.0), Bound::both(-5.0, 5.0)];
        let options = Options {
            pgtol: 1e-8,
            ..Options::default()
        };
        let ours = minimize(
            vec![0.0_f64, 0.0],
            bounds,
            options,
            quadratic(vec![2.0_f64, -1.0]),
        )
        .unwrap();

        assert!((ours.x[0] - 2.0).abs() < 1e-7);
        assert!((ours.x[1] + 1.0).abs() < 1e-7);
        assert!(ours.f.abs() < 1e-12);
    }
}
