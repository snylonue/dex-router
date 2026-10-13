//! Generic fixed-coordinate reduction must preserve the full callback/result ABI.
use dexrouter_lbfgsb::{Bound, Options, minimize};
use num_traits::{Float, FromPrimitive};

fn c<T: Float + FromPrimitive>(v: f64) -> T {
    T::from_f64(v).unwrap()
}

fn coupled<T: Float + FromPrimitive>(x: &[T], g: &mut [T]) -> T {
    let a = x[1] - c::<T>(0.4);
    let b = x[3] - c::<T>(-0.7);
    let h0 = c::<T>(1e6) * (a + b);
    let h2 = c::<T>(1e8) * (a - c::<T>(2.) * b);
    let h4 = c::<T>(1e6) * a * b;
    g[0] = h0;
    g[2] = h2;
    g[4] = h4;
    g[1] = c::<T>(6.) * a
        + b
        + c::<T>(1e6) * (x[0] - c(3.))
        + c::<T>(1e8) * (x[2] - c(-7.))
        + c::<T>(1e6) * b * (x[4] - c(2.));
    g[3] = a + c::<T>(3.) * b + c::<T>(1e6) * (x[0] - c(3.)) - c::<T>(2e8) * (x[2] - c(-7.))
        + c::<T>(1e6) * a * (x[4] - c(2.));
    c::<T>(0.5) * (c::<T>(6.) * a * a + c::<T>(2.) * a * b + c::<T>(3.) * b * b)
        + (x[0] - c(3.)) * h0
        + (x[2] - c(-7.)) * h2
        + (x[4] - c(2.)) * h4
}

fn compare_manual_reduction<T: Float + FromPrimitive + std::fmt::Debug>() {
    for bounded in [false, true] {
        for budget in [1, 2, 3, 6, 30, 100] {
            let variable_bounds = if bounded {
                vec![Bound::upper(c(0.1)), Bound::lower(c(-0.2))]
            } else {
                vec![Bound::unbounded(); 2]
            };
            let bounds = vec![
                Bound::both(c(3.), c(3.)),
                variable_bounds[0],
                Bound::both(c(-7.), c(-7.)),
                variable_bounds[1],
                Bound::both(c(2.), c(2.)),
            ];
            let options = Options {
                max_evaluations: budget,
                pgtol: c(1e-5),
                ..Options::default()
            };
            let mut full_trials = Vec::new();
            let full = minimize(
                vec![c(100.), c(4.), c(100.), c(4.), c(100.)],
                bounds.clone(),
                options.clone(),
                |x, g| {
                    assert_eq!(x.len(), 5);
                    assert_eq!(g.len(), 5);
                    assert_eq!([x[0], x[2], x[4]], [c(3.), c(-7.), c(2.)]);
                    assert!(x.iter().zip(&bounds).all(|(&v, b)| b.project(v) == v));
                    full_trials.push(vec![x[1], x[3]]);
                    Ok(coupled(x, g))
                },
            )
            .unwrap();
            let mut reduced_trials = Vec::new();
            let reduced = minimize(vec![c(4.), c(4.)], variable_bounds, options, |x, g| {
                reduced_trials.push(x.to_vec());
                let mut full_g = vec![T::zero(); 5];
                let f = coupled(&[c(3.), x[0], c(-7.), x[1], c(2.)], &mut full_g);
                g[0] = full_g[1];
                g[1] = full_g[3];
                Ok(f)
            })
            .unwrap();
            // In particular, large changes in fixed-coordinate gradients must
            // not enter yy/theta or alter the free-coordinate trajectory.
            assert_eq!(full_trials, reduced_trials);
            assert_eq!([full.x[1], full.x[3]], reduced.x.as_slice());
            assert_eq!(full.status, reduced.status);
            assert_eq!(full.iterations, reduced.iterations);
            assert_eq!(full.evaluations, reduced.evaluations);
            assert_eq!(full.f, reduced.f);
            assert_eq!(
                full.projected_gradient_inf_norm,
                reduced.projected_gradient_inf_norm
            );
            assert_eq!(full.evaluations, full_trials.len());
            assert!(full.evaluations <= budget);
            let mut expected = vec![T::zero(); 5];
            assert_eq!(coupled(&full.x, &mut expected), full.f);
            assert_eq!(
                full.gradient, expected,
                "return accepted gradients, not rejected trial gradients"
            );
        }
    }
}

#[test]
fn fixed_reduction_matches_manual_problem_f64() {
    compare_manual_reduction::<f64>();
}
#[test]
fn fixed_reduction_matches_manual_problem_f32() {
    compare_manual_reduction::<f32>();
}

#[test]
fn rejected_search_preserves_full_gradient_without_extra_callbacks() {
    let mut calls = 0;
    let result = minimize(
        vec![9_f64, 0.],
        vec![Bound::both(2., 2.), Bound::unbounded()],
        Options::default(),
        |x, g| {
            calls += 1;
            assert_eq!(x[0], 2.);
            g[0] = 123. + x[1]; // A changing fixed gradient in rejected trials.
            g[1] = 1.; // Deliberately inconsistent, so all descent trials fail.
            Ok(x[1] * x[1])
        },
    )
    .unwrap();
    assert_eq!(result.x, [2., 0.]);
    assert_eq!(result.gradient, [123., 1.]);
    assert_eq!(result.evaluations, calls);
}

#[test]
fn fixed_gradients_are_still_validated_and_errors_propagate() {
    for bad_on_first in [true, false] {
        let mut calls = 0;
        let result = minimize(
            vec![1_f64, 1.],
            vec![Bound::both(1., 1.), Bound::unbounded()],
            Options::default(),
            |x, g| {
                calls += 1;
                g[0] = if bad_on_first || calls > 1 {
                    f64::NAN
                } else {
                    1.
                };
                g[1] = x[1];
                Ok(0.5 * x[1] * x[1])
            },
        );
        assert!(result.unwrap_err().to_string().contains("gradient"));
    }
    let error = minimize(
        vec![1_f64],
        vec![Bound::both(1., 1.)],
        Options::default(),
        |_, _| anyhow::bail!("callback error"),
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "callback error");
    assert!(!Bound::both(f64::INFINITY, f64::INFINITY).is_fixed());
    assert!(!Bound::lower(1_f64).is_fixed());
}

#[test]
fn active_bounds_and_initially_zero_gradients_are_not_eliminated() {
    for bound in [Bound::unbounded(), Bound::lower(0_f64)] {
        let result = minimize(
            vec![0_f64, 0.],
            vec![bound, Bound::unbounded()],
            Options {
                pgtol: 1e-9,
                ..Options::default()
            },
            |x, g| {
                g[0] = x[0] - x[1]; // Initially zero, but not a fixed variable.
                g[1] = x[1] - x[0] + x[1] - 1.;
                Ok(0.5 * (x[0] - x[1]).powi(2) + 0.5 * (x[1] - 1.).powi(2))
            },
        )
        .unwrap();
        assert!((result.x[0] - 1.).abs() < 1e-8);
        assert!((result.x[1] - 1.).abs() < 1e-8);
    }
    let upper = 1. + f64::EPSILON;
    let bound = Bound::both(1., upper);
    assert!(!bound.is_fixed());
    let result = minimize(
        vec![1_f64],
        vec![bound],
        Options {
            pgtol: 0.,
            ..Options::default()
        },
        |x, g| {
            g[0] = -1.;
            Ok(-x[0])
        },
    )
    .unwrap();
    assert_eq!(result.x[0], upper);
    assert_eq!(result.evaluations, 2);
}
