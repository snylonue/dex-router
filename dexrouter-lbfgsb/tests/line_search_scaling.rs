//! Analytic regressions for the optimizer itself, with no market or DD callbacks.
use dexrouter_lbfgsb::{Bound, LineSearchFailure, Options, Status, minimize};
use num_traits::{Float, FromPrimitive};

fn narrow_quadratic<T: Float + FromPrimitive + std::fmt::Debug>(curvatures: &[f64]) {
    let c = |x| T::from_f64(x).unwrap();
    for &curvature in curvatures {
        let h = c(curvature);
        for sign in [-T::one(), T::one()] {
            for bound in [
                Bound::unbounded(),
                Bound::lower(-T::one()),
                Bound::upper(T::one()),
                Bound::both(-T::one(), T::one()),
            ] {
                let mut calls = 0;
                let options = Options::<T>::default();
                let pgtol = options.pgtol;
                let result = minimize(vec![T::zero()], vec![bound], options, |x, g| {
                    calls += 1;
                    g[0] = h * x[0] - sign;
                    Ok(c(0.5) * (h * x[0]) * x[0] - sign * x[0])
                })
                .unwrap();
                assert_eq!(result.evaluations, calls);
                assert_eq!(
                    result.status,
                    Status::ConvergedProjectedGradient,
                    "h={curvature}, bound={bound:?}: {result:?}"
                );
                // Independent analytic KKT check, not just the solver's status.
                assert!((h * result.x[0] - sign).abs() <= pgtol, "{result:?}");
                assert!(result.f <= -c(0.49) / h, "{result:?}");
                assert!(
                    calls <= 4,
                    "an accurate quadratic trial must not be clipped: {result:?}"
                );
                if curvature == 1e24 {
                    assert_eq!(calls, 3, "{result:?}");
                    assert_eq!(result.iterations, 1);
                }
            }
        }
    }
}

#[test]
fn nonplateau_narrow_quadratic_f64() {
    narrow_quadratic::<f64>(&[1e24, 1e8, 1e80, 1e160, 1e300]);
}

#[test]
fn nonplateau_narrow_quadratic_f32() {
    narrow_quadratic::<f32>(&[1e24, 1e8, 1e30]);
}

fn flat_quadratic<T: Float + FromPrimitive + std::fmt::Debug>(curvatures: &[f64]) {
    let c = |x| T::from_f64(x).unwrap();
    for &curvature in curvatures {
        let h = c(curvature);
        for sign in [-T::one(), T::one()] {
            let options = Options {
                // Resolving curvature this close to f32 epsilon needs more probes.
                max_line_search_steps: if T::epsilon() > c(1e-10) && curvature < 1e-10 {
                    40
                } else {
                    20
                },
                ..Options::default()
            };
            let result = minimize(
                vec![T::zero()],
                vec![Bound::unbounded()],
                options,
                |x, g| {
                    g[0] = h * x[0] - sign;
                    Ok(c(0.5) * (h * x[0]) * x[0] - sign * x[0])
                },
            )
            .unwrap();
            assert_eq!(
                result.status,
                Status::ConvergedProjectedGradient,
                "h={curvature}: {result:?}"
            );
            assert!((h * result.x[0] - sign).abs() <= Options::<T>::default().pgtol);
            assert!(result.f <= -c(0.49) / h);
            assert!(result.evaluations <= 32, "{result:?}");
            if curvature >= 1e-12_f64 && T::epsilon() < c(1e-10) {
                assert!(result.evaluations <= 5, "{result:?}");
            }
        }
    }
}

#[test]
fn unbracketed_flat_quadratic_f64() {
    flat_quadratic::<f64>(&[1e-12, 1e-6, 1e-16, 1e-18]);
}

#[test]
fn unbracketed_flat_quadratic_f32() {
    flat_quadratic::<f32>(&[1e-10, 1e-6, 1e-12]);
}

#[test]
fn nonlinear_curvature_does_not_trust_endpoint_secant_unconditionally() {
    for sign in [-1_f64, 1.] {
        let result = minimize(
            vec![0.],
            vec![Bound::unbounded()],
            Options::default(),
            |x, g| {
                g[0] = 1e24 * x[0].powi(3) - sign;
                Ok(0.25e24 * x[0].powi(4) - sign * x[0])
            },
        )
        .unwrap();
        assert_eq!(
            result.status,
            Status::ConvergedProjectedGradient,
            "{result:?}"
        );
        assert!((1e24 * result.x[0].powi(3) - sign).abs() <= 1e-5);
        assert!(result.f <= -0.74e-8);
        assert!(result.evaluations <= 21, "{result:?}");
    }
}

#[test]
fn almost_linear_exponential_does_not_trigger_unresolved_extrapolation() {
    for eps in [1e-6_f64, 1e-15] {
        let result = minimize(
            vec![0.],
            vec![Bound::unbounded()],
            Options::default(),
            |x, g| {
                assert!(x[0] < 700., "unsafe exponential trial: {x:?}");
                let exp = x[0].exp();
                g[0] = eps * exp - 1.;
                Ok(eps * (exp - 1.) - x[0])
            },
        )
        .unwrap();
        assert_eq!(
            result.status,
            Status::ConvergedProjectedGradient,
            "{result:?}"
        );
        assert!((eps * result.x[0].exp() - 1.).abs() <= 1e-5);
        assert!((result.x[0] - (1. / eps).ln()).abs() <= 2e-5);
    }
}

#[test]
fn extrapolated_trial_is_capped_at_the_original_feasible_endpoint() {
    let upper = 1e6_f64;
    let result = minimize(
        vec![0.],
        vec![Bound::both(0., upper)],
        Options::default(),
        |x, g| {
            assert!((0. ..=upper).contains(&x[0]));
            g[0] = 1e-12 * x[0] - 1.;
            Ok(0.5e-12 * x[0] * x[0] - x[0])
        },
    )
    .unwrap();
    assert_eq!(
        result.status,
        Status::ConvergedProjectedGradient,
        "{result:?}"
    );
    assert_eq!(result.x, [upper]);
    assert!(result.gradient[0] < 0.);
    assert!(result.evaluations <= 4);
}

#[test]
fn accurate_trial_still_respects_evaluation_and_search_limits() {
    for (options, status, failure) in [
        (
            Options {
                max_evaluations: 2,
                ..Options::default()
            },
            Status::MaxEvaluations,
            LineSearchFailure::EvaluationLimit,
        ),
        (
            Options {
                max_line_search_steps: 1,
                ..Options::default()
            },
            Status::LineSearchFailed,
            LineSearchFailure::TrialLimit,
        ),
    ] {
        for h in [1e24_f64, 1e-12] {
            let mut calls = 0;
            let result = minimize(
                vec![0_f64],
                vec![Bound::unbounded()],
                options.clone(),
                |x, g| {
                    calls += 1;
                    g[0] = h * x[0] - 1.;
                    Ok(0.5 * h * x[0] * x[0] - x[0])
                },
            )
            .unwrap();
            assert_eq!(result.status, status, "{result:?}");
            assert_eq!(result.line_search_failure, Some(failure));
            assert_eq!(result.x, [0.]);
            assert_eq!(result.gradient, [-1.]);
            assert_eq!(result.f, 0.);
            assert_eq!(result.evaluations, 2);
            assert_eq!(calls, 2);
        }
    }
}
