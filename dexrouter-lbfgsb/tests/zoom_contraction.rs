//! Convex C1 objective with a flat-slope region followed by a stiff active region.
use dexrouter_lbfgsb::{Bound, LineSearchFailure, Options, Status, minimize};
use num_traits::{Float, FromPrimitive};

fn activation_wall<T: Float + FromPrimitive + std::fmt::Debug>() {
    let c = |x| T::from_f64(x).unwrap();
    for a in [c(0.875), c(0.9375)] {
        for sign in [-T::one(), T::one()] {
            for bound in [Bound::unbounded(), Bound::both(c(-2.), c(2.))] {
                let h = c(1024.);
                let options = Options::<T>::default();
                let tol = options.pgtol;
                let result = minimize(vec![T::zero()], vec![bound], options, |x, g| {
                    let active = (sign * x[0] - a).max(T::zero());
                    g[0] = sign * (h * active - T::one());
                    Ok(c(0.5) * h * active * active - sign * x[0])
                })
                .unwrap();
                assert_eq!(
                    result.status,
                    Status::ConvergedProjectedGradient,
                    "a={a:?}, sign={sign:?}: {result:?}"
                );
                assert!(
                    (h * (sign * result.x[0] - a).max(T::zero()) - T::one()).abs() <= tol,
                    "{result:?}"
                );
                assert!(
                    (result.x[0] - sign * (a + T::one() / h)).abs() <= tol / h,
                    "{result:?}"
                );
                assert!(result.f <= -a - c(0.49) / h, "{result:?}");
                assert!(result.evaluations <= 64, "{result:?}");
            }
        }
    }
}

#[test]
fn zoom_cannot_crawl_along_one_endpoint_f64() {
    activation_wall::<f64>();
}
#[test]
fn zoom_cannot_crawl_along_one_endpoint_f32() {
    activation_wall::<f32>();
}

#[test]
fn first_accepted_wall_trial_meets_original_wolfe_within_twenty_trials() {
    for a in [0.875_f64, 0.9375] {
        for sign in [-1., 1.] {
            let options = Options {
                max_iterations: 1,
                ..Options::default()
            };
            let (c1, c2) = (options.wolfe_c1, options.wolfe_c2);
            let result = minimize(vec![0.], vec![Bound::unbounded()], options, |x, g| {
                let active = (sign * x[0] - a).max(0.);
                g[0] = sign * (1024. * active - 1.);
                Ok(512. * active * active - sign * x[0])
            })
            .unwrap();
            let step = sign * result.x[0];
            assert_eq!(result.iterations, 1, "{result:?}");
            assert!(matches!(
                result.status,
                Status::MaxIterations | Status::ConvergedProjectedGradient
            ));
            assert!(step > 0. && result.evaluations <= 21);
            assert!(result.f <= -c1 * step);
            assert!((1024. * (step - a).max(0.) - 1.).abs() <= c2);
            assert_eq!(result.line_search_failure, None);
        }
    }
}

#[test]
fn contraction_preserves_hard_limits_and_last_accepted_state() {
    for (options, status, failure) in [
        (
            Options {
                max_evaluations: 5,
                ..Options::default()
            },
            Status::MaxEvaluations,
            LineSearchFailure::EvaluationLimit,
        ),
        (
            Options {
                max_line_search_steps: 4,
                ..Options::default()
            },
            Status::LineSearchFailed,
            LineSearchFailure::TrialLimit,
        ),
    ] {
        let mut calls = 0;
        let result = minimize(vec![0_f64], vec![Bound::unbounded()], options, |x, g| {
            calls += 1;
            let active = (x[0] - 0.9375).max(0.);
            g[0] = 1024. * active - 1.;
            Ok(512. * active * active - x[0])
        })
        .unwrap();
        assert_eq!(result.status, status);
        assert_eq!(result.line_search_failure, Some(failure));
        assert_eq!(result.evaluations, 5);
        assert_eq!(calls, 5);
        assert_eq!(result.x, [0.]);
        assert_eq!(result.f, 0.);
        assert_eq!(result.gradient, [-1.]);
        assert_eq!(result.iterations, 0);
    }
}
