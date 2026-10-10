//! Additional deterministic input coverage and comparison with the classic C/Fortran port.
use dexrouter_lbfgsb::{Bound, Options, Status, minimize};
use num_traits::{Float, FromPrimitive};

fn c<T: Float + FromPrimitive>(v: f64) -> T {
    T::from_f64(v).unwrap()
}
fn converged(status: Status) -> bool {
    matches!(
        status,
        Status::ConvergedProjectedGradient | Status::ConvergedFunctionReduction
    )
}

fn diagonal_matrix<T: Float + FromPrimitive + std::fmt::Debug>() {
    for n in [1, 2, 5, 32, 100] {
        for scale in [1e-6, 1., 1e6] {
            for memory in [1, 5, 17] {
                let mut bounds = Vec::new();
                let mut target = Vec::new();
                let mut expected = Vec::new();
                for i in 0..n {
                    let b = match i % 5 {
                        0 => Bound::unbounded(),
                        1 => Bound::lower(c(-1.)),
                        2 => Bound::upper(c(1.)),
                        3 => Bound::both(c(-1.), c(1.)),
                        _ => Bound::both(c(0.25), c(0.25)),
                    };
                    let t = c(if i % 2 == 0 { 2. } else { -2. });
                    expected.push(b.project(t));
                    target.push(t);
                    bounds.push(b);
                }
                let pgtol = c::<T>(scale) * T::epsilon().sqrt() * c(2.);
                let r = minimize(
                    vec![c::<T>(4.); n],
                    bounds.clone(),
                    Options {
                        memory,
                        pgtol,
                        max_iterations: 3000,
                        max_evaluations: 30000,
                        ..Options::default()
                    },
                    |x, g| {
                        let mut f = T::zero();
                        for i in 0..n {
                            assert_eq!(
                                x[i],
                                bounds[i].project(x[i]),
                                "callback received infeasible point"
                            );
                            let weight = c::<T>(scale * (1. + (i % 7) as f64));
                            let d = x[i] - target[i];
                            g[i] = weight * d;
                            // Remove the unavoidable constrained-minimum offset.
                            // Otherwise rounding of that constant can trigger factr=0
                            // before this test's deliberately tight gradient tolerance.
                            let delta = x[i] - expected[i];
                            let best_d = expected[i] - target[i];
                            f = f + c::<T>(0.5) * weight * delta * (delta + c::<T>(2.) * best_d);
                        }
                        Ok(f)
                    },
                )
                .unwrap();
                assert!(
                    converged(r.status),
                    "n={n} scale={scale} memory={memory}: {r:?}"
                );
                for (actual, expect) in r.x.iter().zip(&expected) {
                    assert!(
                        (*actual - *expect).abs() < T::epsilon().sqrt() * c(100.),
                        "n={n} scale={scale} memory={memory}: {r:?}"
                    );
                }
                assert!(
                    r.projected_gradient_inf_norm <= pgtol * c(10.),
                    "n={n} scale={scale} memory={memory}: {r:?}"
                );
            }
        }
    }
}

#[test]
fn diagonal_input_matrix_f64() {
    diagonal_matrix::<f64>();
}
#[test]
fn diagonal_input_matrix_f32() {
    diagonal_matrix::<f32>();
}

#[test]
fn tiny_nonzero_direction_is_not_failure() {
    let r = minimize(
        vec![1e-10_f64],
        vec![Bound::unbounded()],
        Options {
            pgtol: 1e-15,
            ..Options::default()
        },
        |x, g| {
            g[0] = x[0];
            Ok(0.5 * x[0] * x[0])
        },
    )
    .unwrap();
    assert_eq!(r.status, Status::ConvergedProjectedGradient, "{r:?}");
    assert!(r.x[0].abs() <= 1e-15);
}

fn narrow_quadratics<T: Float + FromPrimitive + std::fmt::Debug>() {
    for weight in [1e8, 1e12] {
        for sign in [-1., 1.] {
            for bounded in [false, true] {
                let target = c::<T>(sign * 1e-8);
                let weight = c::<T>(weight);
                let mut calls = 0;
                let options = Options::default();
                let pgtol = options.pgtol;
                let r = minimize(
                    vec![T::zero()],
                    vec![if bounded {
                        Bound::both(c(-1.), c(1.))
                    } else {
                        Bound::unbounded()
                    }],
                    options,
                    |x, g| {
                        calls += 1;
                        let d = x[0] - target;
                        g[0] = weight * d;
                        Ok(c::<T>(0.5) * weight * d * d)
                    },
                )
                .unwrap();
                assert_eq!(r.status, Status::ConvergedProjectedGradient, "{r:?}");
                assert!((r.x[0] - target).abs() <= pgtol / weight, "{r:?}");
                assert!(r.projected_gradient_inf_norm <= pgtol, "{r:?}");
                assert_eq!(r.evaluations, calls);
                assert!(
                    calls <= 21,
                    "interpolation exceeded the default first-search limit: {r:?}"
                );
            }
        }
    }
}

#[test]
fn narrow_quadratics_converge_with_default_line_search_f64() {
    narrow_quadratics::<f64>();
}

#[test]
fn narrow_quadratics_converge_with_default_line_search_f32() {
    narrow_quadratics::<f32>();
}

#[test]
fn finite_boundary_step_can_satisfy_armijo_without_wolfe_curvature() {
    let mut calls = 0;
    let r = minimize(
        vec![0_f64],
        vec![Bound::both(0., 1.)],
        Options::default(),
        |x, g| {
            calls += 1;
            g[0] = -1.;
            Ok(-x[0])
        },
    )
    .unwrap();
    assert_eq!(r.status, Status::ConvergedProjectedGradient, "{r:?}");
    assert_eq!(r.x, vec![1.]);
    assert_eq!(r.evaluations, calls);
    assert!(
        calls <= 3,
        "repeatedly evaluates a capped boundary: {calls}"
    );
}

#[test]
fn failed_search_does_not_accept_an_increasing_objective() {
    // Deliberately inconsistent derivative: no descent candidate exists.
    let mut calls = 0;
    let r = minimize(
        vec![0_f64],
        vec![Bound::unbounded()],
        Options::default(),
        |x, g| {
            calls += 1;
            g[0] = 1.;
            Ok(x[0] * x[0])
        },
    )
    .unwrap();
    assert_eq!(r.status, Status::LineSearchFailed, "{r:?}");
    assert_eq!(r.x, vec![0.]);
    assert_eq!(r.f, 0.);
    assert_eq!(r.evaluations, calls);
}

#[test]
fn zero_factr_still_allows_equal_value_function_reduction_stop() {
    // A large objective offset can hide progress, even though the gradient
    // is not small. Retain the documented Fortran-compatible factr=0 semantics.
    let options = Options::default();
    assert_eq!(options.factr, 0.);
    let pgtol = options.pgtol;
    let r = minimize(
        vec![1_f64, 1.],
        vec![Bound::both(-10., 10.); 2],
        options,
        |x, g| {
            g[0] = 2. * x[0];
            g[1] = x[1];
            Ok(1e20 + x[0] * x[0] + 0.5 * x[1] * x[1])
        },
    )
    .unwrap();
    assert_eq!(r.status, Status::ConvergedFunctionReduction, "{r:?}");
    assert!(r.projected_gradient_inf_norm > pgtol);
    assert_eq!(r.f, 1e20);
}

#[test]
fn hard_evaluation_budget_and_consistent_last_accepted_state() {
    for budget in [1, 2, 3, 5, 10, 17] {
        let mut calls = 0;
        let r = minimize(
            vec![10_f64],
            vec![Bound::unbounded()],
            Options {
                max_evaluations: budget,
                pgtol: 0.,
                ..Options::default()
            },
            |x, g| {
                calls += 1;
                g[0] = 4. * x[0].powi(3);
                Ok(x[0].powi(4))
            },
        )
        .unwrap();
        assert_eq!(r.evaluations, calls);
        assert!(calls <= budget, "budget={budget} calls={calls} {r:?}");
        assert_eq!(r.status, Status::MaxEvaluations, "budget={budget}: {r:?}");
        assert_eq!(r.f, r.x[0].powi(4));
        assert_eq!(r.gradient[0], 4. * r.x[0].powi(3));
        assert!(r.f <= 10000.);
    }
}

#[test]
fn fixed_variables_and_stationary_start_need_one_evaluation() {
    for (start, bounds) in [
        (
            vec![3_f64, -2.],
            vec![Bound::both(1., 1.), Bound::both(2., 2.)],
        ),
        (vec![0., 0.], vec![Bound::unbounded(); 2]),
    ] {
        let r = minimize(start, bounds, Options::default(), |x, g| {
            g.copy_from_slice(x);
            Ok(x.iter().map(|v| v * v * 0.5).sum())
        })
        .unwrap();
        assert_eq!(r.status, Status::ConvergedProjectedGradient);
        assert_eq!(r.iterations, 0);
        assert_eq!(r.evaluations, 1);
    }
}

#[test]
fn invalid_problem_and_options_are_rejected_before_callback() {
    let default = Options::<f64>::default();
    let mut cases = vec![
        (vec![], vec![], default.clone()),
        (vec![1.], vec![], default.clone()),
        (vec![f64::NAN], vec![Bound::unbounded()], default.clone()),
        (
            vec![f64::INFINITY],
            vec![Bound::unbounded()],
            default.clone(),
        ),
    ];
    for b in [
        Bound::lower(f64::NAN),
        Bound::upper(f64::INFINITY),
        Bound::both(2., 1.),
    ] {
        cases.push((vec![1.], vec![b], default.clone()));
    }
    let options = [
        Options {
            memory: 0,
            ..default.clone()
        },
        Options {
            max_iterations: 0,
            ..default.clone()
        },
        Options {
            max_evaluations: 0,
            ..default.clone()
        },
        Options {
            max_line_search_steps: 0,
            ..default.clone()
        },
        Options {
            factr: -1.,
            ..default.clone()
        },
        Options {
            factr: f64::INFINITY,
            ..default.clone()
        },
        Options {
            pgtol: f64::NAN,
            ..default.clone()
        },
        Options {
            pgtol: f64::INFINITY,
            ..default.clone()
        },
        Options {
            wolfe_c1: 0.,
            ..default.clone()
        },
        Options {
            wolfe_c2: 1.,
            ..default.clone()
        },
        Options {
            wolfe_c1: 0.9,
            wolfe_c2: 0.1,
            ..default
        },
    ];
    for o in options {
        cases.push((vec![1.], vec![Bound::unbounded()], o));
    }
    for (x, b, o) in cases {
        assert!(minimize(x, b, o, |_, _| panic!("invalid input invoked callback")).is_err());
    }
}

#[test]
fn nonfinite_callback_values_and_errors_are_reported() {
    for during_search in [false, true] {
        for bad_gradient in [false, true] {
            for bad_value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut calls = 0;
                let r = minimize(
                    vec![1_f64],
                    vec![Bound::unbounded()],
                    Options::default(),
                    |x, g| {
                        calls += 1;
                        g[0] = x[0];
                        if !during_search || calls > 1 {
                            if bad_gradient {
                                g[0] = bad_value;
                            } else {
                                return Ok(bad_value);
                            }
                        }
                        Ok(0.5 * x[0] * x[0])
                    },
                );
                assert!(r.unwrap_err().to_string().contains("non-finite"));
            }
        }
        let mut calls = 0;
        let r = minimize(
            vec![1_f64],
            vec![Bound::unbounded()],
            Options::default(),
            |x, g| {
                calls += 1;
                if !during_search || calls > 1 {
                    anyhow::bail!("callback sentinel");
                }
                g[0] = x[0];
                Ok(0.5 * x[0] * x[0])
            },
        );
        assert_eq!(r.unwrap_err().to_string(), "callback sentinel");
    }
}

#[test]
fn coupled_spd_quadratics_match_reference() {
    // Q = diagonal + rank-one SPD update. All four bound kinds, including fixed coordinates.
    for n in [2, 5, 20] {
        for memory in [1, 5, 17] {
            let bounds: Vec<_> = (0..n)
                .map(|i| match i % 5 {
                    0 => Bound::unbounded(),
                    1 => Bound::lower(-0.5),
                    2 => Bound::upper(0.5),
                    3 => Bound::both(-0.5, 0.5),
                    _ => Bound::both(0.1, 0.1),
                })
                .collect();
            let target: Vec<_> = (0..n)
                .map(|i| if i % 2 == 0 { 1.5 } else { -1.5 })
                .collect();
            let fg = |x: &[f64], g: &mut [f64]| -> anyhow::Result<f64> {
                assert!(x.iter().zip(&bounds).all(|(&v, b)| v == b.project(v)));
                let sum: f64 = x.iter().zip(&target).map(|(a, b)| a - b).sum();
                let mut f = 0.1 * sum * sum;
                for i in 0..n {
                    let d = x[i] - target[i];
                    let weight = 1. + (i % 7) as f64;
                    g[i] = weight * d + 0.2 * sum;
                    f += 0.5 * weight * d * d;
                }
                Ok(f)
            };
            let opts = Options {
                memory,
                pgtol: 1e-7,
                max_iterations: 3000,
                ..Options::default()
            };
            let ours = minimize(vec![3.; n], bounds.clone(), opts, fg).unwrap();
            assert!(converged(ours.status), "n={n} m={memory}: {ours:?}");
            let mut problem = lbfgsb::LbfgsbProblem::build(vec![3.; n], fg);
            problem.set_bounds(bounds.iter().map(|b| (b.lower, b.upper)));
            let mut reference = lbfgsb::LbfgsbState::new(
                problem,
                lbfgsb::LbfgsbParameter {
                    m: memory,
                    factr: 0.,
                    pgtol: 1e-7,
                    iprint: -1,
                },
            );
            reference.minimize().unwrap();
            assert!(
                (ours.f - reference.fx()).abs() < 1e-8,
                "n={n} m={memory}: {ours:?} reference f={}",
                reference.fx()
            );
            for (&a, &b) in ours.x.iter().zip(reference.x()) {
                assert!(
                    (a - b).abs() < 2e-5,
                    "n={n} m={memory}: {ours:?} ref={:?}",
                    reference.x()
                );
            }
            // Independent KKT residual; don't rely only on the solver's reported norm.
            let mut g = vec![0.; n];
            fg(&ours.x, &mut g).unwrap();
            let residual = g
                .iter()
                .zip(&ours.x)
                .zip(&bounds)
                .map(|((&gi, &xi), b)| {
                    if (b.lower == Some(xi) && gi > 0.) || (b.upper == Some(xi) && gi < 0.) {
                        0.
                    } else {
                        gi.abs()
                    }
                })
                .fold(0_f64, f64::max);
            assert!(
                residual < 2e-5,
                "n={n} m={memory} residual={residual}: {ours:?}"
            );
        }
    }
}
