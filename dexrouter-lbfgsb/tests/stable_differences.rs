use dexrouter_lbfgsb::{Bound, Options, Status, minimize, minimize_with_differences};
use num_traits::{Float, FromPrimitive};

fn offset_quadratic<T: Float + FromPrimitive + std::fmt::Debug>(offset: T, tol: T) {
    let c = |v| T::from_f64(v).unwrap();
    let options = Options {
        pgtol: tol,
        ..Options::default()
    };
    let start = vec![T::one(), T::one(), c(7.0)];
    let bounds = vec![
        Bound::both(c(-2.0), c(2.0)),
        Bound::unbounded(),
        Bound::both(c(7.0), c(7.0)),
    ];
    let value = |x: &[T], g: &mut [T]| {
        g[0] = x[0];
        g[1] = c(100.0) * x[1];
        g[2] = c(123.0);
        Ok(offset + (x[0] * x[0] + c(100.0) * x[1] * x[1]) / c(2.0))
    };
    let plain = minimize(start.clone(), bounds.clone(), options.clone(), value).unwrap();
    assert!(
        plain.projected_gradient_inf_norm > tol,
        "red control: {plain:?}"
    );
    let mut calls = 0;
    let mut last = start.clone();
    let mut origin = start.clone();
    let result = minimize_with_differences(start, bounds, options, |x, g, reference| {
        calls += 1;
        if let Some(r) = reference {
            if r != origin {
                assert_eq!(r, last, "new reference must be the last accepted callback");
                origin = r.to_vec();
            }
        }
        last = x.to_vec();
        let f = value(x, g)?;
        let delta = reference.map_or(T::zero(), |r| {
            ((x[0] - r[0]) * (x[0] + r[0]) + c(100.0) * (x[1] - r[1]) * (x[1] + r[1])) / c(2.0)
        });
        Ok((f, delta))
    })
    .unwrap();
    assert_eq!(calls, result.evaluations);
    assert_eq!(
        result.status,
        Status::ConvergedProjectedGradient,
        "{result:?}"
    );
    assert!(result.x[0].abs() <= tol && (c(100.0) * result.x[1]).abs() <= tol);
    assert_eq!(result.x[2], c(7.0));
    assert_eq!(result.gradient[2], c(123.0));
    assert_eq!(result.f, offset);
}

#[test]
fn large_offset_does_not_hide_progress_f64() {
    offset_quadratic(1e20_f64, 1e-8);
}
#[test]
fn large_offset_does_not_hide_progress_f32() {
    offset_quadratic(1e10_f32, 1e-4);
}

#[test]
fn budgets_errors_and_nonfinite_differences_are_not_hidden() {
    for budget in [1, 2, 3, 10] {
        let mut calls = 0;
        let result = minimize_with_differences(
            vec![2.0_f64],
            vec![Bound::unbounded()],
            Options {
                max_evaluations: budget,
                ..Options::default()
            },
            |x, g, r| {
                calls += 1;
                g[0] = x[0].powi(3);
                Ok((
                    1e20 + x[0].powi(4) / 4.0,
                    r.map_or(0.0, |r| {
                        (x[0] - r[0]) * (x[0] + r[0]) * (x[0] * x[0] + r[0] * r[0]) / 4.0
                    }),
                ))
            },
        )
        .unwrap();
        assert_eq!(result.evaluations, calls);
        assert!(calls <= budget);
    }
    for mode in 0..4 {
        let result = minimize_with_differences(
            vec![1.0_f64],
            vec![Bound::unbounded()],
            Options::default(),
            |_, g, reference| {
                g[0] = 1.0;
                if reference.is_some() {
                    match mode {
                        0 => return Ok((f64::INFINITY, -1.0)),
                        1 => return Ok((0.0, f64::NAN)),
                        2 => g[0] = f64::NAN,
                        _ => anyhow::bail!("oracle failure"),
                    }
                }
                Ok((1e20, -1.0))
            },
        );
        assert!(result.is_err(), "mode={mode}");
    }
}

#[test]
fn accurate_uphill_difference_cannot_be_accepted_on_an_absolute_plateau() {
    let result = minimize_with_differences(
        vec![1.0_f64],
        vec![Bound::unbounded()],
        Options::default(),
        |_, g, reference| {
            g[0] = if reference.is_none() { 1.0 } else { 0.0 };
            Ok((1e20, 1.0))
        },
    )
    .unwrap();
    assert_eq!(result.status, Status::LineSearchFailed);
    assert_eq!(result.x, vec![1.0]);
    assert_eq!(result.gradient, vec![1.0]);
    assert_eq!(result.f, 1e20);
}

#[test]
fn resolved_decrease_must_not_underflow_into_zero_factr_stop() {
    let result = minimize_with_differences(
        vec![1.0_f64, 1.0],
        vec![Bound::unbounded(); 2],
        Options {
            pgtol: 0.0,
            max_iterations: 1,
            ..Options::default()
        },
        |x, g, reference| {
            g[0] = 1e-200 * x[0];
            g[1] = 1e-198 * x[1];
            let difference = reference.map_or(0.0, |r| {
                0.5e-200 * ((x[0] - r[0]) * (x[0] + r[0]) + 100.0 * (x[1] - r[1]) * (x[1] + r[1]))
            });
            Ok((1e200, difference))
        },
    )
    .unwrap();
    assert_eq!(result.status, Status::MaxIterations, "{result:?}");
    assert!(result.projected_gradient_inf_norm > 0.0);
}

#[test]
fn finite_endpoint_and_positive_factr_keep_original_semantics() {
    let endpoint = minimize_with_differences(
        vec![1.0_f64],
        vec![Bound::lower(0.5)],
        Options::default(),
        |x, g, reference| {
            g[0] = 1.0;
            Ok((1e20 + x[0], reference.map_or(0.0, |r| x[0] - r[0])))
        },
    )
    .unwrap();
    assert_eq!(endpoint.x, vec![0.5]);
    assert_eq!(endpoint.status, Status::ConvergedProjectedGradient);
    let result = minimize_with_differences(
        vec![1.0_f64, 1.0],
        vec![Bound::unbounded(); 2],
        Options {
            factr: 1e7,
            ..Options::default()
        },
        |x, g, reference| {
            g[0] = x[0];
            g[1] = 100.0 * x[1];
            Ok((
                1e20,
                reference.map_or(0.0, |r| {
                    ((x[0] - r[0]) * (x[0] + r[0]) + 100.0 * (x[1] - r[1]) * (x[1] + r[1])) / 2.0
                }),
            ))
        },
    )
    .unwrap();
    assert_eq!(result.status, Status::ConvergedFunctionReduction);
    assert!(result.projected_gradient_inf_norm > 0.1);
}
