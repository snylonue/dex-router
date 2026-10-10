//! Adapted from SciPy v1.15.2; see README.md and SCIPY_LICENSE.txt.
use dexrouter_lbfgsb::{Bound, OptimizeResult, Options, Status, minimize};

fn success(r: &OptimizeResult<f64>) {
    assert!(
        matches!(
            r.status,
            Status::ConvergedProjectedGradient | Status::ConvergedFunctionReduction
        ),
        "{r:?}"
    );
}

fn rosen(x: &[f64], g: &mut [f64]) -> anyhow::Result<f64> {
    g.fill(0.0);
    let mut f = 0.0;
    for i in 0..x.len() - 1 {
        let a = 1.0 - x[i];
        let b = x[i + 1] - x[i] * x[i];
        f += a * a + 100.0 * b * b;
        g[i] += -2.0 * a - 400.0 * x[i] * b;
        g[i + 1] += 200.0 * b;
    }
    Ok(f)
}

fn entropy(x: &[f64], g: &mut [f64]) -> anyhow::Result<f64> {
    let features = [
        [1., 1., 1.],
        [1., 1., 0.],
        [1., 0., 1.],
        [1., 0., 0.],
        [1., 0., 0.],
    ];
    let k = [1., 0.3, 0.5];
    let logits: Vec<f64> = features
        .iter()
        .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect();
    let shift = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let z: f64 = logits.iter().map(|v| (v - shift).exp()).sum();
    for j in 0..3 {
        g[j] = features
            .iter()
            .zip(&logits)
            .map(|(r, v)| r[j] * (v - shift).exp() / z)
            .sum::<f64>()
            - k[j];
    }
    Ok(shift + z.ln() - k.iter().zip(x).map(|(a, b)| a * b).sum::<f64>())
}

#[test]
fn scipy_entropy_combined_objective_gradient() {
    let r = minimize(
        vec![0.; 3],
        vec![Bound::unbounded(); 3],
        Options::default(),
        entropy,
    )
    .unwrap();
    success(&r);
    let expected = entropy(&[0., -0.524869316, 0.487525860], &mut [0.; 3]).unwrap();
    assert!((r.f - expected).abs() < 1e-6, "{r:?}");
}

#[test]
fn scipy_ftol() {
    let expected = entropy(&[0., -0.524869316, 0.487525860], &mut [0.; 3]).unwrap();
    let mut previous = f64::INFINITY;
    for ftol in [1e-1, 1e-4, 1e-7, 1e-10] {
        let r = minimize(
            vec![0.; 3],
            vec![Bound::unbounded(); 3],
            Options {
                factr: ftol / f64::EPSILON,
                ..Options::default()
            },
            entropy,
        )
        .unwrap();
        success(&r);
        assert!(r.f <= previous, "{r:?}");
        assert!((r.f - expected).abs() <= ftol * expected.abs(), "{r:?}");
        previous = r.f;
    }
}

#[test]
fn scipy_lower_bound_and_infeasible_start() {
    let r = minimize(
        vec![0., -1.],
        vec![Bound::lower(1.), Bound::unbounded()],
        Options::default(),
        |x, g| {
            assert!(x[0] >= 1.);
            g.copy_from_slice(x);
            Ok(0.5 * (x[0] * x[0] + x[1] * x[1]))
        },
    )
    .unwrap();
    success(&r);
    assert!((r.x[0] - 1.).abs() < 1e-6 && r.x[1].abs() < 1e-6, "{r:?}");
}

#[test]
fn scipy_incorrect_bounds() {
    for bounds in [
        vec![Bound::both(10., 1.), Bound::both(1., 10.)],
        vec![Bound::both(1., 10.), Bound::both(10., 1.)],
        vec![Bound::both(10., 1.); 2],
    ] {
        let error = minimize(vec![0., -1.], bounds, Options::default(), rosen).unwrap_err();
        assert!(error.to_string().contains("bound"));
    }
}

#[test]
fn scipy_maxiter_gh7854() {
    let r = minimize(
        vec![0.; 2],
        vec![Bound::unbounded(); 2],
        Options {
            max_iterations: 5,
            ..Options::default()
        },
        rosen,
    )
    .unwrap();
    assert_eq!(r.status, Status::MaxIterations, "{r:?}");
    assert_eq!(r.iterations, 5);
    let mut g = vec![0.; 2];
    assert_eq!(r.f, rosen(&r.x, &mut g).unwrap());
    assert_eq!(r.gradient, g);
}

#[test]
fn scipy_maxls() {
    let mut calls = 0;
    let r = minimize(
        vec![-1.2, 1.],
        vec![Bound::unbounded(); 2],
        Options {
            max_line_search_steps: 1,
            ..Options::default()
        },
        |x, g| {
            calls += 1;
            rosen(x, g)
        },
    )
    .unwrap();
    assert_eq!(r.status, Status::LineSearchFailed, "{r:?}");
    assert_eq!(r.x, vec![-1.2, 1.]);
    assert_eq!(r.evaluations, calls);
}

#[test]
fn scipy_single_precision_gradient_gh18730() {
    let r = minimize(
        vec![1_f32; 2],
        vec![Bound::unbounded(); 2],
        Options::default(),
        |x, g| {
            for (gi, xi) in g.iter_mut().zip(x) {
                *gi = 2.0 * xi;
            }
            Ok(x.iter().map(|xi| xi * xi).sum())
        },
    )
    .unwrap();
    assert_eq!(r.status, Status::ConvergedProjectedGradient);
    assert!(r.f.abs() <= 1e-15, "{r:?}");
}

#[test]
fn scipy_single_precision_callback_with_f64_iterates_gh18730() {
    let r = minimize(
        vec![1_f64; 2],
        vec![Bound::unbounded(); 2],
        Options::default(),
        |x, g| {
            let single: Vec<f32> = x.iter().map(|&v| v as f32).collect();
            for (gi, &xi) in g.iter_mut().zip(&single) {
                *gi = (2.0 * xi) as f64;
            }
            Ok(single.iter().map(|v| v * v).sum::<f32>() as f64)
        },
    )
    .unwrap();
    success(&r);
    assert!(r.f <= 1e-15, "{r:?}");
}

#[test]
fn scipy_quartic_memory_and_tolerance_sweep() {
    // Upstream hessinv test_1 objective/parameter grid; this API exposes no inverse Hessian.
    for pgtol in [1e-8, 1e-12, 1e-20] {
        for memory in 20..35 {
            let r = minimize(
                vec![20_f64],
                vec![Bound::unbounded()],
                Options {
                    memory,
                    pgtol,
                    factr: 0.,
                    ..Options::default()
                },
                |x, g| {
                    g[0] = 4. * x[0].powi(3);
                    Ok(x[0].powi(4))
                },
            )
            .unwrap();
            success(&r);
            assert!(
                r.projected_gradient_inf_norm <= pgtol,
                "memory={memory} tol={pgtol}: {r:?}"
            );
        }
    }
}

#[test]
fn scipy_maxfun_interruption_gh6162() {
    let mut values = Vec::new();
    let _ = minimize(
        vec![1000.; 7],
        vec![Bound::unbounded(); 7],
        Options {
            max_evaluations: 300,
            ..Options::default()
        },
        |x, g| {
            let f = rosen(x, g)?;
            values.push(f);
            Ok(f)
        },
    )
    .unwrap();
    assert!(values.len() > 100);
    let (offset, _) = values[100..]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap();
    let budget = 100 + offset;
    let target = values[..30].iter().copied().fold(f64::INFINITY, f64::min);
    let mut calls = 0;
    let r = minimize(
        vec![1000.; 7],
        vec![Bound::unbounded(); 7],
        Options {
            max_evaluations: budget,
            ..Options::default()
        },
        |x, g| {
            calls += 1;
            rosen(x, g)
        },
    )
    .unwrap();
    assert_eq!(r.status, Status::MaxEvaluations, "{r:?}");
    assert_eq!(r.evaluations, calls);
    assert!(calls <= budget, "calls={calls} budget={budget}");
    assert!(r.f < target, "{r:?}");
}
