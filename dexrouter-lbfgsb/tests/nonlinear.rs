use dexrouter_lbfgsb::{Bound, Options, Status, minimize};
use num_traits::{Float, FromPrimitive};
use std::sync::Mutex;

// Only the legacy reference is non-reentrant; never lock the Rust solver.
static REFERENCE_LOCK: Mutex<()> = Mutex::new(());

fn c<T: Float + FromPrimitive>(v: f64) -> T {
    T::from_f64(v).unwrap()
}
fn is_success(s: Status) -> bool {
    matches!(
        s,
        Status::ConvergedProjectedGradient | Status::ConvergedFunctionReduction
    )
}
fn rosen<T: Float + FromPrimitive>(x: &[T], g: &mut [T]) -> anyhow::Result<T> {
    g.fill(T::zero());
    let mut f = T::zero();
    for i in 0..x.len() - 1 {
        let a = T::one() - x[i];
        let b = x[i + 1] - x[i] * x[i];
        f = f + a * a + c::<T>(100.) * b * b;
        g[i] = g[i] - c::<T>(2.) * a - c::<T>(400.) * x[i] * b;
        g[i + 1] = g[i + 1] + c::<T>(200.) * b;
    }
    Ok(f)
}

fn rosen_matrix<T: Float + FromPrimitive + std::fmt::Debug>() {
    for n in [2, 5, 10] {
        for memory in [3, 10, 17] {
            for start in [-1.2, 0., 2.] {
                let r = minimize(
                    vec![c::<T>(start); n],
                    vec![Bound::both(c(-3.), c(3.)); n],
                    Options {
                        memory,
                        pgtol: T::epsilon().sqrt() * c(10.),
                        max_iterations: 3000,
                        max_evaluations: 30000,
                        ..Options::default()
                    },
                    rosen,
                )
                .unwrap();
                assert!(
                    is_success(r.status),
                    "n={n} m={memory} start={start}: {r:?}"
                );
                assert!(
                    r.f < c::<T>(if T::epsilon() > c(1e-10) { 1e-5 } else { 1e-12 }),
                    "n={n} m={memory} start={start}: {r:?}"
                );
                assert!(
                    r.x.iter().all(|&v| (v - T::one()).abs()
                        < c::<T>(if T::epsilon() > c(1e-10) { 5e-3 } else { 1e-5 })),
                    "n={n} m={memory} start={start}: {r:?}"
                );
            }
        }
    }
}

#[test]
fn rosenbrock_multistart_f64() {
    rosen_matrix::<f64>();
}
#[test]
fn rosenbrock_multistart_f32() {
    rosen_matrix::<f32>();
}

#[test]
fn rosenbrock_active_upper_bound_matches_reference() {
    for start in [vec![-1.2, 1.], vec![0., 0.], vec![5., -5.]] {
        let bounds = vec![Bound::both(-2., 0.8), Bound::unbounded()];
        let r = minimize(
            start.clone(),
            bounds.clone(),
            Options {
                pgtol: 1e-7,
                ..Options::default()
            },
            rosen::<f64>,
        )
        .unwrap();
        assert!(is_success(r.status), "{r:?}");
        assert!(
            (r.x[0] - 0.8).abs() < 1e-8 && (r.x[1] - 0.64).abs() < 1e-6,
            "{r:?}"
        );
        assert!((r.f - 0.04).abs() < 1e-10, "{r:?}");
        let mut problem = lbfgsb::LbfgsbProblem::build(start, rosen::<f64>);
        problem.set_bounds(bounds.iter().map(|b| (b.lower, b.upper)));
        let mut reference = lbfgsb::LbfgsbState::new(
            problem,
            lbfgsb::LbfgsbParameter {
                pgtol: 1e-7,
                factr: 0.,
                ..Default::default()
            },
        );
        {
            let _guard = REFERENCE_LOCK.lock().unwrap();
            reference.minimize().unwrap();
        }
        assert!((r.f - reference.fx()).abs() < 1e-10);
    }
}

#[test]
fn convex_exponential_with_mixed_bounds_and_multiple_scales() {
    for scale in [1e-4, 1., 1e4] {
        for start in [-6., 0., 6.] {
            let a = [0.1, 0.5, 2., 10.];
            let bounds = vec![
                Bound::unbounded(),
                Bound::lower(0.),
                Bound::upper(0.),
                Bound::both(-1., 1.),
            ];
            let r = minimize(
                vec![start; 4],
                bounds.clone(),
                Options {
                    pgtol: scale * 1e-7,
                    max_iterations: 3000,
                    ..Options::default()
                },
                |x, g| {
                    let mut f = 0.;
                    for i in 0..4 {
                        g[i] = scale * (x[i].exp() - a[i]);
                        f += scale * (x[i].exp() - a[i] * x[i]);
                    }
                    Ok(f)
                },
            )
            .unwrap();
            assert!(is_success(r.status), "scale={scale} start={start}: {r:?}");
            for i in 0..4 {
                assert!(
                    (r.x[i] - bounds[i].project(a[i].ln())).abs() < 1e-5,
                    "scale={scale} start={start}: {r:?}"
                );
            }
        }
    }
}

#[test]
fn ill_conditioned_quadratic() {
    // Eight orders of magnitude between largest and smallest eigenvalues.
    for memory in [5, 17] {
        let weights: Vec<f64> = (0..10)
            .map(|i| 10_f64.powf(-4. + 8. * i as f64 / 9.))
            .collect();
        let r = minimize(
            vec![0.; 10],
            vec![Bound::unbounded(); 10],
            Options {
                memory,
                pgtol: 1e-7,
                // Small memories need many steps for condition number 1e8.
                max_iterations: 20000,
                max_evaluations: 100000,
                ..Options::default()
            },
            |x, g| {
                let mut f = 0.;
                for i in 0..10 {
                    let d = x[i] - 1.;
                    g[i] = weights[i] * d;
                    f += 0.5 * weights[i] * d * d;
                }
                Ok(f)
            },
        )
        .unwrap();
        assert!(is_success(r.status), "m={memory}: {r:?}");
        assert!(
            r.x.iter().all(|&v| (v - 1.).abs() < 2e-3),
            "m={memory}: {r:?}"
        );
        assert!(r.f < 1e-9, "m={memory}: {r:?}");
    }
}

#[test]
fn seeded_dense_spd_quadratics_match_reference() {
    let mut seed = 123456789_u64;
    let mut rand = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 32) as f64 / u32::MAX as f64) * 2. - 1.
    };
    for case in 0..12 {
        let n = 2 + case % 7;
        let a: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| rand()).collect()).collect();
        let target: Vec<f64> = (0..n).map(|_| rand() * 3.).collect();
        let start: Vec<f64> = (0..n).map(|_| rand() * 5.).collect();
        let bounds: Vec<_> = (0..n)
            .map(|i| match (i + case) % 5 {
                0 => Bound::unbounded(),
                1 => Bound::lower(-0.5),
                2 => Bound::upper(0.5),
                3 => Bound::both(-1., 1.),
                _ => Bound::both(0., 0.),
            })
            .collect();
        let fg = |x: &[f64], g: &mut [f64]| -> anyhow::Result<f64> {
            let d: Vec<_> = x.iter().zip(&target).map(|(x, t)| x - t).collect();
            let ad: Vec<f64> = a
                .iter()
                .map(|row| row.iter().zip(&d).map(|(a, d)| a * d).sum())
                .collect();
            for j in 0..n {
                g[j] = 0.1 * d[j] + a.iter().zip(&ad).map(|(row, v)| row[j] * v).sum::<f64>();
            }
            Ok(0.5
                * (ad.iter().map(|v| v * v).sum::<f64>()
                    + 0.1 * d.iter().map(|v| v * v).sum::<f64>()))
        };
        let r = minimize(
            start.clone(),
            bounds.clone(),
            Options {
                pgtol: 1e-8,
                ..Options::default()
            },
            fg,
        )
        .unwrap();
        assert!(is_success(r.status), "case={case}: {r:?}");
        let mut problem = lbfgsb::LbfgsbProblem::build(start, fg);
        problem.set_bounds(bounds.iter().map(|b| (b.lower, b.upper)));
        let mut reference = lbfgsb::LbfgsbState::new(
            problem,
            lbfgsb::LbfgsbParameter {
                m: 17,
                factr: 0.,
                pgtol: 1e-8,
                iprint: -1,
            },
        );
        {
            let _guard = REFERENCE_LOCK.lock().unwrap();
            reference.minimize().unwrap();
        }
        assert!(
            (r.f - reference.fx()).abs() < 1e-9,
            "case={case}: {r:?} ref={}",
            reference.fx()
        );
        for (&x, &y) in r.x.iter().zip(reference.x()) {
            assert!(
                (x - y).abs() < 2e-5,
                "case={case}: {r:?} ref={:?}",
                reference.x()
            );
        }
    }
}
