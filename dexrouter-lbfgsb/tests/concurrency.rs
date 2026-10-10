use dexrouter_lbfgsb::{Bound, OptimizeResult, Options, Status, minimize};
use num_traits::{Float, FromPrimitive};
use std::sync::Barrier;

const THREADS: usize = 8;
const CASES: usize = 32;

fn c<T: Float + FromPrimitive>(value: f64) -> T {
    T::from_f64(value).unwrap()
}

fn solve<T: Float + FromPrimitive + std::fmt::Debug>(
    case: usize,
    barrier: Option<&Barrier>,
) -> OptimizeResult<T> {
    let n = [2, 5, 10, 32][case % 4];
    let bounds: Vec<_> = (0..n)
        .map(|i| match (i + case) % 5 {
            0 => Bound::unbounded(),
            1 => Bound::lower(c(-0.5)),
            2 => Bound::upper(c(0.5)),
            3 => Bound::both(c(-1.0), c(1.0)),
            _ => Bound::both(c(0.25), c(0.25)),
        })
        .collect();
    let target: Vec<_> = (0..n)
        .map(|i| c::<T>((i as f64 - n as f64 * 0.5) / 3.0 + case as f64 / 16.0))
        .collect();
    let mut calls = 0;
    let result = minimize(
        vec![c::<T>(3.0); n],
        bounds.clone(),
        Options {
            memory: [3, 10, 17][case % 3],
            pgtol: if T::epsilon() > c(1e-10) {
                c(1e-4)
            } else {
                c(1e-7)
            },
            ..Options::default()
        },
        |x, g| {
            // Synchronize inside each solve, not merely at thread startup.
            // All workers are simultaneously in the optimizer's callback.
            if calls == 0 {
                if let Some(barrier) = barrier {
                    barrier.wait();
                }
            }
            calls += 1;
            if barrier.is_some() {
                std::thread::yield_now();
            }
            let mut f = T::zero();
            for i in 0..n {
                let d = x[i] - target[i];
                let weight = c::<T>(1.0 + (i % 7) as f64);
                g[i] = weight * d;
                // Subtract the constrained minimum algebraically to avoid a
                // large constant objective floor masking f32 improvements.
                let optimum = bounds[i].project(target[i]);
                f = f + c::<T>(0.5) * weight * (x[i] - optimum) * (d + optimum - target[i]);
            }
            Ok(f)
        },
    )
    .unwrap();
    assert_eq!(result.evaluations, calls);
    assert!(
        matches!(
            result.status,
            Status::ConvergedProjectedGradient | Status::ConvergedFunctionReduction
        ),
        "case={case}: {result:?}"
    );
    let tolerance = if T::epsilon() > c(1e-10) {
        c(2e-4)
    } else {
        c(1e-6)
    };
    for ((&x, &target), bound) in result.x.iter().zip(&target).zip(&bounds) {
        assert!(
            (x - bound.project(target)).abs() <= tolerance,
            "case={case}: {result:?}"
        );
    }
    result
}

fn concurrent_matches_serial<T: Float + FromPrimitive + std::fmt::Debug + Send + Sync>() {
    let expected: Vec<_> = (0..CASES).map(|case| solve::<T>(case, None)).collect();
    let barrier = Barrier::new(THREADS);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                scope.spawn(|| {
                    for (case, expected) in expected.iter().enumerate() {
                        let actual = solve::<T>(case, Some(&barrier));
                        assert_eq!(actual.x, expected.x);
                        assert_eq!(actual.f, expected.f);
                        assert_eq!(actual.gradient, expected.gradient);
                        assert_eq!(actual.status, expected.status);
                        assert_eq!(actual.iterations, expected.iterations);
                        assert_eq!(actual.evaluations, expected.evaluations);
                        assert_eq!(
                            actual.projected_gradient_inf_norm,
                            expected.projected_gradient_inf_norm
                        );
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    });
}

#[test]
fn concurrent_f64_solves_match_serial_results() {
    concurrent_matches_serial::<f64>();
}

#[test]
fn concurrent_f32_solves_match_serial_results() {
    concurrent_matches_serial::<f32>();
}
