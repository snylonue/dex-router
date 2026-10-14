//! Regressions for finite gradients whose unscaled directional dot product
//! underflows to zero or overflows, despite a well-conditioned objective.
use dexrouter_lbfgsb::{Bound, Options, Status, minimize};
use num_traits::{Float, FromPrimitive};

fn check<T: Float + FromPrimitive + std::fmt::Debug>(exponents: &[i32]) {
    let c = |v| T::from_f64(v).unwrap();
    let goal = if T::epsilon() > c(1e-10) {
        c(1e-5)
    } else {
        c(1e-9)
    };
    for &exponent in exponents {
        let scale = c(2_f64.powi(exponent));
        for sign in [-T::one(), T::one()] {
            for bound in [
                Bound::unbounded(),
                Bound::lower(c(-2.)),
                Bound::upper(c(2.)),
                Bound::both(c(-2.), c(2.)),
            ] {
                let mut calls = 0;
                let options = Options {
                    pgtol: goal * scale.min(T::one()),
                    max_iterations: 20,
                    max_evaluations: 100,
                    ..Options::default()
                };
                let r = minimize(vec![sign], vec![bound], options, |x, g| {
                    calls += 1;
                    assert!(x[0].is_finite());
                    if let Some(l) = bound.lower {
                        assert!(x[0] >= l);
                    }
                    if let Some(u) = bound.upper {
                        assert!(x[0] <= u);
                    }
                    g[0] = scale * x[0];
                    let f = c(0.5) * (scale * x[0]) * x[0];
                    assert!(f.is_finite() && g[0].is_finite());
                    Ok(f)
                })
                .unwrap();
                assert_eq!(
                    r.status,
                    Status::ConvergedProjectedGradient,
                    "exponent={exponent}, bound={bound:?}: {r:?}"
                );
                assert!(r.x[0].abs() <= goal, "independent normalized KKT: {r:?}");
                assert_eq!(calls, r.evaluations);
                assert!(
                    calls <= 4,
                    "condition-one quadratic should not consume the trial budget: {r:?}"
                );
            }
        }
    }
}

fn overflowing_unit_slope<T: Float + FromPrimitive + std::fmt::Debug>(exponent: i32) {
    let c = |v| T::from_f64(v).unwrap();
    let scale = c(2_f64.powi(exponent));
    let start = (0..8)
        .map(|i| if i % 2 == 0 { c(0.5) } else { c(-0.5) })
        .collect();
    let mut calls = 0;
    let r = minimize(
        start,
        vec![Bound::unbounded(); 8],
        Options::default(),
        |x, g| {
            calls += 1;
            let mut f = T::zero();
            for (x, g) in x.iter().zip(g) {
                *g = scale * *x;
                f = f + c(0.5) * (scale * *x) * *x;
                assert!(g.is_finite());
            }
            assert!(f.is_finite());
            Ok(f)
        },
    )
    .unwrap();
    assert_eq!(r.status, Status::ConvergedProjectedGradient, "{r:?}");
    assert!(r.x.iter().all(|x| *x == T::zero()), "{r:?}");
    assert_eq!(r.evaluations, calls);
    assert_eq!(calls, 2);
}

#[test]
fn sum_of_unit_direction_products_can_overflow_f32() {
    overflowing_unit_slope::<f32>(126);
}
#[test]
fn sum_of_unit_direction_products_can_overflow_f64() {
    overflowing_unit_slope::<f64>(1022);
}

#[test]
fn finite_tiny_and_huge_gradient_f32() {
    check::<f32>(&[-120, -100, -80, -30, -22, 0, 80, 100, 120]);
}
#[test]
fn finite_tiny_and_huge_gradient_f64() {
    check::<f64>(&[-600, -540, -60, -52, 0, 540, 600]);
}
