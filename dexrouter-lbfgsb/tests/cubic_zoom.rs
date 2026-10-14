//! C1 convex activation walls: a bracket can straddle two different quadratics.
use dexrouter_lbfgsb::{Bound, Options, minimize};

macro_rules! activation_wall {
    ($name:ident, $float:ty, $exponents:expr) => {
        #[test]
        fn $name() {
            for exponent in $exponents {
                for scale_exponent in [-40, 0, 40] {
                    for sign in [-1., 1.] {
                        let stiffness = (2 as $float).powi(exponent);
                        let scale = (2 as $float).powi(scale_exponent);
                        let sign: $float = sign;
                        let mut calls = 0;
                        let result = minimize(
                            vec![0 as $float],
                            vec![Bound::unbounded()],
                            Options {
                                max_iterations: 1,
                                wolfe_c1: 0.001,
                                pgtol: scale * 1e-8,
                                factr: 0.,
                                ..Options::default()
                            },
                            |x, g| {
                                calls += 1;
                                let t = sign * x[0];
                                let active = (t - 0.0625).max(0.);
                                g[0] = sign * scale * (-1. + 2. * stiffness * active);
                                Ok(scale * (-t + stiffness * active * active))
                            },
                        )
                        .unwrap();
                        let t = f64::from(sign * result.x[0]);
                        let k = f64::from(stiffness);
                        let independent_gradient = -1. + 2. * k * (t - 0.0625).max(0.);
                        assert_eq!(calls, result.evaluations);
                        assert!(calls <= 21, "{result:?}");
                        assert_eq!(
                            result.iterations, 1,
                            "k=2^{exponent}, scale=2^{scale_exponent}, {result:?}"
                        );
                        assert!(
                            independent_gradient.abs() <= 0.9,
                            "independent g={independent_gradient}: {result:?}"
                        );
                        // Check the original Armijo/Wolfe contracts independently,
                        // not a special early-exit status or a relaxed goal.
                        let independent_value = -t + k * (t - 0.0625).max(0.).powi(2);
                        assert!(independent_value <= -0.001 * t, "{result:?}");
                        assert!(
                            f64::from(result.gradient[0] / scale).abs() <= 0.9,
                            "{result:?}"
                        );
                    }
                }
            }
        }
    };
}
activation_wall!(
    activation_wall_default_budget_f64,
    f64,
    [14, 20, 24, 30, 40]
);
activation_wall!(activation_wall_default_budget_f32, f32, [14, 20, 24]);
