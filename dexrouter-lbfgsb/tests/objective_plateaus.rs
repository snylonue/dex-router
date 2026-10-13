use dexrouter_lbfgsb::{Bound, Options, Status, minimize};

#[test]
fn plateau_derivative_interpolation_resolves_extremely_short_steps_f64() {
    for h in [1e4_f64, 1e12, 1e24, 1e100] {
        let result = minimize(
            vec![0.01 / h],
            vec![Bound::both(-1., 1.)],
            Options {
                factr: 0.,
                pgtol: 1e-9,
                max_iterations: 50,
                max_evaluations: 1000,
                wolfe_c1: 0.001,
                ..Options::default()
            },
            |x, g| {
                g[0] = h * x[0];
                Ok(1e200 + 0.5 * h * x[0] * x[0])
            },
        )
        .unwrap();
        assert_eq!(
            result.status,
            Status::ConvergedProjectedGradient,
            "h={h}, {result:?}"
        );
        assert!(result.gradient[0].abs() <= 1e-9);
        assert!(result.evaluations <= 4);
    }
}

#[test]
fn plateau_derivative_interpolation_resolves_extremely_short_steps_f32() {
    for h in [1e4_f32, 1e12, 1e24, 1e30] {
        let result = minimize(
            vec![0.01 / h],
            vec![Bound::both(-1., 1.)],
            Options {
                factr: 0.,
                pgtol: 1e-5,
                max_iterations: 50,
                max_evaluations: 1000,
                wolfe_c1: 0.001,
                ..Options::default()
            },
            |x, g| {
                g[0] = h * x[0];
                Ok(1e37 + 0.5 * h * x[0] * x[0])
            },
        )
        .unwrap();
        assert_eq!(
            result.status,
            Status::ConvergedProjectedGradient,
            "h={h}, {result:?}"
        );
        assert!(result.gradient[0].abs() <= 1e-5);
        assert!(result.evaluations <= 4);
    }
}

#[test]
fn rounded_objective_offset_preserves_derivative_bracket_f64() {
    for offset in [0.0, 1e20] {
        let result = minimize(
            vec![1e-6_f64],
            vec![Bound::both(-1.0, 1.0)],
            Options {
                factr: 0.0,
                pgtol: 1e-9,
                max_iterations: 50,
                max_evaluations: 1000,
                wolfe_c1: 0.001,
                ..Options::default()
            },
            |x, g| {
                g[0] = 1e4 * x[0];
                Ok(offset + 0.5 * 1e4 * x[0] * x[0])
            },
        )
        .unwrap();
        assert_eq!(
            result.status,
            Status::ConvergedProjectedGradient,
            "offset={offset}, result={result:?}"
        );
        assert!(result.gradient[0].abs() <= 1e-9);
        assert!(result.x[0].abs() <= 1e-13);
        assert!(result.evaluations <= 8, "must not exhaust the zoom budget");
    }
}

#[test]
fn rounded_objective_offset_preserves_derivative_bracket_f32() {
    for offset in [0.0, 1e8] {
        let result = minimize(
            vec![1e-6_f32],
            vec![Bound::both(-1.0, 1.0)],
            Options {
                factr: 0.0,
                pgtol: 1e-5,
                max_iterations: 50,
                max_evaluations: 1000,
                wolfe_c1: 0.001,
                ..Options::default()
            },
            |x, g| {
                g[0] = 1e4 * x[0];
                Ok(offset + 0.5 * 1e4 * x[0] * x[0])
            },
        )
        .unwrap();
        assert_eq!(
            result.status,
            Status::ConvergedProjectedGradient,
            "offset={offset}, result={result:?}"
        );
        assert!(result.gradient[0].abs() <= 1e-5);
        assert!(result.x[0].abs() <= 1e-9);
        assert!(result.evaluations <= 8, "must not exhaust the zoom budget");
    }
}
