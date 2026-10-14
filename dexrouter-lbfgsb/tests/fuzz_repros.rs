//! Literal reduced witnesses, independent of future proptest shrink ordering.
use dexrouter_lbfgsb::{Bound, Options, Status, minimize};

#[test]
fn small_objective_scale_must_not_make_first_trial_stationary() {
    let scale = 2_f32.powi(-30);
    for start in [0_f32, 0.25, -0.25, 1., -1.] {
        let target = if start == 0. { 1. } else { 0. };
        let mut calls = 0;
        let r = minimize(
            vec![start],
            vec![Bound::unbounded()],
            Options {
                memory: 1,
                pgtol: 0.25e-5 * scale,
                ..Options::default()
            },
            |x, g| {
                calls += 1;
                let d = x[0] - target;
                g[0] = scale * d;
                Ok(0.5 * (scale * d) * d)
            },
        )
        .unwrap();
        assert_eq!(r.status, Status::ConvergedProjectedGradient, "{r:?}");
        assert!((r.x[0] - target).abs() <= 1e-5, "{r:?}");
        assert_eq!(calls, r.evaluations);
        assert!(calls <= 4, "{r:?}");
    }
}

// Previously default20 failed (residual 1.82e-4, 187 callbacks), whereas an
// explicit80 reached the independent goal. Same-side curvature rescue must now
// reach that original goal with BOTH budgets, without raising the default20.
#[test]
fn reduced_wide_scale_wall_converges_with_the_default_search_budget() {
    let scale = 2_f32.powi(-90);
    for budget in [20, 80] {
        let mut calls = 0;
        let r = minimize(
            vec![0.25_f32],
            vec![Bound::unbounded()],
            Options {
                memory: 1,
                pgtol: 0.25e-5 * scale,
                max_line_search_steps: budget,
                ..Options::default()
            },
            |x, g| {
                calls += 1;
                let d = x[0] + 2.875;
                let a = (0.5 * d).max(0.);
                g[0] = scale * (2. * d + 512. * a);
                Ok(scale * (d * d + 512. * a * a))
            },
        )
        .unwrap();
        let d = f64::from(r.x[0]) + 2.875;
        let residual = (2. * d + 512. * (0.5 * d).max(0.)).abs();
        eprintln!(
            "search_budget={budget}, residual={residual}, status={:?}, evaluations={calls}",
            r.status
        );
        assert_eq!(calls, r.evaluations);
        assert!(residual <= 1e-5, "budget={budget}, {r:?}");
        assert_eq!(r.status, Status::ConvergedProjectedGradient, "{r:?}");
        assert!(calls <= 41, "budget={budget}, {r:?}");
    }
}

#[test]
fn tiny_scale_wall_zoom_can_recover_from_a_rounded_stationary_trial() {
    let scale = 2_f32.powi(-90);
    let r = minimize(
        vec![0.25_f32],
        vec![Bound::unbounded()],
        Options {
            memory: 1,
            pgtol: 0.25e-5 * scale,
            ..Options::default()
        },
        |x, g| {
            let d = x[0] + 0.625;
            let a = (-0.5 * d).max(0.);
            g[0] = scale * (32. * d - 1024. * a);
            Ok(scale * (16. * d * d + 1024. * a * a))
        },
    )
    .unwrap();
    let d = f64::from(r.x[0]) + 0.625;
    let residual = (32. * d - 1024. * (-0.5 * d).max(0.)).abs();
    assert!(residual <= 1e-5, "independent residual={residual}: {r:?}");
}

#[test]
fn scaled_activation_wall_f32() {
    let scale = 2_f32.powi(80);
    let mut trace = Vec::new();
    let r = minimize(
        vec![0.25_f32],
        vec![Bound::unbounded()],
        Options {
            memory: 1,
            pgtol: 0.25e-5,
            ..Options::default()
        },
        |x, g| {
            let d = x[0] + 0.625;
            let a = (-0.25 * d).max(0.);
            g[0] = scale * (d / 32. - 64. * a);
            let f = scale * (d * d / 64. + 128. * a * a);
            trace.push((x[0], f, g[0]));
            Ok(f)
        },
    )
    .unwrap_or_else(|error| panic!("{error}: trace={trace:?}"));
    if (r.gradient[0] / scale).abs() > 1e-5 {
        eprintln!(
            "result={r:?}, normalized_gradient={}, trace={trace:?}",
            r.gradient[0] / scale
        );
    }
    assert!((r.gradient[0] / scale).abs() <= 1e-5, "{r:?}");
}
