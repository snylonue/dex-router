//! Followup witnesses: trial-slope overflow and a representable restart ray.
use dexrouter_lbfgsb::{Bound, Options, Status, minimize};

struct Term {
    row: Vec<f32>,
    weight: f32,
    shift: Option<f32>,
}
struct Fixture {
    start: Vec<f32>,
    target: Vec<f32>,
    bounds: Vec<Bound<f32>>,
    normal: Vec<f32>,
    terms: Vec<Term>,
    mu: f32,
    scale: f32,
    memory: usize,
    max_calls: usize,
}
fn term(row: &[f32], weight: f32, shift: Option<f32>) -> Term {
    Term {
        row: row.to_vec(),
        weight,
        shift,
    }
}
fn check(case: Fixture) {
    let mut calls = 0;
    let r = minimize(
        case.start.clone(),
        case.bounds.clone(),
        Options {
            memory: case.memory,
            pgtol: 0.25e-5 * case.scale.min(1.),
            ..Options::default()
        },
        |x, g| {
            calls += 1;
            assert!(calls <= 10_000);
            let d: Vec<_> = x.iter().zip(&case.target).map(|(x, t)| x - t).collect();
            let mut f = 0.;
            for i in 0..x.len() {
                assert_eq!(case.bounds[i].project(x[i]), x[i]);
                g[i] = case.mu * d[i] + case.normal[i];
                f = f + 0.5 * case.mu * d[i] * d[i] + case.normal[i] * d[i];
            }
            for term in &case.terms {
                let q = term.row.iter().zip(&d).fold(0., |s, (a, d)| s + a * d);
                let (value, derivative) = match term.shift {
                    None => (0.5 * q * q, q),
                    Some(t) if t >= 0. && t + q >= 0. => (0.5 * q * q, q),
                    Some(t) if t <= 0. && t + q <= 0. => (0., 0.),
                    Some(t) if t < 0. => (0.5 * (t + q) * (t + q), t + q),
                    Some(t) => (t * (-q - 0.5 * t), -t),
                };
                f += term.weight * value;
                for (gi, a) in g.iter_mut().zip(&term.row) {
                    *gi += a * term.weight * derivative;
                }
            }
            for gi in g {
                *gi *= case.scale;
                assert!(gi.is_finite());
            }
            let f = case.scale * f;
            assert!(f.is_finite());
            Ok(f)
        },
    )
    .unwrap();
    // Independent uncentred f64 derivative; never use the reported PG as oracle.
    let point: Vec<_> = r.x.iter().map(|x| f64::from(*x)).collect();
    let mut gradient: Vec<_> = point
        .iter()
        .zip(&case.target)
        .zip(&case.normal)
        .map(|((x, t), n)| f64::from(case.mu) * (x - f64::from(*t)) + f64::from(*n))
        .collect();
    for term in &case.terms {
        let q = term
            .row
            .iter()
            .zip(&point)
            .map(|(a, x)| f64::from(*a) * x)
            .sum::<f64>()
            - term
                .row
                .iter()
                .zip(&case.target)
                .map(|(a, t)| f64::from(*a) * f64::from(*t))
                .sum::<f64>();
        let derivative = term
            .shift
            .map_or(q, |t| (f64::from(t) + q).max(0.) - f64::from(t).max(0.));
        for (gi, a) in gradient.iter_mut().zip(&term.row) {
            *gi += f64::from(*a) * f64::from(term.weight) * derivative;
        }
    }
    let residual = point
        .iter()
        .zip(gradient)
        .zip(&case.bounds)
        .map(|((&x, g), b)| {
            let mut p = x - g;
            if let Some(l) = b.lower {
                p = p.max(f64::from(l));
            }
            if let Some(u) = b.upper {
                p = p.min(f64::from(u));
            }
            (x - p).abs()
        })
        .fold(0_f64, f64::max);
    eprintln!(
        "calls={calls}, residual={residual:e}, status={:?}",
        r.status
    );
    assert_eq!(calls, r.evaluations);
    assert!(residual <= 1e-5, "{residual:e}, {r:?}");
    assert_eq!(r.status, Status::ConvergedProjectedGradient, "{r:?}");
    assert!(calls <= case.max_calls, "{r:?}");
}

#[test]
fn diagonal_restart_preserves_progress_near_the_f32_lattice_with_fixed_variables() {
    check(Fixture {
        start: vec![4.25, -1.25, 4.5, 5.5, 2.5, -6.75, -6.25],
        target: vec![1.875, 0.625, 2.75, 0.875, -0.625, 2.125, 2.875],
        bounds: vec![
            Bound::upper(3.875),
            Bound::upper(0.625),
            Bound::unbounded(),
            Bound::upper(0.875),
            Bound::both(-0.625, -0.625),
            Bound::upper(2.75),
            Bound::unbounded(),
        ],
        normal: vec![0., -1.75, 0., -1.25, 0.5, 0., 0.],
        mu: 0.5,
        scale: 32.,
        memory: 3,
        max_calls: 180,
        terms: vec![
            term(&[0., 0.25, 0., 0.25, 0., 0., -0.25], 1., None),
            term(&[0.25, 0., -0.5, 0.5, 0., 0., 0.], 2048., None),
            term(&[0., 0., 0., 0.5, 0., -0.5, -0.5], 2_f32.powi(-6), None),
        ],
    });
}

#[test]
fn finite_initial_slope_must_leave_headroom_for_trial_derivatives() {
    check(Fixture {
        start: vec![-6.75, -4.5, 4.5, 7.25, 0.25, -6.5, 0.25],
        target: vec![1.125, 0.875, 3., 1.25, -1.5, 0.875, -2.875],
        bounds: vec![
            Bound::both(1.125, 1.75),
            Bound::unbounded(),
            Bound::both(3., 3.25),
            Bound::both(-0.75, 1.25),
            Bound::upper(-1.5),
            Bound::upper(0.875),
            Bound::unbounded(),
        ],
        normal: vec![0.75, 0., 1., -1., -2., -1., 0.],
        mu: 1.,
        scale: 2_f32.powi(70),
        memory: 1,
        max_calls: 600,
        terms: vec![
            term(
                &[0.5, 0.25, -0.5, -0.25, 0., -0.25, -0.5],
                1024.,
                Some(0.25),
            ),
            term(&[0., 0.5, 0.25, -0.5, -0.25, 0., -0.25], 1., None),
            term(&[0., 0., 0.5, -0.25, 0.5, 0., 0.], 1., None),
            term(&[0.; 7], 1., None),
            term(&[0., 0., 0., 0., 0., 0.25, 0.], 2_f32.powi(-6), None),
        ],
    });
}
