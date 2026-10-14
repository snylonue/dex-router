//! Literal witnesses from the post-5690661 discovery campaign.
use dexrouter_lbfgsb::{Bound, Options, Status, minimize};

#[test]
fn resolved_small_curvature_must_not_exhaust_unbracketed_expansion() {
    let scale = 2_f64.powi(-30);
    let mut calls = 0;
    let r = minimize(
        vec![-0.25],
        vec![Bound::upper(0.5)],
        Options {
            memory: 1,
            pgtol: 0.25e-7 * scale,
            ..Options::default()
        },
        |x, g| {
            calls += 1;
            let mut f = 0.5 * 0.125 * x[0] * x[0];
            g[0] = 0.125 * x[0];
            for (a, h) in [(-0.25, 0.125), (0.25, 2048.)] {
                let q = a * x[0];
                f += h * (0.5 * q * q);
                g[0] += a * h * q;
            }
            g[0] *= scale;
            Ok(scale * f)
        },
    )
    .unwrap();
    let exact_gradient = 128.1328125 * r.x[0];
    let residual = (r.x[0] - (r.x[0] - exact_gradient).min(0.5)).abs();
    assert_eq!(calls, r.evaluations);
    assert!(residual <= 1e-7, "residual={residual:e}, {r:?}");
    assert_eq!(r.status, Status::ConvergedProjectedGradient, "{r:?}");
    eprintln!("independent residual={residual:e}, calls={calls}, {r:?}");
    assert!(calls <= 21, "{r:?}");
}

fn quadratic_f32(
    start: Vec<f32>,
    target: Vec<f32>,
    bounds: Vec<Bound<f32>>,
    normal: Vec<f32>,
    terms: Vec<(Vec<f32>, f32)>,
    mu: f32,
    scale: f32,
) {
    let mut calls = 0;
    let r = minimize(
        start,
        bounds.clone(),
        Options {
            memory: 1,
            pgtol: 0.25e-5 * scale.min(1.),
            ..Options::default()
        },
        |x, g| {
            calls += 1;
            let d: Vec<_> = x.iter().zip(&target).map(|(x, t)| x - t).collect();
            let mut f = 0.;
            for i in 0..x.len() {
                g[i] = mu * d[i] + normal[i];
                f = f + 0.5 * mu * d[i] * d[i] + normal[i] * d[i];
            }
            for (row, h) in &terms {
                let q = row.iter().zip(&d).fold(0., |s, (a, d)| s + a * d);
                f += h * (0.5 * q * q);
                for (gi, a) in g.iter_mut().zip(row) {
                    *gi += a * h * q;
                }
            }
            for gi in g {
                *gi *= scale;
            }
            Ok(scale * f)
        },
    )
    .unwrap();
    let d: Vec<_> =
        r.x.iter()
            .zip(&target)
            .map(|(x, t)| f64::from(*x) - f64::from(*t))
            .collect();
    let mut gradient: Vec<_> = d
        .iter()
        .zip(&normal)
        .map(|(d, n)| f64::from(mu) * d + f64::from(*n))
        .collect();
    for (row, h) in &terms {
        let q = row
            .iter()
            .zip(&d)
            .map(|(a, d)| f64::from(*a) * d)
            .sum::<f64>();
        for (gi, a) in gradient.iter_mut().zip(row) {
            *gi += f64::from(*a) * f64::from(*h) * q;
        }
    }
    let residual =
        r.x.iter()
            .zip(gradient)
            .zip(&bounds)
            .map(|((x, g), b)| {
                let x = f64::from(*x);
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
    eprintln!("independent residual={residual:e}, calls={calls}, {r:?}");
    assert_eq!(calls, r.evaluations);
    assert!(residual <= 1e-5, "residual={residual:e}, {r:?}");
    assert_eq!(r.status, Status::ConvergedProjectedGradient, "{r:?}");
    assert!(calls <= 41, "{r:?}");
}

#[test]
fn large_objective_scale_must_not_destroy_a_bounded_quadratic_model() {
    quadratic_f32(
        vec![-6.5, 0.25],
        vec![0.875, -2.875],
        vec![Bound::upper(0.875), Bound::unbounded()],
        vec![-2., 0.],
        vec![
            (vec![0., -0.25], 128.),
            (vec![-0.5, 0.], 256.),
            (vec![0.25, -0.25], 512.),
        ],
        1.,
        2_f32.powi(60),
    );
}

#[test]
fn bounded_coupled_quadratic_must_reach_the_independent_f32_goal() {
    quadratic_f32(
        vec![-6.75, 6.5, -6.5],
        vec![2.375, 0.375, 2.875],
        vec![Bound::upper(3.125), Bound::upper(0.375), Bound::unbounded()],
        vec![0., -1., 0.],
        vec![
            (vec![0.25, -0.25, 0.], 0.5),
            (vec![0., 0., 0.], 1.),
            (vec![0.25, -0.5, -0.5], 1024.),
        ],
        0.5,
        2_f32.powi(-11),
    );
}
