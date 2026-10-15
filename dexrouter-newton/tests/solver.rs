use dexrouter_newton::{
    Bound, Evaluation, Hessian, Options, Status, minimize, projected_gradient_norm,
};
use std::{cell::Cell, rc::Rc};

#[derive(Clone)]
struct Dense(Vec<Vec<f64>>);
impl Hessian for Dense {
    fn apply(&self, v: &[f64], out: &mut [f64]) -> anyhow::Result<()> {
        for (o, row) in out.iter_mut().zip(&self.0) {
            *o = row.iter().zip(v).map(|(a, b)| a * b).sum();
        }
        Ok(())
    }
    fn diagonal(&self, out: &mut [f64]) -> anyhow::Result<()> {
        for (i, v) in out.iter_mut().enumerate() {
            *v = self.0[i][i];
        }
        Ok(())
    }
}
fn oracle(
    h: &Dense,
    target: &[f64],
    linear: &[f64],
    offset: f64,
    x: &[f64],
    g: &mut [f64],
    reference: Option<&[f64]>,
) -> Evaluation {
    let d: Vec<_> = x.iter().zip(target).map(|(x, t)| x - t).collect();
    h.apply(&d, g).unwrap();
    let value = offset
        + d.iter()
            .zip(g.iter())
            .map(|(x, g)| 0.5 * x * g)
            .sum::<f64>()
        + linear.iter().zip(&d).map(|(a, b)| a * b).sum::<f64>();
    for (g, l) in g.iter_mut().zip(linear) {
        *g += l;
    }
    let difference = reference.map_or(0.0, |r| {
        let delta: Vec<_> = x.iter().zip(r).map(|(x, r)| x - r).collect();
        let mid: Vec<_> = x
            .iter()
            .zip(r)
            .zip(target)
            .map(|((x, r), t)| 0.5 * (x + r) - t)
            .collect();
        let mut gm = vec![0.0; x.len()];
        h.apply(&mid, &mut gm).unwrap();
        delta
            .iter()
            .zip(gm)
            .zip(linear)
            .map(|((d, g), l)| d * (g + l))
            .sum()
    });
    Evaluation::with_difference(value, difference)
}

#[test]
fn normalized_hvp_avoids_intermediate_overflow() {
    struct Factored;
    impl Hessian for Factored {
        fn apply(&self, v: &[f64], out: &mut [f64]) -> anyhow::Result<()> {
            assert!(v[0].abs() <= 1.0);
            out[0] = 1e-300 * (1e200 * (1e100 * v[0]));
            Ok(())
        }
    }
    let r = minimize(
        vec![0.0],
        vec![Bound::unbounded()],
        Options {
            pgtol: 1e85,
            ..Default::default()
        },
        |x, g, _| {
            g[0] = x[0] - 1e100;
            Ok(Evaluation::value(0.5 * g[0] * g[0]))
        },
        |_| Ok(Factored),
    )
    .unwrap();
    assert_eq!(r.status, Status::ConvergedProjectedGradient);
    assert!((r.x[0] / 1e100 - 1.0).abs() < 1e-14);
}

#[test]
fn reference_rebasing_and_nonfinite_domain_backtracking() {
    let mut last = vec![100.0];
    let mut origin = last.clone();
    let mut calls = 0;
    let mut invalid = 0;
    let r = minimize(
        last.clone(),
        vec![Bound::unbounded()],
        Options::default(),
        |x, g, reference| {
            if let Some(reference) = reference {
                if reference != origin {
                    assert_eq!(reference, last);
                    origin = reference.to_vec();
                }
            } else {
                assert_eq!(calls, 0);
            }
            calls += 1;
            last = x.to_vec();
            if x[0] <= 0.0 {
                invalid += 1;
                g[0] = f64::NAN;
                return Ok(Evaluation::value(f64::NAN));
            }
            g[0] = 1.0 - 10.0 / x[0];
            let value = x[0] - 10.0 * x[0].ln();
            Ok(Evaluation::with_difference(
                value,
                reference.map_or(0.0, |r| {
                    let d = x[0] - r[0];
                    d - 10.0 * (d / r[0]).ln_1p()
                }),
            ))
        },
        |x| Ok(Dense(vec![vec![10.0 / (x[0] * x[0])]])),
    )
    .unwrap();
    assert_eq!(calls, r.evaluations);
    assert!(invalid > 0);
    assert!((r.x[0] - 10.0).abs() < 1e-6);
    assert_eq!(r.status, Status::ConvergedProjectedGradient);
}

#[test]
fn singular_laplacian_and_local_metric_damping_preserve_weak_modes() {
    let h = Dense(vec![vec![1.0, -1.0], vec![-1.0, 1.0]]);
    let r = minimize(
        vec![3.0, 1.0],
        vec![Bound::unbounded(); 2],
        Options::default(),
        |x, g, r| Ok(oracle(&h, &[0.0, 0.0], &[0.0, 0.0], 0.0, x, g, r)),
        |_| Ok(h.clone()),
    )
    .unwrap();
    assert_eq!(r.status, Status::ConvergedProjectedGradient);
    assert!((r.x[0] - r.x[1]).abs() < 1e-8);
    let h = Dense(vec![vec![1e-12, 0.0], vec![0.0, 1e12]]);
    let r = minimize(
        vec![1.0, 1.0],
        vec![Bound::unbounded(); 2],
        Options {
            initial_damping: 1e-6,
            pgtol: 1e-20,
            ..Default::default()
        },
        |x, g, r| Ok(oracle(&h, &[0.0, 0.0], &[0.0, 0.0], 0.0, x, g, r)),
        |_| Ok(h.clone()),
    )
    .unwrap();
    assert_eq!(r.status, Status::ConvergedProjectedGradient);
    assert!(r.x.iter().all(|x| x.abs() < 1e-8));
}

#[test]
fn coupled_boxes_match_known_kkt_and_lbfgsb() {
    let mut seed = 7_u64;
    let mut random = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 32) as f64) / (u32::MAX as f64)
    };
    for case in 0..64 {
        let n = 2 + case % 7;
        let vector: Vec<_> = (0..n).map(|_| 2.0 * random() - 1.0).collect();
        let h = Dense(
            (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| {
                            vector[i] * vector[j]
                                + if i == j {
                                    10_f64.powi((i % 5) as i32 - 2)
                                } else {
                                    0.0
                                }
                        })
                        .collect()
                })
                .collect(),
        );
        let bounds: Vec<_> = (0..n)
            .map(|i| {
                if i == 0 && case % 2 == 0 {
                    Bound::new(0.3, 0.3)
                } else {
                    Bound::new(-1.0, 1.0)
                }
            })
            .collect();
        let target: Vec<_> = (0..n)
            .map(|i| {
                if bounds[i].lower == bounds[i].upper {
                    0.3
                } else {
                    match i % 3 {
                        0 => -1.0,
                        1 => 1.0,
                        _ => 0.2,
                    }
                }
            })
            .collect();
        let linear: Vec<_> = target
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                if bounds[i].lower == bounds[i].upper {
                    1e8
                } else if t == -1.0 {
                    0.1
                } else if t == 1.0 {
                    -0.1
                } else {
                    0.0
                }
            })
            .collect();
        let start = vec![if case % 2 == 0 { 0.0 } else { 2.0 }; n];
        let calls = Cell::new(0);
        let ours = minimize(
            start.clone(),
            bounds.clone(),
            Options::default(),
            |x, g, r| {
                calls.set(calls.get() + 1);
                Ok(oracle(&h, &target, &linear, 0.0, x, g, r))
            },
            |_| Ok(h.clone()),
        )
        .unwrap();
        let reference = dexrouter_lbfgsb::minimize_with_differences(
            start,
            bounds
                .iter()
                .map(|b| dexrouter_lbfgsb::Bound::both(b.lower, b.upper))
                .collect(),
            dexrouter_lbfgsb::Options {
                pgtol: 1e-8,
                max_iterations: 2000,
                ..Default::default()
            },
            |x, g, r| {
                let e = oracle(&h, &target, &linear, 0.0, x, g, r);
                Ok((e.value, e.difference.unwrap()))
            },
        )
        .unwrap();
        assert_eq!(ours.evaluations, calls.get());
        for (name, x) in [("newton", &ours.x), ("lbfgsb", &reference.x)] {
            let mut g = vec![0.0; n];
            oracle(&h, &target, &linear, 0.0, x, &mut g, None);
            let pg = projected_gradient_norm(x, &g, &bounds);
            assert!(
                pg < 2e-6,
                "case={case} {name} pg={pg} newton={ours:?} lbfgsb={reference:?}"
            );
            for i in 0..n {
                assert!(x[i] >= bounds[i].lower && x[i] <= bounds[i].upper);
                assert!((x[i] - target[i]).abs() < 1e-4);
            }
        }
    }
}

#[test]
fn accurate_differences_resolve_large_offsets_without_function_stopping() {
    let h = Dense(vec![vec![1.0, 0.0], vec![0.0, 100.0]]);
    let result = minimize(
        vec![1.0, 1.0],
        vec![Bound::unbounded(); 2],
        Options::default(),
        |x, g, r| Ok(oracle(&h, &[0.0, 0.0], &[0.0, 0.0], 1e20, x, g, r)),
        |_| Ok(h.clone()),
    )
    .unwrap();
    assert_eq!(result.status, Status::ConvergedProjectedGradient);
    assert_eq!(result.f, 1e20);
    assert!(result.gradient.iter().all(|g| g.abs() <= 1e-8));
}

#[test]
fn indefinite_and_zero_curvature_models_are_safeguarded() {
    for coefficient in [-1.0, 0.0, 1.0] {
        let h = Dense(vec![vec![coefficient]]);
        let result = minimize(
            vec![0.0],
            vec![Bound::new(-2.0, 2.0)],
            Options::default(),
            |x, g, r| {
                g[0] = coefficient * x[0] - 3.0;
                Ok(Evaluation::with_difference(
                    0.5 * coefficient * x[0] * x[0] - 3.0 * x[0],
                    r.map_or(0.0, |r| {
                        (x[0] - r[0]) * (0.5 * coefficient * (x[0] + r[0]) - 3.0)
                    }),
                ))
            },
            |_| Ok(h.clone()),
        )
        .unwrap();
        assert_eq!(
            result.status,
            Status::ConvergedProjectedGradient,
            "{result:?}"
        );
        assert_eq!(result.x, vec![2.0]);
    }
}

#[test]
fn hard_budgets_and_last_accepted_state() {
    for fg_budget in [1, 2, 3, 10] {
        let calls = Cell::new(0);
        let result = minimize(
            vec![3.0],
            vec![Bound::unbounded()],
            Options {
                max_evaluations: fg_budget,
                ..Default::default()
            },
            |x, g, _| {
                calls.set(calls.get() + 1);
                g[0] = x[0].powi(3);
                Ok(Evaluation::value(x[0].powi(4) / 4.0))
            },
            |x| Ok(Dense(vec![vec![3.0 * x[0] * x[0]]])),
        )
        .unwrap();
        assert!(result.evaluations <= fg_budget);
        assert_eq!(result.evaluations, calls.get());
        assert_eq!(result.gradient[0], result.x[0].powi(3));
        assert_eq!(result.f, result.x[0].powi(4) / 4.0);
        if fg_budget == 1 {
            assert_eq!(result.status, Status::MaxEvaluations);
            assert_eq!(result.hessian_evaluations, 0);
        }
    }
    let h = Dense(vec![vec![2.0, 0.9], vec![0.9, 1.0]]);
    let result = minimize(
        vec![3.0, 1.0],
        vec![Bound::unbounded(); 2],
        Options {
            max_hessian_products: 1,
            ..Default::default()
        },
        |x, g, r| Ok(oracle(&h, &[0.0, 0.0], &[0.0, 0.0], 0.0, x, g, r)),
        |_| Ok(h.clone()),
    )
    .unwrap();
    assert_eq!(result.hessian_products, 1);
    assert_eq!(result.status, Status::MaxHessianProducts);
    assert_eq!(result.x, vec![3.0, 1.0]);
}

#[test]
fn fixed_variables_errors_and_nonfinite_trials() {
    let result = minimize(
        vec![2.0],
        vec![Bound::new(1.0, 1.0)],
        Options::default(),
        |x, g, _| {
            g[0] = 123.0;
            Ok(Evaluation::value(x[0]))
        },
        |_| -> anyhow::Result<Dense> { panic!("fixed variables need no model") },
    )
    .unwrap();
    assert_eq!(result.x, vec![1.0]);
    assert_eq!(result.gradient, vec![123.0]);
    assert_eq!(result.evaluations, 1);
    let calls = Cell::new(0);
    assert!(
        minimize(
            vec![0.0],
            vec![Bound::new(2.0, 1.0)],
            Options::default(),
            |_, _, _| {
                calls.set(1);
                Ok(Evaluation::value(0.0))
            },
            |_| Ok(Dense(vec![]))
        )
        .is_err()
    );
    assert_eq!(calls.get(), 0);
    assert!(
        minimize(
            vec![1.0],
            vec![Bound::unbounded()],
            Options::default(),
            |_, g, _| {
                g[0] = 1.0;
                anyhow::bail!("oracle failed")
            },
            |_| Ok(Dense(vec![]))
        )
        .is_err()
    );
    let result = minimize(
        vec![1.0],
        vec![Bound::unbounded()],
        Options::default(),
        |_, g, r| {
            g[0] = 1.0;
            Ok(Evaluation::with_difference(
                if r.is_none() { 1.0 } else { f64::NAN },
                -1.0,
            ))
        },
        |_| Ok(Dense(vec![vec![1.0]])),
    )
    .unwrap();
    assert_eq!(result.x, vec![1.0]);
    assert_eq!(result.f, 1.0);
    assert_ne!(result.status, Status::ConvergedProjectedGradient);
}

#[test]
fn rounded_stagnation_is_not_convergence_and_increases_are_rejected() {
    let result = minimize(
        vec![1e20],
        vec![Bound::unbounded()],
        Options::default(),
        |_, g, r| {
            g[0] = 1.0;
            Ok(Evaluation::with_difference(
                1.0,
                if r.is_some() { 1.0 } else { 0.0 },
            ))
        },
        |_| Ok(Dense(vec![vec![1.0]])),
    )
    .unwrap();
    assert_eq!(result.status, Status::NumericalStagnation);
    assert_eq!(result.evaluations, 1);
    let result = minimize(
        vec![1.0],
        vec![Bound::unbounded()],
        Options::default(),
        |_, g, _| {
            g[0] = 1.0;
            Ok(Evaluation::with_difference(1e20, 1.0))
        },
        |_| Ok(Dense(vec![vec![1.0]])),
    )
    .unwrap();
    assert_eq!(result.status, Status::LineSearchFailed);
    assert_eq!(result.x, vec![1.0]);
}

#[test]
fn product_counts_include_truncated_cg_and_model_errors_propagate() {
    struct Count(Rc<Cell<usize>>);
    impl Hessian for Count {
        fn apply(&self, v: &[f64], out: &mut [f64]) -> anyhow::Result<()> {
            self.0.set(self.0.get() + 1);
            out.copy_from_slice(v);
            Ok(())
        }
    }
    let count = Rc::new(Cell::new(0));
    let result = minimize(
        vec![2.0],
        vec![Bound::unbounded()],
        Options::default(),
        |x, g, _| {
            g[0] = x[0];
            Ok(Evaluation::value(0.5 * x[0] * x[0]))
        },
        |_| Ok(Count(count.clone())),
    )
    .unwrap();
    assert_eq!(result.hessian_products, count.get());
    assert_eq!(result.hessian_products, 1);
    assert!(
        minimize(
            vec![1.0],
            vec![Bound::unbounded()],
            Options::default(),
            |_, g, _| {
                g[0] = 1.0;
                Ok(Evaluation::value(1.0))
            },
            |_| -> anyhow::Result<Dense> { anyhow::bail!("bad model") }
        )
        .is_err()
    );
}
