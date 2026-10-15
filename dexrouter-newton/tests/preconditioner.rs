use dexrouter_newton::{Bound, Evaluation, Hessian, Options, Preconditioner, Status, minimize};
use std::{cell::Cell, rc::Rc};
struct Model {
    mode: usize,
    calls: Rc<Cell<usize>>,
}
impl Preconditioner for Model {
    fn apply(&self, r: &[f64], out: &mut [f64]) -> anyhow::Result<()> {
        self.calls.set(self.calls.get() + 1);
        match self.mode {
            1 => anyhow::bail!("metric application error"),
            2 => out.fill(f64::NAN),
            3 => out[0] = r[0], // Deliberately incomplete, must not reuse old data.
            _ => {
                out[0] = (3.0 * r[0] - r[1]) / 11.0;
                out[1] = (4.0 * r[1] - r[0]) / 11.0;
            }
        }
        Ok(())
    }
}
impl Hessian for Model {
    fn apply(&self, v: &[f64], out: &mut [f64]) -> anyhow::Result<()> {
        out[0] = 4.0 * v[0] + v[1];
        out[1] = v[0] + 3.0 * v[1];
        Ok(())
    }
    fn diagonal(&self, out: &mut [f64]) -> anyhow::Result<()> {
        out.copy_from_slice(&[4.0, 3.0]);
        Ok(())
    }
    fn preconditioner(
        &self,
        metric: &[f64],
        free: &[bool],
        damping: f64,
    ) -> anyhow::Result<Option<Box<dyn Preconditioner>>> {
        assert_eq!(metric, &[4.0, 3.0]);
        assert!(free[0]);
        assert_eq!(damping, 0.0);
        if self.mode == 0 {
            anyhow::bail!("metric construction error");
        }
        Ok(Some(Box::new(Self {
            mode: self.mode,
            calls: self.calls.clone(),
        })))
    }
}
#[test]
fn exact_inverse_metric_respects_masks_and_hvp_budgets() {
    for fixed in [false, true] {
        let calls = Rc::new(Cell::new(0));
        let counter = calls.clone();
        let bounds = vec![
            Bound::unbounded(),
            if fixed {
                Bound::new(2.0, 2.0)
            } else {
                Bound::unbounded()
            },
        ];
        let result = minimize(
            vec![1.0, 2.0],
            bounds,
            Options {
                max_hessian_products: 1,
                ..Default::default()
            },
            |x, g, r| {
                let a = x[0] - 0.2;
                let b = x[1] - 2.0;
                g[0] = 4.0 * a + b;
                g[1] = a + 3.0 * b;
                let f = 2.0 * a * a + a * b + 1.5 * b * b;
                let delta = r.map_or(0.0, |r| {
                    let da = x[0] - r[0];
                    let db = x[1] - r[1];
                    da * (2.0 * (x[0] + r[0] - 0.4) + 0.5 * (x[1] + r[1] - 4.0))
                        + db * (0.5 * (x[0] + r[0] - 0.4) + 1.5 * (x[1] + r[1] - 4.0))
                });
                Ok(Evaluation::with_difference(f, delta))
            },
            |_| {
                Ok(Model {
                    mode: 4,
                    calls: counter.clone(),
                })
            },
        )
        .unwrap();
        assert_eq!(
            result.status,
            Status::ConvergedProjectedGradient,
            "{result:?}"
        );
        assert_eq!(result.hessian_products, 1);
        assert!(calls.get() >= 1);
        assert!((result.x[0] - 0.2).abs() < 1e-12);
        assert!((result.x[1] - 2.0).abs() < 1e-12);
    }
}
#[test]
fn custom_metric_errors_and_incomplete_products_are_not_hidden() {
    for mode in 0..4 {
        let calls = Rc::new(Cell::new(0));
        let result = minimize(
            vec![1.0, 2.0],
            vec![Bound::unbounded(); 2],
            Options::default(),
            |x, g, _| {
                g[0] = 4.0 * x[0] + x[1];
                g[1] = x[0] + 3.0 * x[1];
                Ok(Evaluation::value(
                    2.0 * x[0] * x[0] + x[0] * x[1] + 1.5 * x[1] * x[1],
                ))
            },
            |_| {
                Ok(Model {
                    mode,
                    calls: calls.clone(),
                })
            },
        );
        assert!(result.is_err(), "mode={mode}");
    }
}
