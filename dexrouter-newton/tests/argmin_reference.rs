//! Argmin Newton-CG is an unconstrained reference, not a bounded solver.
//! All three receive the same f/g and coordinates; accurate differences are
//! deliberately disabled here. Assert independent stationarity, not status labels.
use argmin::{
    core::{
        CostFunction, Error, Executor, Gradient, Hessian as ArgminHessian, IterState, KV,
        Problem as ArgminProblem, Solver, State, TerminationReason, TerminationStatus,
    },
    solver::{linesearch::MoreThuenteLineSearch, newton::NewtonCG},
};
use dexrouter_newton::{Bound, Evaluation, Hessian, Options, minimize};
use std::{cell::Cell, rc::Rc};

#[derive(Clone)]
struct Problem {
    scale: f64,
    quartic: bool,
    counts: Rc<[Cell<usize>; 3]>,
}
impl Problem {
    fn derivatives(&self, x: &[f64]) -> (f64, Vec<f64>, Vec<Vec<f64>>) {
        let a = x[0] - 0.3;
        let b = x[1] + 0.7;
        let q = if self.quartic { 0.1 } else { 0.0 };
        (
            self.scale * (0.5 * (5.0 * a * a + 4.0 * a * b + b * b) + q * (a.powi(4) + b.powi(4))),
            vec![
                self.scale * (5.0 * a + 2.0 * b + 4.0 * q * a.powi(3)),
                self.scale * (2.0 * a + b + 4.0 * q * b.powi(3)),
            ],
            vec![
                vec![self.scale * (5.0 + 12.0 * q * a * a), self.scale * 2.0],
                vec![self.scale * 2.0, self.scale * (1.0 + 12.0 * q * b * b)],
            ],
        )
    }
}
impl CostFunction for Problem {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, x: &Vec<f64>) -> Result<f64, Error> {
        self.counts[0].set(self.counts[0].get() + 1);
        anyhow::ensure!(self.counts[0].get() <= 10000, "reference cost budget");
        Ok(self.derivatives(x).0)
    }
}
impl Gradient for Problem {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Result<Vec<f64>, Error> {
        self.counts[1].set(self.counts[1].get() + 1);
        anyhow::ensure!(self.counts[1].get() <= 10000, "reference gradient budget");
        Ok(self.derivatives(x).1)
    }
}
impl ArgminHessian for Problem {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;
    fn hessian(&self, x: &Vec<f64>) -> Result<Self::Hessian, Error> {
        self.counts[2].set(self.counts[2].get() + 1);
        Ok(self.derivatives(x).2)
    }
}
type NewtonState = IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64>;
// Override cost-difference stopping: the reference uses the same independent
// gradient target. Extra terminal gradient checks are counted, then cached for
// the next Newton iteration. No change to its CG or line-search algorithm.
struct GradientStop<S> {
    inner: S,
    problem: Problem,
    goal: f64,
}
impl<S: Solver<Problem, NewtonState>> Solver<Problem, NewtonState> for GradientStop<S> {
    fn name(&self) -> &str {
        "argmin Newton-CG with common gradient stopping"
    }
    fn init(
        &mut self,
        p: &mut ArgminProblem<Problem>,
        s: NewtonState,
    ) -> Result<(NewtonState, Option<KV>), Error> {
        let (s, kv) = self.inner.init(p, s)?;
        let g = self.problem.gradient(s.get_param().unwrap())?;
        Ok((s.gradient(g), kv))
    }
    fn next_iter(
        &mut self,
        p: &mut ArgminProblem<Problem>,
        s: NewtonState,
    ) -> Result<(NewtonState, Option<KV>), Error> {
        let (s, kv) = self.inner.next_iter(p, s)?;
        let g = self.problem.gradient(s.get_param().unwrap())?;
        Ok((s.gradient(g), kv))
    }
    fn terminate(&mut self, s: &NewtonState) -> TerminationStatus {
        if s.get_gradient()
            .unwrap()
            .iter()
            .all(|g| g.abs() <= self.goal)
        {
            TerminationStatus::Terminated(TerminationReason::SolverConverged)
        } else {
            TerminationStatus::NotTerminated
        }
    }
}
struct Matrix(Vec<Vec<f64>>);
impl Hessian for Matrix {
    fn apply(&self, v: &[f64], out: &mut [f64]) -> anyhow::Result<()> {
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.0[i].iter().zip(v).map(|(a, b)| a * b).sum();
        }
        Ok(())
    }
    fn diagonal(&self, out: &mut [f64]) -> anyhow::Result<()> {
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.0[i][i];
        }
        Ok(())
    }
}
#[test]
fn newton_cg_matches_argmin_and_lbfgsb_on_smooth_controls() {
    for scale in [1e-4, 1.0, 1e4] {
        for quartic in [false, true] {
            let p = Problem {
                scale,
                quartic,
                counts: Rc::new(std::array::from_fn(|_| Cell::new(0))),
            };
            let start = vec![2.0, -1.5];
            let goal = scale * 1e-7;
            let ours = minimize(
                start.clone(),
                vec![Bound::unbounded(); 2],
                Options {
                    pgtol: goal,
                    max_iterations: 1000,
                    ..Default::default()
                },
                |x, g, _| {
                    let (f, gradient, _) = p.derivatives(x);
                    g.copy_from_slice(&gradient);
                    Ok(Evaluation::value(f))
                },
                |x| Ok(Matrix(p.derivatives(x).2)),
            )
            .unwrap();
            let lb = dexrouter_lbfgsb::minimize(
                start.clone(),
                vec![dexrouter_lbfgsb::Bound::unbounded(); 2],
                dexrouter_lbfgsb::Options {
                    pgtol: goal,
                    max_iterations: 1000,
                    ..Default::default()
                },
                |x, g| {
                    let (f, gradient, _) = p.derivatives(x);
                    g.copy_from_slice(&gradient);
                    Ok(f)
                },
            )
            .unwrap();
            let line = MoreThuenteLineSearch::<Vec<f64>, Vec<f64>, f64>::new();
            let solver = GradientStop {
                inner: NewtonCG::new(line),
                problem: p.clone(),
                goal,
            };
            let other = Executor::new(p.clone(), solver)
                .configure(|s| s.param(start).max_iters(1000))
                .run()
                .unwrap();
            for (name, x) in [
                ("ours", &ours.x),
                ("argmin", other.state.get_param().unwrap()),
                ("lbfgsb", &lb.x),
            ] {
                let (f, g, _) = p.derivatives(x);
                let residual = g.iter().map(|g| g.abs()).fold(0.0, f64::max);
                assert!(
                    residual <= goal,
                    "scale={scale} quartic={quartic} {name}: residual={residual} goal={goal}"
                );
                assert!(f.abs() / scale < 1e-12);
                assert!((x[0] - 0.3).abs() < 1e-6 && (x[1] + 0.7).abs() < 1e-6);
            }
            println!(
                "scale={scale} quartic={quartic} ours(FG={},HVP={}) lbfgsb(FG={}) argmin(F={},G={},H={})",
                ours.evaluations,
                ours.hessian_products,
                lb.evaluations,
                p.counts[0].get(),
                p.counts[1].get(),
                p.counts[2].get()
            );
        }
    }
}
