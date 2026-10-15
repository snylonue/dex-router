//! Matrix-free, box-constrained damped Newton-CG for general f64 objectives.
//! No application-specific prices, graph assumptions or acceptance tolerances.
//! Hessians must be symmetric; indefinite and singular models are safeguarded.

#![doc = include_str!("../README.md")]

mod solver;
pub use solver::minimize;

#[derive(Debug, Clone, Copy)]
pub struct Bound {
    pub lower: f64,
    pub upper: f64,
}
impl Bound {
    pub fn new(lower: f64, upper: f64) -> Self {
        Self { lower, upper }
    }
    pub fn unbounded() -> Self {
        Self::new(f64::NEG_INFINITY, f64::INFINITY)
    }
    pub(crate) fn project(self, x: f64) -> f64 {
        x.max(self.lower).min(self.upper)
    }
}

/// A fixed symmetric positive-definite inverse metric for one PCG solve.
/// Write every component. This changes only the linear solver, not the objective.
pub trait Preconditioner {
    fn apply(&self, residual: &[f64], solution: &mut [f64]) -> anyhow::Result<()>;
}

/// A symmetric local Hessian, constructed once per outer iteration. Products
/// must write every component, including fixed coordinates. The solver masks
/// fixed/active directions before calling apply. No dense matrix is required.
pub trait Hessian {
    fn apply(&self, vector: &[f64], product: &mut [f64]) -> anyhow::Result<()>;
    /// Jacobi metric. Negative entries are allowed for indefinite models; their
    /// magnitudes are used for preconditioning. Zero entries use unit metric.
    /// This is not a claim that the model is positive definite.
    fn diagonal(&self, diagonal: &mut [f64]) -> anyhow::Result<()> {
        diagonal.fill(1.0);
        Ok(())
    }
    /// Optionally build a fixed SPD preconditioner for H_FF + damping*D_FF.
    /// metric is the solver's positive Jacobi D, and free is the active mask.
    /// None retains the original Jacobi arithmetic; construction uses no HVPs.
    /// Errors and nonfinite custom products are not silently ignored.
    fn preconditioner(
        &self,
        _metric: &[f64],
        _free: &[bool],
        _damping: f64,
    ) -> anyhow::Result<Option<Box<dyn Preconditioner>>> {
        Ok(None)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Evaluation {
    pub value: f64,
    /// Optional accurate f(x)-f(reference), never a surrogate or residual norm.
    /// Ignored on the initial callback. None uses ordinary value subtraction.
    pub difference: Option<f64>,
}
impl Evaluation {
    pub fn value(value: f64) -> Self {
        Self {
            value,
            difference: None,
        }
    }
    pub fn with_difference(value: f64, difference: f64) -> Self {
        Self {
            value,
            difference: Some(difference),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Options {
    pub pgtol: f64,
    pub max_iterations: usize,
    pub max_evaluations: usize,
    pub max_hessian_products: usize,
    pub max_cg_iterations: usize,
    /// Per direction (Newton and at most one gradient fallback) trial cap.
    pub max_line_search_steps: usize,
    pub armijo: f64,
    /// Maximum backtracking factor. Resolved, consistent quadratic models may
    /// propose a smaller factor, bounded below by 0.2 * backtrack.
    pub backtrack: f64,
    pub cg_relative_tolerance: f64,
    /// Dimensionless damping in the diagonal metric, not an objective penalty.
    pub initial_damping: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            pgtol: 1e-8,
            max_iterations: 1000,
            max_evaluations: 10000,
            max_hessian_products: 100000,
            max_cg_iterations: 200,
            max_line_search_steps: 40,
            armijo: 1e-4,
            backtrack: 0.5,
            cg_relative_tolerance: 0.1,
            initial_damping: 0.0,
        }
    }
}
impl Options {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.pgtol.is_finite() && self.pgtol >= 0.0, "invalid pgtol");
        anyhow::ensure!(
            self.max_iterations > 0
                && self.max_evaluations > 0
                && self.max_hessian_products > 0
                && self.max_cg_iterations > 0
                && self.max_line_search_steps > 0,
            "budgets must be positive"
        );
        anyhow::ensure!(
            self.armijo.is_finite() && self.armijo > 0.0 && self.armijo < 0.5,
            "armijo must be in (0, 0.5)"
        );
        anyhow::ensure!(
            self.backtrack.is_finite() && self.backtrack > 0.0 && self.backtrack < 1.0,
            "backtrack must be in (0, 1)"
        );
        anyhow::ensure!(
            self.cg_relative_tolerance.is_finite()
                && self.cg_relative_tolerance > 0.0
                && self.cg_relative_tolerance < 1.0,
            "cg relative tolerance must be in (0, 1)"
        );
        anyhow::ensure!(
            self.initial_damping.is_finite() && self.initial_damping >= 0.0,
            "invalid damping"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    ConvergedProjectedGradient,
    MaxIterations,
    MaxEvaluations,
    MaxHessianProducts,
    LineSearchFailed,
    NumericalStagnation,
}

#[derive(Debug, Clone)]
pub struct OptimizeResult {
    pub x: Vec<f64>,
    pub f: f64,
    pub gradient: Vec<f64>,
    pub projected_gradient_inf_norm: f64,
    pub status: Status,
    pub iterations: usize,
    /// Joint objective/gradient callbacks, including rejected nonfinite trials.
    pub evaluations: usize,
    pub hessian_evaluations: usize,
    pub hessian_products: usize,
    pub gradient_fallbacks: usize,
    pub curvature_truncations: usize,
}

/// Cancellation-free infinity norm of x - P(x-g), the usual box stationarity map.
pub fn projected_gradient_norm(x: &[f64], g: &[f64], bounds: &[Bound]) -> f64 {
    assert_eq!(x.len(), g.len());
    assert_eq!(x.len(), bounds.len());
    x.iter()
        .zip(g)
        .zip(bounds)
        .map(|((&x, &g), b)| {
            if g > 0.0 {
                g.min(x - b.lower)
            } else {
                (-g).min(b.upper - x)
            }
        })
        .fold(0.0, f64::max)
}

pub(crate) fn dot(x: &[f64], y: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for (&x, &y) in x.iter().zip(y) {
        let value = x * y;
        let next = sum + value;
        correction += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    sum + correction
}
pub(crate) fn norm(x: &[f64]) -> f64 {
    x.iter().map(|x| x.abs()).fold(0.0, f64::max)
}
