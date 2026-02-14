use argmin::core::{CostFunction, Gradient};
use ndarray::Array1;

pub struct Utility<T>(pub T);

pub trait UtilityConjugate {
    fn value(&self, v: &Array1<f64>) -> f64;

    fn grad(&self, v: &Array1<f64>) -> Array1<f64>;
}

impl<T: UtilityConjugate> CostFunction for Utility<T> {
    type Param = Array1<f64>;

    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, anyhow::Error> {
        Ok(self.0.value(param))
    }
}

impl<T: UtilityConjugate> Gradient for Utility<T> {
    type Param = Array1<f64>;

    type Gradient = Array1<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, anyhow::Error> {
        Ok(self.0.grad(param))
    }
}

pub struct NonnegativeLinear {
    pub c: Array1<f64>,
}

impl NonnegativeLinear {
    pub fn feasible(&self, v: &Array1<f64>) -> bool {
        assert!(self.c.shape() == v.shape());
        (&self.c - v).fold(true, |acc, &d| acc && d <= 0.0)
    }
}

impl UtilityConjugate for NonnegativeLinear {
    fn value(&self, v: &Array1<f64>) -> f64 {
        if self.feasible(v) { 0.0 } else { f64::INFINITY }
    }

    fn grad(&self, v: &Array1<f64>) -> Array1<f64> {
        let shape = v.raw_dim();
        if self.feasible(v) {
            Array1::zeros(shape)
        } else {
            Array1::from_elem(shape, f64::INFINITY)
        }
    }
}
