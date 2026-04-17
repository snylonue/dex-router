use core::f64;

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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct BasketLiquidation {
    pub out: usize,
    pub inputs: Array1<f64>,
}

impl UtilityConjugate for BasketLiquidation {
    fn value(&self, v: &Array1<f64>) -> f64 {
        if v[self.out] >= 1.0 {
            v.iter()
                .zip(self.inputs.iter())
                .enumerate()
                .map(|(i, (vi, inputi))| if i == self.out { 0.0 } else { vi * inputi })
                .sum()
        } else {
            f64::INFINITY
        }
    }

    fn grad(&self, v: &Array1<f64>) -> Array1<f64> {
        if v[self.out] >= 1.0 {
            let mut g = self.inputs.clone();
            g[self.out] = 0.0;
            g
        } else {
            Array1::from_elem(self.inputs.raw_dim(), f64::INFINITY)
        }
    }
}
