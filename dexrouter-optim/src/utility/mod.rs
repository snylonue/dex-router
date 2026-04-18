use core::f64;

use argmin::core::{CostFunction, Gradient};
use ndarray::Array1;

pub struct Utility<T>(pub T);

pub trait UtilityConjugate {
    fn value(&self, v: &Array1<f64>) -> f64;

    fn grad(&self, v: &Array1<f64>) -> Array1<f64>;

    fn lower_bounds(&self) -> Array1<f64>;

    fn upper_bounds(&self) -> Array1<f64>;
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

    fn lower_bounds(&self) -> Array1<f64> {
        &self.c + Array1::from_elem(self.c.dim(), 1e-8)
    }

    fn upper_bounds(&self) -> Array1<f64> {
        Array1::from_elem(self.c.dim(), f64::INFINITY)
    }
}

#[derive(Debug, Clone)]
pub struct BasketLiquidation {
    pub out: usize,
    pub inputs: Array1<f64>,
}

impl UtilityConjugate for BasketLiquidation {
    fn value(&self, v: &Array1<f64>) -> f64 {
        assert!(v.len() == self.inputs.len());
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

    fn lower_bounds(&self) -> Array1<f64> {
        let mut b = Array1::from_elem(self.inputs.dim(), f64::EPSILON.sqrt());
        b[self.out] += 1.0;
        b
    }

    fn upper_bounds(&self) -> Array1<f64> {
        Array1::from_elem(self.inputs.dim(), f64::INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

    #[test]
    fn test_basket_liquidation() {
        let obj = BasketLiquidation {
            out: 0,
            inputs: array![0.0, 1.0],
        };

        let v1 = array![2.0, 3.0];
        assert_eq!(obj.value(&v1), 3.0);

        let v2 = Array1::from_elem(2, 0.5);
        assert!(obj.value(&v2).is_infinite());

        let v3 = Array1::from_elem(2, 2.0);
        let grad1 = obj.grad(&v3);
        assert_eq!(grad1, array![0.0, 1.0]);

        let v4 = Array1::from_elem(2, 0.5);
        let grad2 = obj.grad(&v4);
        assert!(grad2.iter().all(|&x| x.is_infinite()));
    }
}
