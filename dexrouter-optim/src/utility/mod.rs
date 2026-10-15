use core::f64;

use argmin::core::{CostFunction, Gradient};
use ndarray::Array1;

pub struct Utility<T>(pub T);

pub trait UtilityConjugate {
    /// Promise convexity of value(z^2) on the positive, box-constrained domain.
    /// Unknown or signed utilities must keep the default false.
    fn convex_in_sqrt_prices(&self) -> bool {
        false
    }

    fn value(&self, v: &Array1<f64>) -> f64;

    fn grad(&self, v: &Array1<f64>) -> Array1<f64>;

    /// Stable change in value for a centered objective.
    fn value_change(&self, v: &Array1<f64>, reference: &Array1<f64>) -> f64 {
        self.value(v) - self.value(reference)
    }

    fn lower_bounds(&self) -> Array1<f64>;

    fn upper_bounds(&self) -> Array1<f64>;

    /// Opt in only when the finite objective is a sum of per-token terms and
    /// its domain is exactly the supplied box. Unknown utilities stay coupled.
    fn is_separable(&self) -> bool {
        false
    }

    /// Promise that this coordinate has no objective effect anywhere in the
    /// supplied box. Used only to omit isolated, market-free price variables.
    fn is_constant_coordinate(&self, _index: usize) -> bool {
        false
    }

    /// Exact diagonal Hessian on the finite domain. Returning Some also promises
    /// that off-diagonal entries vanish; None preserves finite-difference fallback.
    fn diagonal_hessian(&self, _v: &Array1<f64>) -> Option<Array1<f64>> {
        None
    }

    /// Optional normalization for a nonnegative, degree-one homogeneous dual.
    /// The utility must guarantee that fixing this price to its lower bound
    /// preserves the mathematical optimum. Numerical floors are separate.
    fn price_anchor(&self) -> Option<(usize, f64)> {
        None
    }
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
    fn convex_in_sqrt_prices(&self) -> bool {
        true
    }

    fn is_constant_coordinate(&self, _index: usize) -> bool {
        true
    }

    fn is_separable(&self) -> bool {
        true
    }

    fn diagonal_hessian(&self, v: &Array1<f64>) -> Option<Array1<f64>> {
        Some(Array1::zeros(v.len()))
    }

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
        self.c.clone()
    }

    fn upper_bounds(&self) -> Array1<f64> {
        Array1::from_elem(self.c.dim(), f64::INFINITY)
    }

    fn price_anchor(&self) -> Option<(usize, f64)> {
        let mut nonzero = self.c.iter().enumerate().filter(|(_, c)| **c != 0.0);
        let (index, &price) = nonzero.next()?;
        (price > 0.0 && price.is_finite() && nonzero.next().is_none()).then_some((index, price))
    }
}

#[derive(Debug, Clone)]
pub struct BasketLiquidation {
    pub out: usize,
    pub inputs: Array1<f64>,
}

impl UtilityConjugate for BasketLiquidation {
    fn convex_in_sqrt_prices(&self) -> bool {
        self.inputs.iter().all(|v| v.is_finite() && *v >= 0.0)
    }

    fn is_constant_coordinate(&self, index: usize) -> bool {
        index == self.out || self.inputs[index] == 0.0
    }

    fn is_separable(&self) -> bool {
        true
    }

    fn diagonal_hessian(&self, v: &Array1<f64>) -> Option<Array1<f64>> {
        Some(Array1::zeros(v.len()))
    }

    fn value_change(&self, v: &Array1<f64>, reference: &Array1<f64>) -> f64 {
        if v[self.out] < 1.0 || reference[self.out] < 1.0 {
            return f64::INFINITY;
        }
        let mut sum = crate::numerics::Sum::default();
        for i in 0..v.len() {
            if i != self.out {
                sum.add((v[i] - reference[i]) * self.inputs[i]);
            }
        }
        sum.total()
    }

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
        let mut b = Array1::zeros(self.inputs.dim());
        b[self.out] = 1.0;
        b
    }

    fn upper_bounds(&self) -> Array1<f64> {
        Array1::from_elem(self.inputs.dim(), f64::INFINITY)
    }

    fn price_anchor(&self) -> Option<(usize, f64)> {
        self.inputs
            .iter()
            .all(|x| x.is_finite() && *x >= 0.0)
            .then_some((self.out, 1.0))
    }
}

/// Explicit opt-in for utilities evaluated on the complete hi+lo physical price.
pub trait PreciseUtility: UtilityConjugate {
    fn precise_value(&self, p: &[crate::precision::Price]) -> f64;
    fn precise_grad(&self, p: &[crate::precision::Price]) -> Array1<f64>;
    fn precise_value_change(
        &self,
        p: &[crate::precision::Price],
        reference: &[crate::precision::Price],
    ) -> f64;
    /// Exact regional diagonal, promising zero off-diagonal utility curvature.
    fn precise_diagonal_hessian(&self, p: &[crate::precision::Price]) -> Array1<f64>;
}
impl PreciseUtility for NonnegativeLinear {
    fn precise_value(&self, p: &[crate::precision::Price]) -> f64 {
        if p.len() == self.c.len() && p.iter().zip(&self.c).all(|(&p, &c)| p >= c.into()) {
            0.0
        } else {
            f64::INFINITY
        }
    }
    fn precise_grad(&self, p: &[crate::precision::Price]) -> Array1<f64> {
        Array1::from_elem(
            p.len(),
            if self.precise_value(p).is_finite() {
                0.0
            } else {
                f64::INFINITY
            },
        )
    }
    fn precise_value_change(
        &self,
        p: &[crate::precision::Price],
        r: &[crate::precision::Price],
    ) -> f64 {
        self.precise_value(p) - self.precise_value(r)
    }
    fn precise_diagonal_hessian(&self, p: &[crate::precision::Price]) -> Array1<f64> {
        Array1::zeros(p.len())
    }
}
impl PreciseUtility for BasketLiquidation {
    fn precise_value(&self, p: &[crate::precision::Price]) -> f64 {
        if p[self.out] < 1.0.into() {
            return f64::INFINITY;
        }
        let mut sum = crate::precision::Price::from(0.0);
        for (i, &v) in p.iter().enumerate() {
            if i != self.out {
                sum = sum + v * self.inputs[i].into();
            }
        }
        sum.rounded()
    }
    fn precise_grad(&self, p: &[crate::precision::Price]) -> Array1<f64> {
        if p[self.out] < 1.0.into() {
            Array1::from_elem(p.len(), f64::INFINITY)
        } else {
            let mut g = self.inputs.clone();
            g[self.out] = 0.0;
            g
        }
    }
    fn precise_value_change(
        &self,
        p: &[crate::precision::Price],
        r: &[crate::precision::Price],
    ) -> f64 {
        if p[self.out] < 1.0.into() || r[self.out] < 1.0.into() {
            return f64::INFINITY;
        }
        let mut sum = crate::precision::Price::from(0.0);
        for i in 0..p.len() {
            if i != self.out {
                sum = sum + (p[i] - r[i]) * self.inputs[i].into();
            }
        }
        sum.rounded()
    }
    fn precise_diagonal_hessian(&self, p: &[crate::precision::Price]) -> Array1<f64> {
        Array1::zeros(p.len())
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
