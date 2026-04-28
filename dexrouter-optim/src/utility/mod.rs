use num_traits::Float;

use ndarray::Array1;

pub trait UtilityConjugate<F: Float> {
    fn value(&self, v: &Array1<F>) -> F;

    fn grad(&self, v: &Array1<F>) -> Array1<F>;

    fn lower_bounds(&self) -> Array1<F>;

    fn upper_bounds(&self) -> Array1<F>;
}

#[derive(Debug, Clone)]
pub struct NonnegativeLinear<F: Float> {
    pub c: Array1<F>,
}

impl<F: Float> NonnegativeLinear<F> {
    pub fn feasible(&self, v: &Array1<F>) -> bool {
        assert!(self.c.shape() == v.shape());
        (&self.c - v).fold(true, |acc, &d| acc && d <= F::zero())
    }
}

impl<F: Float> UtilityConjugate<F> for NonnegativeLinear<F> {
    fn value(&self, v: &Array1<F>) -> F {
        if self.feasible(v) {
            F::zero()
        } else {
            F::infinity()
        }
    }

    fn grad(&self, v: &Array1<F>) -> Array1<F> {
        let shape = v.raw_dim();
        if self.feasible(v) {
            Array1::zeros(shape)
        } else {
            Array1::from_elem(shape, F::infinity())
        }
    }

    fn lower_bounds(&self) -> Array1<F> {
        &self.c + Array1::from_elem(self.c.dim(), F::from(1e-8).unwrap())
    }

    fn upper_bounds(&self) -> Array1<F> {
        Array1::from_elem(self.c.dim(), F::infinity())
    }
}

#[derive(Debug, Clone)]
pub struct BasketLiquidation<F: Float> {
    pub out: usize,
    pub inputs: Array1<F>,
}

impl<F: Float> UtilityConjugate<F> for BasketLiquidation<F> {
    fn value(&self, v: &Array1<F>) -> F {
        assert!(v.len() == self.inputs.len());
        if v[self.out] >= F::one() {
            v.iter()
                .zip(self.inputs.iter())
                .enumerate()
                .fold(F::zero(), |acc, (i, (vi, inputi))| {
                    if i == self.out { acc } else { acc + *vi * *inputi }
                })
        } else {
            F::infinity()
        }
    }

    fn grad(&self, v: &Array1<F>) -> Array1<F> {
        if v[self.out] >= F::one() {
            let mut g = self.inputs.clone();
            g[self.out] = F::zero();
            g
        } else {
            Array1::from_elem(self.inputs.raw_dim(), F::infinity())
        }
    }

    fn lower_bounds(&self) -> Array1<F> {
        let mut b = Array1::from_elem(self.inputs.dim(), F::epsilon().sqrt());
        b[self.out] = b[self.out] + F::one();
        b
    }

    fn upper_bounds(&self) -> Array1<F> {
        Array1::from_elem(self.inputs.dim(), F::infinity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

    #[test]
    fn test_basket_liquidation() {
        let obj = BasketLiquidation::<f64> {
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
