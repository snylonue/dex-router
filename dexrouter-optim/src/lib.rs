use argmin::core::{CostFunction, Gradient};
use ndarray::Array1;

use crate::{
    market::Market,
    utility::{Utility, UtilityConjugate},
};

pub mod market;
pub mod utility;

pub struct Route<U, M> {
    pub objective: Utility<U>,
    pub markets: Vec<(M, (usize, usize))>,
}

impl<U: UtilityConjugate, M: Market> CostFunction for Route<U, M> {
    type Param = Array1<f64>;

    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, anyhow::Error> {
        Ok(self.objective.cost(param)?
            + self
                .markets
                .iter()
                .map(|&(ref m, (idx0, idx1))| {
                    let v = [param[idx0], param[idx1]];
                    let (input, output) = m.arbitrage(v);
                    (output[0] * v[0] + output[1] * v[1]) - (input[0] * v[0] + input[1] * v[1])
                })
                .sum::<f64>())
    }
}

impl<U: UtilityConjugate, M: Market> Gradient for Route<U, M> {
    type Param = Array1<f64>;

    type Gradient = Array1<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, anyhow::Error> {
        let mut g = self.objective.gradient(param)?;

        for &(ref m, (idx0, idx1)) in &self.markets {
            let (input, output) = m.arbitrage([param[idx0], param[idx1]]);
            g[idx0] += output[0] - input[0];
            g[idx1] += output[1] - input[1];
        }

        Ok(g)
    }
}

#[cfg(test)]
mod tests {
    use argmin::core::{CostFunction, Gradient};
    use ndarray::arr1;

    use crate::{
        Route,
        market::{Market, UniswapV2, UniswapV3},
        utility::{NonnegativeLinear, Utility},
    };

    impl<M: Market + ?Sized> Market for Box<M> {
        fn arbitrage(&self, v: [f64; 2]) -> ([f64; 2], [f64; 2]) {
            (**self).arbitrage(v)
        }
    }

    #[test]
    fn route_gradient_matches_finite_difference_uniswap_v2() {
        let route = Route {
            objective: Utility(NonnegativeLinear { c: arr1(&[1.0, 1.0]) }),
            markets: vec![(UniswapV2::new(10.0, 10.0, 0.997), (0, 1))],
        };

        // Keep this far from any no-trade boundary to avoid non-smooth points.
        // Also keep it strictly feasible for the indicator utility (v >= c).
        let v = arr1(&[2.0, 1.5]);
        let g = route.gradient(&v).unwrap();

        let eps = 1e-6;
        for i in 0..2 {
            let mut vp = v.clone();
            let mut vm = v.clone();
            vp[i] += eps;
            vm[i] -= eps;
            let cp = route.cost(&vp).unwrap();
            let cm = route.cost(&vm).unwrap();
            let g_fd = (cp - cm) / (2.0 * eps);
            assert!((g[i] - g_fd).abs() <= 1e-4);
        }
    }

    #[test]
    fn route_gradient_matches_finite_difference_mixed_v2_v3() {
        let route = Route {
            objective: Utility(NonnegativeLinear { c: arr1(&[1.0, 1.0]) }),
            markets: vec![
                (Box::new(UniswapV2::new(5.0, 10.0, 0.997)) as Box<dyn Market>, (0, 1)),
                (
                    Box::new(UniswapV3::new(0.75, vec![1.0, 0.5], vec![10.0], 0.997))
                        as Box<dyn Market>,
                    (0, 1),
                ),
            ],
        };

        // Stay strictly inside the feasible utility region and away from the
        // no-trade boundaries for both pools.
        let v = arr1(&[1.2, 1.1]);
        let g = route.gradient(&v).unwrap();

        let eps = 1e-6;
        for i in 0..2 {
            let mut vp = v.clone();
            let mut vm = v.clone();
            vp[i] += eps;
            vm[i] -= eps;
            let cp = route.cost(&vp).unwrap();
            let cm = route.cost(&vm).unwrap();
            let g_fd = (cp - cm) / (2.0 * eps);
            assert!((g[i] - g_fd).abs() <= 1e-4);
        }
    }
}
