use argmin::core::{CostFunction, Gradient};
use ndarray::{Array1, Ix};

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
