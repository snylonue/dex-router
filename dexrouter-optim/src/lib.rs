use argmin::core::{CostFunction, Gradient};
use ndarray::{Array1, Array2};

use crate::{market::Market, utility::UtilityConjugate};

pub mod market;
mod numerics;
pub mod utility;

#[derive(Debug, Clone)]
pub struct Route<U, M> {
    pub objective: U,
    pub markets: Vec<(M, (usize, usize))>,
    pub tokens: usize,
}

impl<U, M: Market> Route<U, M> {
    pub fn arbitrage(&self, p: Array1<f64>) -> (Array2<f64>, Array2<f64>) {
        let mut inputs = Array2::zeros([self.markets.len(), self.tokens]);
        let mut outputs = Array2::zeros([self.markets.len(), self.tokens]);

        for (i, (m, (idx0, idx1))) in self.markets.iter().enumerate() {
            let (input, output) = m.arbitrage([p[*idx0], p[*idx1]]);
            inputs[[i, *idx0]] = input[0];
            inputs[[i, *idx1]] = input[1];
            outputs[[i, *idx0]] = output[0];
            outputs[[i, *idx1]] = output[1];
        }

        (inputs, outputs)
    }
}

/// Joint evaluation, including gross per-token turnover for residual scaling.
#[derive(Debug, Clone)]
pub struct RouteEvaluation {
    pub cost: f64,
    pub gradient: Array1<f64>,
    pub turnover: Array1<f64>,
    pub(crate) market_evaluations: Vec<market::MarketEvaluation>,
}

impl<U: UtilityConjugate, M: Market> Route<U, M> {
    pub fn evaluate(&self, price: &Array1<f64>) -> RouteEvaluation {
        self.evaluate_centered(price, None).0
    }

    pub(crate) fn evaluate_centered(
        &self,
        price: &Array1<f64>,
        reference: Option<(&Array1<f64>, &RouteEvaluation)>,
    ) -> (RouteEvaluation, f64) {
        let mut change = numerics::Sum::default();
        if let Some((reference_price, _)) = reference {
            change.add(self.objective.value_change(price, reference_price));
        }
        let mut market_evaluations = Vec::with_capacity(self.markets.len());
        let mut cost = numerics::Sum::default();
        cost.add(self.objective.value(price));
        let mut gradient = vec![numerics::Sum::default(); self.tokens];
        for (sum, &g) in gradient.iter_mut().zip(self.objective.grad(price).iter()) {
            sum.add(g);
        }
        let mut turnover = vec![numerics::Sum::default(); self.tokens];
        for (index, (market, (a, b))) in self.markets.iter().enumerate() {
            let trade = market.evaluate([price[*a], price[*b]]);
            cost.add(trade.value);
            market_evaluations.push(trade);
            if let Some((reference_price, reference)) = reference {
                change.add(market.value_change(
                    [price[*a], price[*b]],
                    [reference_price[*a], reference_price[*b]],
                    trade,
                    reference.market_evaluations[index],
                ));
            }
            for (local, token) in [*a, *b].into_iter().enumerate() {
                gradient[token].add(trade.output[local] - trade.input[local]);
                turnover[token].add(trade.input[local] + trade.output[local]);
            }
        }
        (
            RouteEvaluation {
                cost: cost.total(),
                gradient: Array1::from_iter(gradient.into_iter().map(numerics::Sum::total)),
                turnover: Array1::from_iter(turnover.into_iter().map(numerics::Sum::total)),
                market_evaluations,
            },
            change.total(),
        )
    }

    pub fn utility(&self, v: &Array1<f64>) -> f64 {
        self.evaluate(v).cost
    }
}

impl<U: UtilityConjugate, M: Market> CostFunction for Route<U, M> {
    type Param = Array1<f64>;

    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, anyhow::Error> {
        Ok(self.utility(param))
    }
}

impl<U: UtilityConjugate, M: Market> Gradient for Route<U, M> {
    type Param = Array1<f64>;

    type Gradient = Array1<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, anyhow::Error> {
        Ok(self.evaluate(param).gradient)
    }
}

mod solver;
pub use solver::{
    SolveAssessment, SolveDiagnostics, SolveOptions, solve_price, solve_price_with_init_p,
    solve_price_with_init_p_checked, solve_price_with_options,
};

#[cfg(test)]
mod tests {
    use argmin::core::{CostFunction, Gradient};
    use ndarray::arr1;

    use crate::{
        Route,
        market::{Market, UniswapV2, UniswapV3},
        utility::NonnegativeLinear,
    };

    impl<M: Market + ?Sized> Market for Box<M> {
        fn arbitrage(&self, v: [f64; 2]) -> ([f64; 2], [f64; 2]) {
            (**self).arbitrage(v)
        }
    }

    #[test]
    fn route_gradient_matches_finite_difference_uniswap_v2() {
        let route = Route {
            objective: NonnegativeLinear {
                c: arr1(&[1.0, 1.0]),
            },
            markets: vec![(UniswapV2::new(10.0, 10.0, 0.997), (0, 1))],
            tokens: 2,
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
            objective: NonnegativeLinear {
                c: arr1(&[1.0, 1.0]),
            },
            markets: vec![
                (
                    Box::new(UniswapV2::new(5.0, 10.0, 0.997)) as Box<dyn Market>,
                    (0, 1),
                ),
                (
                    Box::new(UniswapV3::new(0.75, vec![1.0, 0.5], vec![10.0], 0.997))
                        as Box<dyn Market>,
                    (0, 1),
                ),
            ],
            tokens: 2,
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
