use serde::{Deserialize, Serialize};

use crate::market::Market;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UniswapV3 {
    // sqrt of current price
    current_price: f64,
    current_tick: usize,
    // sqrt of lower price
    lower_prices: Vec<f64>,
    // sqrt of liquidity
    liquidity: Vec<f64>,
    fee: f64,
}

impl UniswapV3 {
    pub fn new(current_price: f64, lower_prices: Vec<f64>, liquidity: Vec<f64>, fee: f64) -> Self {
        Self {
            current_price,
            current_tick: lower_prices
                .partition_point(|p| p >= &current_price)
                .checked_sub(1)
                .unwrap_or(0),
            lower_prices,
            liquidity,
            fee,
        }
    }

    pub fn scaled(mut self, c0: f64, c1: f64) -> Self {
        self.liquidity = self.liquidity.into_iter().map(|l| l * c0 * c1).collect();
        self.lower_prices = self.lower_prices.into_iter().map(|p| p * c1 / c0).collect();
        self.current_price *= c1 / c0;
        self
    }

    pub fn current_tick(&self) -> usize {
        self.current_tick
    }

    pub fn lower_prices(&self) -> &[f64] {
        &self.lower_prices
    }
}

impl Market for UniswapV3 {
    fn arbitrage(&self, v: [f64; 2]) -> ([f64; 2], [f64; 2]) {
        let p = v[0] / v[1];
        let p0 = self.current_price.powi(2);

        if p < p0 * self.fee {
            let prices = &self.lower_prices[self.current_tick..];
            let liquidity = &self.liquidity[self.current_tick..];
            let mut input = [0.0; 2];
            let mut output = [0.0; 2];
            let mut initial = true;

            for i in 0..self.liquidity.len() - self.current_tick {
                let k = liquidity[i];
                if k.abs() <= f64::EPSILON {
                    initial = false;
                    continue;
                }
                let pu = prices[i];
                let pl = prices.get(i + 1).copied().unwrap_or_default();
                let p_cur = if i > 0 { pu } else { self.current_price };
                let range = BoundedLiquidity::new(k, p_cur, pl);
                let (delta0, delta1) = range.arbitrage_pos((p / self.fee).sqrt());
                if !initial && (delta0.abs() <= f64::EPSILON || delta1.abs() <= f64::EPSILON) {
                    break;
                }
                initial = false;
                input[0] += delta0;
                output[1] += delta1;
            }
            input[0] /= self.fee;
            (input, output)
        } else if p > p0 / self.fee {
            // current_price is at or below the lowest tick — no valid tick
            // boundary below to serve as the starting point for arb_neg.
            if self.current_tick >= self.lower_prices.len().saturating_sub(1) {
                return Default::default();
            }
            let prices = &self.lower_prices[1..=self.current_tick + 1];
            let liquidity = &self.liquidity[..=self.current_tick];
            let mut input = [0.0; 2];
            let mut output = [0.0; 2];
            let mut initial = true;

            for i in (0..=self.current_tick).rev() {
                let k = liquidity[i];
                if k.abs() <= f64::EPSILON {
                    initial = false;
                    continue;
                }
                let pl = prices[i];
                let pu = if i == 0 {
                    self.lower_prices[0]
                } else {
                    prices[i - 1]
                };
                let p_cur = if i < self.current_tick {
                    pl
                } else {
                    self.current_price
                };
                let range = BoundedLiquidity::new(k, p_cur, pu);
                let (delta0, delta1) = range.arbitrage_neg((p * self.fee).sqrt());
                if !initial && (delta0.abs() <= f64::EPSILON || delta1.abs() <= f64::EPSILON) {
                    break;
                }
                initial = false;
                input[1] += delta1;
                output[0] += delta0;
            }
            input[1] /= self.fee;
            (input, output)
        } else {
            Default::default()
        }
    }
}

#[derive(Debug)]
pub struct BoundedLiquidity {
    k: f64,
    p0: f64,
    p1: f64,
}

impl BoundedLiquidity {
    pub fn new(k: f64, p0: f64, p1: f64) -> Self {
        // debug_assert!((k * k - (r1 + alpha) * (r2 + beta)).abs() <= f64::EPSILON);
        Self { k, p0, p1 }
    }

    /// p: sqrt(p / gamma)
    pub fn arbitrage_pos(&self, p: f64) -> (f64, f64) {
        let delta1 = self.k * (self.p0 - p) / self.p0 / p;
        if delta1 <= f64::EPSILON {
            (0.0, 0.0)
        } else {
            let delta1_max = self.k * ((self.p0 - self.p1) / self.p0 / self.p1);
            if delta1 >= delta1_max {
                (delta1_max, self.k * (self.p0 - self.p1))
            } else {
                (delta1, self.k * (self.p0 - p))
            }
        }
    }

    /// p: sqrt(p * gamma)
    pub fn arbitrage_neg(&self, p: f64) -> (f64, f64) {
        let delta2 = self.k * (p - self.p0);
        if delta2 <= f64::EPSILON {
            (0.0, 0.0)
        } else {
            let delta2_max = self.k * (self.p1 - self.p0);
            if delta2 >= delta2_max {
                (self.k * (self.p1 - self.p0) / self.p0 / self.p1, delta2_max)
            } else {
                (self.k * (p - self.p0) / self.p0 / p, delta2)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::{Market, UniswapV3};

    fn allclose(x: [f64; 2], y: [f64; 2]) -> bool {
        arr1(&x).abs_diff_eq(&arr1(&y), 1e-12)
    }

    #[test]
    fn uniswap_v3_arb_neg() {
        let pool = UniswapV3::new(
            3.872983346207417,
            vec![
                5.477225575051661,
                4.47213595499958,
                3.1622776601683795,
                2.23606797749979,
            ],
            vec![1.0, 1.4142135623730951, 1.224744871391589, 0.0],
            0.997,
        );

        let (input, output) = pool.arbitrage([25.0, 1.0]);

        assert!(allclose(input, [0.0, 1.3718035675347677]));
        assert!(allclose(output, [0.07222672671131006, 0.0]))
    }

    #[test]
    fn uniswap_v3_arb_pos() {
        let pool = UniswapV3::new(
            3.872983346207417,
            vec![
                5.477225575051661,
                4.47213595499958,
                3.1622776601683795,
                2.23606797749979,
            ],
            vec![1.0, 1.4142135623730951, 1.224744871391589, 0.0],
            0.997,
        );

        let (input, output) = pool.arbitrage([10.0, 1.0]);

        assert!(allclose(input, [0.08163881601325118, 0.0]));
        assert!(allclose(output, [0.0, 0.9983662848277683]))
    }
}
