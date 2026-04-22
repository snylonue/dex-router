use serde::{Deserialize, Serialize};

pub trait Market {
    fn arbitrage(&self, v: [f64; 2]) -> ([f64; 2], [f64; 2]);
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct UniswapV2 {
    /// Reserves for token0 and token1.
    reserves: [f64; 2],
    /// Multiplicative fee factor on input (e.g. 0.997 for a 0.3% fee).
    fee: f64,
}

impl UniswapV2 {
    pub fn new(reserve0: f64, reserve1: f64, fee: f64) -> Self {
        debug_assert!(reserve0 >= 0.0);
        debug_assert!(reserve1 >= 0.0);
        debug_assert!(fee > 0.0 && fee <= 1.0);
        Self {
            reserves: [reserve0, reserve1],
            fee,
        }
    }

    fn arb_in(&self, p: f64, r: f64) -> f64 {
        let k = self.reserves[0] * self.reserves[1];
        (((self.fee * p * k).sqrt() - r) / self.fee).max(0.0)
    }

    fn arb_out(&self, p: f64, r: f64) -> f64 {
        let k = self.reserves[0] * self.reserves[1];
        (r - (k / (p * self.fee)).sqrt()).max(0.0)
    }
}

impl Market for UniswapV2 {
    fn arbitrage(&self, v: [f64; 2]) -> ([f64; 2], [f64; 2]) {
        (
            [
                self.arb_in(v[1] / v[0], self.reserves[0]),
                self.arb_in(v[0] / v[1], self.reserves[1]),
            ],
            [
                self.arb_out(v[0] / v[1], self.reserves[0]),
                self.arb_out(v[1] / v[0], self.reserves[1]),
            ],
        )
    }
}

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
                let range =
                    BoundedLiquidity::new(k, p_cur, pl);
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
                let range =
                    BoundedLiquidity::new(k, p_cur, pu);
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

    use super::{Market, UniswapV2, UniswapV3};

    fn allclose(x: [f64; 2], y: [f64; 2]) -> bool {
        arr1(&x).abs_diff_eq(&arr1(&y), 1e-4)
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

    #[test]
    fn uniswap_v2_arb_token0_in() {
        let pool = UniswapV2::new(10.0, 10.0, 0.997);
        let (input, output) = pool.arbitrage([0.8, 1.0]);
        assert!(allclose(input, [1.167057954747296, 0.0]));
        assert!(allclose(output, [0.0, 1.0422814195522145]));
    }

    #[test]
    fn uniswap_v2_arb_token1_in() {
        let pool = UniswapV2::new(10.0, 10.0, 0.997);
        let (input, output) = pool.arbitrage([1.3, 1.0]);
        assert!(allclose(input, [0.0, 1.3888051889315436]));
        assert!(allclose(output, [1.2162342617354003, 0.0]));
    }

    #[test]
    fn uniswap_v2_no_trade_band() {
        let pool = UniswapV2::new(10.0, 10.0, 0.997);
        let (input, output) = pool.arbitrage([1.0, 1.0]);
        assert!(allclose(input, [0.0, 0.0]));
        assert!(allclose(output, [0.0, 0.0]));
    }

    #[test]
    fn uniswap_v2_price_aligns_after_trade() {
        let pool = UniswapV2::new(10.0, 10.0, 0.997);

        // Token0 -> Token1 direction.
        let v = [0.8, 1.0];
        let (input, _output) = pool.arbitrage(v);
        let a = pool.fee * input[0];
        let x1 = pool.reserves[0] + a;
        let y1 = pool.reserves[0] * pool.reserves[1] / x1;
        let p_after = y1 / x1;
        assert!((p_after - (v[0] / v[1]) / pool.fee).abs() <= 1e-10);

        // Token1 -> Token0 direction.
        let v = [1.3, 1.0];
        let (input, _output) = pool.arbitrage(v);
        let a = pool.fee * input[1];
        let y1 = pool.reserves[1] + a;
        let x1 = pool.reserves[0] * pool.reserves[1] / y1;
        let p_after = y1 / x1;
        assert!((p_after - (v[0] / v[1]) * pool.fee).abs() <= 1e-10);
    }

    #[test]
    fn mixed_v2_v3_pools_can_create_complementary_arbitrage_flows() {
        let valuation = [1.0, 1.0];

        // This v2 pool values token0 cheaply relative to the external valuation,
        // so the optimal trade sells token0 into the pool and receives token1.
        let v2 = UniswapV2::new(5.0, 10.0, 0.997);
        let (v2_input, v2_output) = v2.arbitrage(valuation);
        assert!(v2_input[0] > 0.0);
        assert_eq!(v2_input[1], 0.0);
        assert_eq!(v2_output[0], 0.0);
        assert!(v2_output[1] > 0.0);

        // This single-range v3 pool values token0 expensively relative to the
        // same external valuation, so the optimal trade goes in the opposite
        // direction and receives token0 for token1 input.
        let v3 = UniswapV3::new(0.75, vec![1.0, 0.5], vec![10.0], 0.997);
        let (v3_input, v3_output) = v3.arbitrage(valuation);
        assert_eq!(v3_input[0], 0.0);
        assert!(v3_input[1] > 0.0);
        assert!(v3_output[0] > 0.0);
        assert_eq!(v3_output[1], 0.0);

        let net = [
            v2_output[0] + v3_output[0] - v2_input[0] - v3_input[0],
            v2_output[1] + v3_output[1] - v2_input[1] - v3_input[1],
        ];

        assert!(net[0] > 0.0, "expected net token0 gain, got {net:?}");
        assert!(net[1] > 0.0, "expected net token1 gain, got {net:?}");
    }
}
