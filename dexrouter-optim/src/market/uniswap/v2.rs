use serde::{Deserialize, Serialize};

use crate::market::{Market, MarketEvaluation};

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
}

impl Market for UniswapV2 {
    fn relative_curvature(&self, v: [f64; 2]) -> Option<f64> {
        let weighted = [self.reserves[0] * v[0], self.reserves[1] * v[1]];
        let active = self.fee == 1.0
            || weighted[1] * self.fee > weighted[0]
            || weighted[0] * self.fee > weighted[1];
        Some(if active {
            0.5 * self.reserves[0].sqrt() * self.reserves[1].sqrt() / self.fee.sqrt()
                * v[0].sqrt()
                * v[1].sqrt()
        } else {
            0.0
        })
    }

    fn arbitrage(&self, v: [f64; 2]) -> ([f64; 2], [f64; 2]) {
        let trade = self.evaluate(v);
        (trade.input, trade.output)
    }

    fn value_change(
        &self,
        v: [f64; 2],
        reference_v: [f64; 2],
        trade: MarketEvaluation,
        reference: MarketEvaluation,
    ) -> f64 {
        let direction = |t: MarketEvaluation| {
            if t.input[0] > 0.0 {
                1
            } else if t.input[1] > 0.0 {
                -1
            } else {
                0
            }
        };
        if direction(trade) == direction(reference) {
            let linear = (reference.output[0] - reference.input[0]) * (v[0] - reference_v[0])
                + (reference.output[1] - reference.input[1]) * (v[1] - reference_v[1]);
            if direction(trade) == 0 {
                return linear;
            }
            let curvature = self.reserves[0].sqrt() * self.reserves[1].sqrt() / self.fee.sqrt();
            linear + curvature * super::support_remainder(v, reference_v)
        } else {
            trade.value - reference.value
        }
    }

    fn evaluate(&self, v: [f64; 2]) -> MarketEvaluation {
        let mut trade = MarketEvaluation::default();
        // In the active direction the optimized payoff is a squared difference
        // of square roots, not the difference of two large trade valuations.
        let weighted = [self.reserves[0] * v[0], self.reserves[1] * v[1]];
        for (input, output) in [(0, 1), (1, 0)] {
            if weighted[output] * self.fee > weighted[input] {
                let root_out = weighted[output].sqrt();
                let root_in = (weighted[input] / self.fee).sqrt();
                let difference =
                    (weighted[output] - weighted[input] / self.fee) / (root_out + root_in);
                trade.input[input] = (self.reserves[input] / self.fee) * (difference / root_in);
                trade.output[output] = self.reserves[output] * (difference / root_out);
                trade.value = difference * difference;
                break;
            }
        }
        trade
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::UniswapV2;
    use crate::Market;

    fn allclose(x: [f64; 2], y: [f64; 2]) -> bool {
        arr1(&x).abs_diff_eq(&arr1(&y), 1e-12)
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
}
