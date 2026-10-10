use serde::{Deserialize, Serialize};

use crate::{
    market::{Market, MarketEvaluation},
    numerics::Sum,
};

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

    fn local_region(&self, v: [f64; 2]) -> (i8, Option<usize>) {
        let (_, gap_down) = sqrt_price_gap(v, self.fee, true, self.current_price);
        let (_, gap_up) = sqrt_price_gap(v, self.fee, false, self.current_price);
        if gap_down < 0.0 {
            for i in self.current_tick..self.liquidity.len() {
                if self.liquidity[i] <= 0.0 {
                    continue;
                }
                let start = if i == self.current_tick {
                    self.current_price
                } else {
                    self.lower_prices[i]
                };
                let end = self.lower_prices.get(i + 1).copied().unwrap_or(0.0);
                if sqrt_price_gap(v, self.fee, true, start).1 >= 0.0 {
                    break;
                }
                if sqrt_price_gap(v, self.fee, true, end).1 >= 0.0 {
                    return (1, Some(i));
                }
            }
            (1, None)
        } else if gap_up > 0.0 {
            if self.current_tick >= self.lower_prices.len().saturating_sub(1) {
                return (0, None);
            }
            for i in (0..=self.current_tick).rev() {
                if self.liquidity[i] <= 0.0 {
                    continue;
                }
                let start = if i == self.current_tick {
                    self.current_price
                } else {
                    self.lower_prices[i + 1]
                };
                let end = self.lower_prices[i];
                if sqrt_price_gap(v, self.fee, false, start).1 <= 0.0 {
                    break;
                }
                if sqrt_price_gap(v, self.fee, false, end).1 <= 0.0 {
                    return (-1, Some(i));
                }
            }
            (-1, None)
        } else {
            (0, None)
        }
    }

    pub fn lower_prices(&self) -> &[f64] {
        &self.lower_prices
    }
}

impl Market for UniswapV3 {
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
        let region = self.local_region(v);
        if region == self.local_region(reference_v) {
            let linear = (reference.output[0] - reference.input[0]) * (v[0] - reference_v[0])
                + (reference.output[1] - reference.input[1]) * (v[1] - reference_v[1]);
            let remainder = region.1.map_or(0.0, |i| {
                self.liquidity[i] / self.fee.sqrt() * super::support_remainder(v, reference_v)
            });
            linear + remainder
        } else {
            trade.value - reference.value
        }
    }

    fn evaluate(&self, v: [f64; 2]) -> MarketEvaluation {
        let (_, gap_down) = sqrt_price_gap(v, self.fee, true, self.current_price);
        let (_, gap_up) = sqrt_price_gap(v, self.fee, false, self.current_price);

        if gap_down < 0.0 {
            let prices = &self.lower_prices[self.current_tick..];
            let liquidity = &self.liquidity[self.current_tick..];
            let mut input = [Sum::default(); 2];
            let mut output = [Sum::default(); 2];
            let mut value = Sum::default();
            let mut initial = true;

            for i in 0..self.liquidity.len() - self.current_tick {
                let k = liquidity[i];
                if k <= 0.0 {
                    initial = false;
                    continue;
                }
                let pu = prices[i];
                let pl = prices.get(i + 1).copied().unwrap_or_default();
                let p_cur = if i > 0 { pu } else { self.current_price };
                let range = BoundedLiquidity::new(k, p_cur, pl);
                let trade = range.evaluate(v, self.fee, true);
                let (delta0, delta1) = (trade.input[0], trade.output[1]);
                if !initial && (delta0 == 0.0 || delta1 == 0.0) {
                    break;
                }
                initial = false;
                input[0].add(delta0);
                output[1].add(delta1);
                value.add(trade.value);
            }
            MarketEvaluation {
                input: [input[0].total(), 0.0],
                output: [0.0, output[1].total()],
                value: value.total(),
            }
        } else if gap_up > 0.0 {
            // current_price is at or below the lowest tick — no valid tick
            // boundary below to serve as the starting point for arb_neg.
            if self.current_tick >= self.lower_prices.len().saturating_sub(1) {
                return Default::default();
            }
            let prices = &self.lower_prices[1..=self.current_tick + 1];
            let liquidity = &self.liquidity[..=self.current_tick];
            let mut input = [Sum::default(); 2];
            let mut output = [Sum::default(); 2];
            let mut value = Sum::default();
            let mut initial = true;

            for i in (0..=self.current_tick).rev() {
                let k = liquidity[i];
                if k <= 0.0 {
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
                let trade = range.evaluate(v, self.fee, false);
                let (delta0, delta1) = (trade.output[0], trade.input[1]);
                if !initial && (delta0 == 0.0 || delta1 == 0.0) {
                    break;
                }
                initial = false;
                input[1].add(delta1);
                output[0].add(delta0);
                value.add(trade.value);
            }
            MarketEvaluation {
                input: [0.0, input[1].total()],
                output: [output[0].total(), 0.0],
                value: value.total(),
            }
        } else {
            Default::default()
        }
    }
}

// Retain price differences smaller than one ulp of the square root. Multiplying
// a rounded sqrt difference by large liquidity otherwise quantizes the gradient.
fn sqrt_price_gap(v: [f64; 2], fee: f64, token0_in: bool, start: f64) -> (f64, f64) {
    let (a_factor, b_factor) = if token0_in { (1.0, fee) } else { (fee, 1.0) };
    let target = ((v[0] / v[1]) * a_factor / b_factor).sqrt();
    let a = v[0] * a_factor;
    let a_error = v[0].mul_add(a_factor, -a);
    let b0 = v[1] * b_factor;
    let b0_error = v[1].mul_add(b_factor, -b0);
    let b1 = b0 * start;
    let b1_error = b0.mul_add(start, -b1) + b0_error * start;
    let b = b1 * start;
    let b_error = b1.mul_add(start, -b) + b1_error * start;
    let gap = ((a - b) + (a_error - b_error)) / (b0 * (target + start));
    (target, gap)
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

    fn evaluate(&self, v: [f64; 2], fee: f64, token0_in: bool) -> MarketEvaluation {
        let (target, gap) = sqrt_price_gap(v, fee, token0_in, self.p0);
        let (distance, capacity, weight) = if token0_in {
            (-gap, self.p0 - self.p1, v[1])
        } else {
            (gap, self.p1 - self.p0, v[1] / fee)
        };
        if distance <= 0.0 {
            return MarketEvaluation::default();
        }
        let (_, end_gap) = sqrt_price_gap(v, fee, token0_in, self.p1);
        let exhausted = if token0_in {
            end_gap <= 0.0
        } else {
            end_gap >= 0.0
        };
        let d = if exhausted { capacity } else { distance };
        let end = if exhausted { self.p1 } else { target };
        let token1 = self.k * d;
        let token0 = token1 / self.p0 / end;
        // The interior value is quadratic in the accurate price gap. An
        // exhausted range adds its nonnegative marginal profit at the boundary.
        let boundary = if exhausted {
            end_gap.abs() * (end + target) / end
        } else {
            0.0
        };
        let value = self.k * weight * d * (d + boundary) / self.p0;
        if token0_in {
            MarketEvaluation {
                input: [token0 / fee, 0.0],
                output: [0.0, token1],
                value,
            }
        } else {
            MarketEvaluation {
                input: [0.0, token1 / fee],
                output: [token0, 0.0],
                value,
            }
        }
    }

    /// p: sqrt(p / gamma)
    pub fn arbitrage_pos(&self, p: f64) -> (f64, f64) {
        let delta1 = self.k * (self.p0 - p) / self.p0 / p;
        if p >= self.p0 {
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
        if p <= self.p0 {
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
    fn retains_sub_ulp_sqrt_price_gaps_in_both_directions() {
        let liquidity = 1e12;
        let pool = UniswapV3::new(1.0, vec![2.0, 0.5], vec![liquidity, 0.0], 1.0);
        for v0 in [1.0 - f64::EPSILON / 2.0, 1.0 + f64::EPSILON] {
            let expected_gap = (v0 - 1.0) / (v0.sqrt() + 1.0);
            let trade = pool.evaluate([v0, 1.0]);
            let token1 = if v0 < 1.0 {
                trade.output[1]
            } else {
                trade.input[1]
            };
            assert!((token1 / (liquidity * expected_gap.abs()) - 1.0).abs() < 1e-15);
            let expected_value = liquidity * expected_gap.powi(2);
            assert!((trade.value / expected_value - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn very_small_target_does_not_exhaust_zero_price_boundary() {
        let pool = UniswapV3::new(1.0, vec![2.0, 0.0], vec![1.0, 0.0], 1.0);
        let trade = pool.evaluate([1e-40, 1.0]);
        assert!(trade.input[0].is_finite());
        assert!((trade.input[0] / 1e20 - 1.0).abs() < 1e-15);
        assert!((trade.output[1] - 1.0).abs() < 1e-15);
        assert!((trade.value - 1.0).abs() < 1e-15);
    }

    #[test]
    fn joint_values_match_trades_across_fees_and_tick_boundaries() {
        for fee in [1.0, 0.997] {
            let pool = UniswapV3::new(
                1.0,
                vec![1.2, 1.05, 0.95, 0.8],
                vec![3000.0, 2000.0, 1000.0, 0.0],
                fee,
            );
            for v0 in [0.1, 0.7, 0.9, 1.0, 1.1, 1.5, 3.0] {
                let trade = pool.evaluate([v0, 1.0]);
                let value =
                    v0 * (trade.output[0] - trade.input[0]) + trade.output[1] - trade.input[1];
                assert!(trade.value >= 0.0);
                assert!(
                    (trade.value - value).abs() <= 1e-10 * trade.value.max(1.0),
                    "fee={fee} v0={v0}: {trade:?} vs {value}"
                );
                if v0 != 1.0 {
                    let h = v0 * 1e-5;
                    let numerical = (pool.evaluate([v0 + h, 1.0]).value
                        - pool.evaluate([v0 - h, 1.0]).value)
                        / (2.0 * h);
                    let gradient = trade.output[0] - trade.input[0];
                    assert!(
                        (numerical - gradient).abs() <= 1e-7 * gradient.abs().max(1.0),
                        "fee={fee} v0={v0}: {numerical} vs {gradient}"
                    );
                }
            }
        }
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
