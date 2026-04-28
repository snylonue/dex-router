use num_traits::Float;
use serde::{Deserialize, Serialize};

pub trait Market<F: Float> {
    fn arbitrage(&self, v: [F; 2]) -> ([F; 2], [F; 2]);
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct UniswapV2<F: Float> {
    reserves: [F; 2],
    fee: F,
}

impl<F: Float> UniswapV2<F> {
    pub fn new(reserve0: F, reserve1: F, fee: F) -> Self {
        debug_assert!(reserve0 >= F::zero());
        debug_assert!(reserve1 >= F::zero());
        debug_assert!(fee > F::zero() && fee <= F::one());
        Self {
            reserves: [reserve0, reserve1],
            fee,
        }
    }

    fn arb_in(&self, p: F, r: F) -> F {
        let k = self.reserves[0] * self.reserves[1];
        (((self.fee * p * k).sqrt() - r) / self.fee).max(F::zero())
    }

    fn arb_out(&self, p: F, r: F) -> F {
        let k = self.reserves[0] * self.reserves[1];
        (r - (k / (p * self.fee)).sqrt()).max(F::zero())
    }
}

impl<F: Float> Market<F> for UniswapV2<F> {
    fn arbitrage(&self, v: [F; 2]) -> ([F; 2], [F; 2]) {
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
pub struct UniswapV3<F: Float> {
    current_price: F,
    current_tick: usize,
    lower_prices: Vec<F>,
    liquidity: Vec<F>,
    fee: F,
}

impl<F: Float> UniswapV3<F> {
    pub fn new(current_price: F, lower_prices: Vec<F>, liquidity: Vec<F>, fee: F) -> Self {
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

impl<F: Float> Market<F> for UniswapV3<F> {
    fn arbitrage(&self, v: [F; 2]) -> ([F; 2], [F; 2]) {
        let p = v[0] / v[1];
        let p0 = self.current_price.powi(2);

        if p < p0 * self.fee {
            let prices = &self.lower_prices[self.current_tick..];
            let liquidity = &self.liquidity[self.current_tick..];
            let mut input = [F::zero(); 2];
            let mut output = [F::zero(); 2];
            let mut initial = true;

            for i in 0..self.liquidity.len() - self.current_tick {
                let k = liquidity[i];
                if k.abs() <= F::epsilon() {
                    initial = false;
                    continue;
                }
                let pu = prices[i];
                let pl = prices.get(i + 1).copied().unwrap_or(F::zero());
                let p_cur = if i > 0 { pu } else { self.current_price };
                let range = BoundedLiquidity::new(k, p_cur, pl);
                let (delta0, delta1) = range.arbitrage_pos((p / self.fee).sqrt());
                if !initial && (delta0.abs() <= F::epsilon() || delta1.abs() <= F::epsilon()) {
                    break;
                }
                initial = false;
                input[0] = input[0] + delta0;
                output[1] = output[1] + delta1;
            }
            input[0] = input[0] / self.fee;
            (input, output)
        } else if p > p0 / self.fee {
            let prices = &self.lower_prices[1..=self.current_tick + 1];
            let liquidity = &self.liquidity[..=self.current_tick];
            let mut input = [F::zero(); 2];
            let mut output = [F::zero(); 2];
            let mut initial = true;

            for i in (0..=self.current_tick).rev() {
                let k = liquidity[i];
                if k.abs() <= F::epsilon() {
                    initial = false;
                    continue;
                }
                let pl = prices[i];
                let pu = if i == 0 {
                    self.lower_prices[0]
                } else {
                    prices[i - 1]
                };
                let p_cur = if i < self.current_tick { pl } else { self.current_price };
                let range = BoundedLiquidity::new(k, p_cur, pu);
                let (delta0, delta1) = range.arbitrage_neg((p * self.fee).sqrt());
                if !initial && (delta0.abs() <= F::epsilon() || delta1.abs() <= F::epsilon()) {
                    break;
                }
                initial = false;
                input[1] = input[1] + delta1;
                output[0] = output[0] + delta0;
            }
            input[1] = input[1] / self.fee;
            (input, output)
        } else {
            ([F::zero(), F::zero()], [F::zero(), F::zero()])
        }
    }
}

#[derive(Debug)]
pub struct BoundedLiquidity<F: Float> {
    k: F,
    p0: F,
    p1: F,
}

impl<F: Float> BoundedLiquidity<F> {
    pub fn new(k: F, p0: F, p1: F) -> Self {
        Self { k, p0, p1 }
    }

    pub fn arbitrage_pos(&self, p: F) -> (F, F) {
        let delta1 = self.k * (self.p0 - p) / self.p0 / p;
        if delta1 <= F::epsilon() {
            (F::zero(), F::zero())
        } else {
            let delta1_max = self.k * ((self.p0 - self.p1) / self.p0 / self.p1);
            if delta1 >= delta1_max {
                (delta1_max, self.k * (self.p0 - self.p1))
            } else {
                (delta1, self.k * (self.p0 - p))
            }
        }
    }

    pub fn arbitrage_neg(&self, p: F) -> (F, F) {
        let delta2 = self.k * (p - self.p0);
        if delta2 <= F::epsilon() {
            (F::zero(), F::zero())
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
