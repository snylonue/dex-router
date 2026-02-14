pub trait Market {
    fn arbitrage(&self, v: [f64; 2]) -> ([f64; 2], [f64; 2]);
}

#[derive(Debug)]
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
            current_tick: {
                let reversed = {
                    let mut r = lower_prices.clone();
                    r.reverse();
                    r
                };
                match reversed.binary_search_by(|&f| f.total_cmp(&current_price)) {
                    Ok(idx) => idx,
                    Err(idx) => idx - 1,
                }
            },
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
                let alpha = k / pu;
                let beta = k * pl;
                let p_cur = if i > 0 { pu } else { self.current_price };
                let range =
                    BoundedLiquidity::new(k, alpha, beta, k / p_cur - alpha, k * p_cur - beta);
                let (delta0, delta1) = range.arbitrage_pos(p / self.fee);
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
                let alpha = k / pu;
                let beta = k * pl;
                let p_cur = if i < self.current_tick {
                    pl
                } else {
                    self.current_price
                };
                let range = BoundedLiquidity::new(
                    k,
                    beta,
                    alpha,
                    k * p_cur - beta,
                    k / p_cur - alpha,
                );
                let (delta0, delta1) = range.arbitrage_pos(1.0 / (self.fee * p));
                if !initial && (delta0.abs() <= f64::EPSILON || delta1.abs() <= f64::EPSILON) {
                    break;
                }
                initial = false;
                input[1] += delta0;
                output[0] += delta1;
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
    alpha: f64,
    beta: f64,
    r1: f64,
    r2: f64,
}

impl BoundedLiquidity {
    pub fn new(k: f64, alpha: f64, beta: f64, r1: f64, r2: f64) -> Self {
        debug_assert!(k - (r1 + alpha) * (r2 + beta) <= f64::EPSILON);
        Self {
            k,
            alpha,
            beta,
            r1,
            r2,
        }
    }

    pub fn arbitrage_pos(&self, p: f64) -> (f64, f64) {
        let delta1 = (self.k / p.sqrt()) - (self.r1 + self.alpha);
        if delta1 <= 0.0 {
            Default::default()
        } else {
            let delta1_max = self.k.powi(2) / self.beta - (self.r1 + self.alpha);
            if delta1 >= delta1_max {
                (delta1_max, self.r2)
            } else {
                let delta2 = (self.r2 + self.beta) - (self.k * p.sqrt());
                (delta1, delta2)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::{Market, UniswapV3};

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

        assert!(allclose(input, [0.0, 1.371820235138894]));
        assert!(allclose(output, [0.07222755758392213, 0.0]))
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
