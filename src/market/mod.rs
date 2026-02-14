pub trait Market {
    fn arbitrage(&self, v: [f64; 2]) -> ([f64; 2], [f64; 2]);
}

#[derive(Debug)]
pub struct UniswapV3 {
    current_price: f64,
    current_tick: usize,
    lower_ticks: Vec<i32>,
    // sqrt of liquidity
    liquidity: Vec<f64>,
    fee: f64,
}

impl UniswapV3 {
    pub fn new(
        current_price: f64,
        current_tick: usize,
        lower_ticks: Vec<i32>,
        liquidity: Vec<f64>,
        fee: f64,
    ) -> Self {
        Self {
            current_price,
            current_tick,
            lower_ticks,
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
            let prices = self.lower_ticks[self.current_tick..]
                .iter()
                .map(|&tick| 1.0001_f64.powi(tick))
                .collect::<Vec<_>>();
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
                let alpha = k / pu.sqrt();
                let beta = k * pl.sqrt();
                let p_cur = if i > 0 { pu } else { p0 };
                let range = BoundedLiquidity::new(
                    k,
                    alpha,
                    beta,
                    k / p_cur.sqrt() - alpha,
                    k * p_cur.sqrt() - beta,
                );
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
            let prices = self.lower_ticks[1..=self.current_tick + 1]
                .iter()
                .map(|&tick| 1.0001_f64.powi(tick))
                .collect::<Vec<_>>();
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
                    1.0001_f64.powi(self.lower_ticks[0])
                } else {
                    prices[i - 1]
                };
                let alpha = k / pu.sqrt();
                let beta = k * pl.sqrt();
                let p_cur = if i < self.current_tick { pl } else { p0 };
                let range = BoundedLiquidity::new(
                    k,
                    beta,
                    alpha,
                    k * p_cur.sqrt() - beta,
                    k / p_cur.sqrt() - alpha,
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
        arr1(&x).abs_diff_eq(&arr1(&y), 1e-10)
    }

    #[test]
    fn uniswap_v3_arb_neg() {
        let pool = UniswapV3::new(
            3.872983346207417,
            1,
            vec![34014, 29959, 23027, 16095],
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
            1,
            vec![34014, 29959, 23027, 16095],
            vec![1.0, 1.4142135623730951, 1.224744871391589, 0.0],
            0.997,
        );

        let (input, output) = pool.arbitrage([10.0, 1.0]);

        assert!(allclose(input, [0.08163881601325118, 0.0]));
        assert!(allclose(output, [0.0, 0.9983662848277683]))
    }
}
