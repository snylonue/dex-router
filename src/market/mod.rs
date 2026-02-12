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

impl UniswapV3 {}

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
                let alpha = k / pu;
                let beta = k * pu;
                let p_cur = if i > 0 { pu } else { p0 };
                let range =
                    BoundedLiquidity::new(k, alpha, beta, k / p_cur - alpha, k * p_cur + beta);
                let (delta0, delta1) = range.arbitrage_pos(p0 / self.fee);
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
                let alpha = k / pl;
                let beta = k * pl;
                let p_cur = if i < self.current_tick { pl } else { p0 };
                let range =
                    BoundedLiquidity::new(k, beta, alpha, k * p_cur + beta, k / p_cur - alpha);
                let (delta0, delta1) = range.arbitrage_pos(1.0 / (self.fee * p0));
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
        let delta1 = (self.k / p).sqrt() - (self.r1 + self.alpha);
        if delta1 <= 0.0 {
            Default::default()
        } else {
            let delta1_max = self.k / self.beta - (self.r1 + self.alpha);
            if delta1 >= delta1_max {
                (delta1_max, self.r2)
            } else {
                let delta2 = (self.r2 + self.beta) - (self.k * p).sqrt();
                (delta1, delta2)
            }
        }
    }
}
