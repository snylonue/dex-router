use ndarray::Array2;

pub trait Market {
    fn arbitrage(&self, v: [f64; 2]) -> ([f64; 2], [f64; 2]);
}

#[derive(Debug)]
pub struct UniswapV3 {
    current_price: f64,
    current_tick: usize,
    lower_ticks: Vec<f64>,
    liquidity: Vec<f64>,
    fee: f64,
}

impl UniswapV3 {}

impl Market for UniswapV3 {
    fn arbitrage(&self, v: [f64; 2]) -> ([f64; 2], [f64; 2]) {
        let p = v[0] / v[1];

        if self.current_price * self.fee <= p && p <= self.current_price / self.fee {
            Default::default()
        } else if p < self.current_price * self.fee {
            let mut input = [0.0; 2];
            let mut output = [0.0; 2];

            todo!()
        } else {
            todo!()
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
    // SAFETY: invarient k = (r1 + alpha)(r2 + beta)
    pub unsafe fn new_unchecked(k: f64, alpha: f64, beta: f64, r1: f64, r2: f64) -> Self {
        Self {
            k,
            alpha,
            beta,
            r1,
            r2,
        }
    }

    pub fn new(k: f64, alpha: f64, beta: f64, r1: f64, r2: f64) -> Self {
        assert!(k - (r1 + alpha) * (r2 + beta) <= f64::EPSILON);
        unsafe { Self::new_unchecked(k, alpha, beta, r1, r2) }
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
