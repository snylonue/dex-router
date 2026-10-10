use num_traits::Float;

use crate::{axpy, dot};

#[derive(Debug, Clone)]
struct Correction<T> {
    s: Vec<T>,
    y: Vec<T>,
    ys: T,
}

#[derive(Debug, Clone)]
pub(crate) struct LbfgsMemory<T> {
    capacity: usize,
    pairs: Vec<Correction<T>>,
}

impl<T: Float> LbfgsMemory<T> {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            pairs: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.pairs.clear();
    }

    pub(crate) fn push(&mut self, s: Vec<T>, y: Vec<T>) -> bool {
        let ys = dot(&y, &s);
        let scale = dot(&s, &s).sqrt() * dot(&y, &y).sqrt();
        if !ys.is_finite() || !scale.is_finite() || ys <= T::epsilon() * scale {
            return false;
        }
        if self.capacity == 0 {
            return false;
        }
        if self.pairs.len() == self.capacity {
            self.pairs.remove(0);
        }
        self.pairs.push(Correction { s, y, ys });
        true
    }

    pub(crate) fn theta(&self) -> T {
        self.pairs.last().map_or(T::one(), |pair| {
            let yy = dot(&pair.y, &pair.y);
            if pair.ys > T::zero() {
                yy / pair.ys
            } else {
                T::one()
            }
        })
    }

    /// Stable O(m n) two-loop application of the inverse BFGS model.
    pub(crate) fn inverse_hessian_product(&self, v: &[T]) -> Vec<T> {
        let mut q = v.to_vec();
        let mut alphas = vec![T::zero(); self.pairs.len()];
        for (i, pair) in self.pairs.iter().enumerate().rev() {
            alphas[i] = dot(&pair.s, &q) / pair.ys;
            axpy(&mut q, -alphas[i], &pair.y);
        }
        let theta = self.theta();
        let mut r: Vec<T> = q.iter().map(|&value| value / theta).collect();
        for (i, pair) in self.pairs.iter().enumerate() {
            let beta = dot(&pair.y, &r) / pair.ys;
            axpy(&mut r, alphas[i] - beta, &pair.s);
        }
        r
    }

    pub(crate) fn hessian_product(&self, v: &[T]) -> Vec<T> {
        self.hessian_product_prefix(v, self.pairs.len())
    }

    fn hessian_product_prefix(&self, v: &[T], end: usize) -> Vec<T> {
        let theta = self.theta();
        let mut bs_cache: Vec<Vec<T>> = Vec::with_capacity(end);
        for i in 0..end {
            let pair = &self.pairs[i];
            let mut bs = pair
                .s
                .iter()
                .map(|&value| theta * value)
                .collect::<Vec<_>>();
            for (j, prev_bs) in bs_cache.iter().enumerate() {
                let prev = &self.pairs[j];
                apply_bfgs_update(&mut bs, &pair.s, prev, prev_bs);
            }
            bs_cache.push(bs);
        }

        let mut out = v.iter().map(|&value| theta * value).collect::<Vec<_>>();
        for (pair, bs) in self.pairs.iter().take(end).zip(bs_cache.iter()) {
            apply_bfgs_update(&mut out, v, pair, bs);
        }
        out
    }
}

fn apply_bfgs_update<T: Float>(out: &mut [T], original_v: &[T], pair: &Correction<T>, bs: &[T]) {
    let sbs = dot(&pair.s, bs);
    if sbs > T::zero() {
        let alpha = dot(&pair.s, out) / sbs;
        axpy(out, -alpha, bs);
    }

    if pair.ys > T::zero() {
        let alpha = dot(&pair.y, original_v) / pair.ys;
        axpy(out, alpha, &pair.y);
    }
}
