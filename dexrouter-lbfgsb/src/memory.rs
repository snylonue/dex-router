use std::sync::OnceLock;

use num_traits::Float;

use crate::{axpy, dot};

#[cfg(test)]
#[path = "memory_tests.rs"]
mod tests;

#[derive(Debug, Clone)]
struct Correction<T> {
    s: Vec<T>,
    y: Vec<T>,
    ys: T,
}

#[derive(Debug, Clone)]
struct HessianUpdate<T> {
    // B_i s_i and its curvature, where B_i includes only earlier corrections.
    bs: Vec<T>,
    sbs: T,
}

#[derive(Debug, Clone)]
pub(crate) struct LbfgsMemory<T> {
    capacity: usize,
    pairs: Vec<Correction<T>>,
    theta: T,
    // Depends only on the correction history, never on the product's input or
    // the free-variable mask. OnceLock preserves shared-read thread safety.
    hessian_cache: OnceLock<Vec<HessianUpdate<T>>>,
}

impl<T: Float> LbfgsMemory<T> {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            pairs: Vec::with_capacity(capacity),
            theta: T::one(),
            hessian_cache: OnceLock::new(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.pairs.clear();
        self.theta = T::one();
        self.hessian_cache.take();
    }

    pub(crate) fn push(&mut self, s: Vec<T>, y: Vec<T>) -> bool {
        let ys = dot(&y, &s);
        let yy = dot(&y, &y);
        let scale = dot(&s, &s).sqrt() * yy.sqrt();
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
        self.theta = yy / ys;
        // Even without eviction, a new theta changes every B_i s_i. Appending
        // just one cached update would leave the old prefixes inconsistent.
        self.hessian_cache.take();
        true
    }

    pub(crate) fn theta(&self) -> T {
        self.theta
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

    /// O(m n) application after a lazy O(m^2 n) rebuild per memory state.
    pub(crate) fn hessian_product(&self, v: &[T]) -> Vec<T> {
        let cache = self
            .hessian_cache
            .get_or_init(|| self.build_hessian_cache());
        let mut out = v
            .iter()
            .map(|&value| self.theta * value)
            .collect::<Vec<_>>();
        for (pair, update) in self.pairs.iter().zip(cache) {
            apply_bfgs_update(&mut out, v, pair, update);
        }
        out
    }

    fn build_hessian_cache(&self) -> Vec<HessianUpdate<T>> {
        let mut cache = Vec::with_capacity(self.pairs.len());
        for pair in &self.pairs {
            let mut bs = pair
                .s
                .iter()
                .map(|&value| self.theta * value)
                .collect::<Vec<_>>();
            for (prev, update) in self.pairs.iter().zip(&cache) {
                apply_bfgs_update(&mut bs, &pair.s, prev, update);
            }
            let sbs = dot(&pair.s, &bs);
            cache.push(HessianUpdate { bs, sbs });
        }
        cache
    }
}

fn apply_bfgs_update<T: Float>(
    out: &mut [T],
    original_v: &[T],
    pair: &Correction<T>,
    update: &HessianUpdate<T>,
) {
    if update.sbs > T::zero() {
        let alpha = dot(&pair.s, out) / update.sbs;
        axpy(out, -alpha, &update.bs);
    }

    if pair.ys > T::zero() {
        let alpha = dot(&pair.y, original_v) / pair.ys;
        axpy(out, alpha, &pair.y);
    }
}
