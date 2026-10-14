use std::sync::OnceLock;

use num_traits::Float;

use crate::{all_finite, axpy, compact::CompactModel, dot, norm_inf};

#[cfg(test)]
#[path = "memory_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "memory_compact_tests.rs"]
mod compact_tests;

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
    compact_cache: OnceLock<Option<CompactModel<T>>>,
}

impl<T: Float> LbfgsMemory<T> {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            pairs: Vec::with_capacity(capacity),
            theta: T::one(),
            hessian_cache: OnceLock::new(),
            compact_cache: OnceLock::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn hessian_cache_is_initialized(&self) -> bool {
        self.hessian_cache.get().is_some()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    pub(crate) fn clear(&mut self) {
        self.pairs.clear();
        self.theta = T::one();
        self.hessian_cache.take();
        self.compact_cache.take();
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
        let theta = yy / ys;
        // Finite positive curvature alone does not guarantee a representable
        // diagonal: yy can underflow, or yy/ys can overflow. Reject before
        // evicting history or invalidating its caches.
        if !theta.is_finite() || theta <= T::zero() {
            return false;
        }
        if self.pairs.len() == self.capacity {
            self.pairs.remove(0);
        }
        self.pairs.push(Correction { s, y, ys });
        self.theta = theta;
        // Even without eviction, a new theta changes every B_i s_i. Appending
        // just one cached update would leave the old prefixes inconsistent.
        self.hessian_cache.take();
        self.compact_cache.take();
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

    /// O(m) diagonal of the symmetric rank-two model after the shared lazy rebuild.
    /// Divide before multiplying: squaring a finite y_i or (B s)_i can overflow.
    pub(crate) fn hessian_diagonal_at(&self, i: usize) -> T {
        let cache = self
            .hessian_cache
            .get_or_init(|| self.build_hessian_cache());
        let mut diagonal = self.theta;
        for (pair, update) in self.pairs.iter().zip(cache) {
            if update.sbs > T::zero() {
                diagonal = diagonal - (update.bs[i] / update.sbs) * update.bs[i];
            }
            if pair.ys > T::zero() {
                diagonal = diagonal + (pair.y[i] / pair.ys) * pair.y[i];
            }
        }
        diagonal
    }

    /// Solve B_FF step_F = rhs_F with a matrix of order at most 2m.
    /// Reject unreliable compact solves so the caller can use the CG safeguard.
    pub(crate) fn solve_reduced(&self, rhs: &[T], free: &[usize]) -> Option<Vec<T>> {
        if !all_finite(rhs)
            || free.last().is_some_and(|&i| i >= rhs.len())
            || free.windows(2).any(|w| w[0] >= w[1])
        {
            return None;
        }
        let rhs_norm = free
            .iter()
            .fold(T::zero(), |norm, &i| norm.max(rhs[i].abs()));
        if rhs_norm == T::zero() {
            return Some(vec![T::zero(); rhs.len()]);
        }
        if self.is_empty() {
            let mut step = vec![T::zero(); rhs.len()];
            for &i in free {
                step[i] = rhs[i];
            }
            return Some(step);
        }
        let model = self
            .compact_cache
            .get_or_init(|| self.build_compact_model())
            .as_ref()?;
        let mut step = model.solve(self.theta, rhs, free)?;
        let tolerance = T::epsilon().sqrt() * rhs_norm;
        // Check against the existing sequential BFGS arithmetic, not just the
        // low-rank reconstruction. Two refinement steps reuse the same factor.
        for refinement in 0..=2 {
            let product = self.hessian_product(&step);
            let mut residual = vec![T::zero(); rhs.len()];
            for &i in free {
                residual[i] = rhs[i] - product[i];
            }
            if !all_finite(&residual) {
                return None;
            }
            if norm_inf(&residual) <= tolerance {
                return Some(step);
            }
            if refinement == 2 {
                break;
            }
            let correction = model.solve(self.theta, &residual, free)?;
            axpy(&mut step, T::one(), &correction);
            if !all_finite(&step) {
                return None;
            }
        }
        None
    }

    fn build_compact_model(&self) -> Option<CompactModel<T>> {
        if !self.theta.is_finite() || self.theta <= T::zero() {
            return None;
        }
        let cache = self
            .hessian_cache
            .get_or_init(|| self.build_hessian_cache());
        let root_theta = self.theta.sqrt();
        let mut basis = Vec::with_capacity(2 * self.pairs.len());
        for (pair, update) in self.pairs.iter().zip(cache) {
            if !update.sbs.is_finite() || update.sbs <= T::zero() {
                return None;
            }
            let root_sbs = update.sbs.sqrt();
            let root_ys = pair.ys.sqrt();
            // B/theta = I + U J U^T, J = diag(-1,+1,...).
            // Normalize the columns to avoid raw s/y magnitude disparities.
            basis.push(
                update
                    .bs
                    .iter()
                    .map(|&v| (v / root_sbs) / root_theta)
                    .collect(),
            );
            basis.push(pair.y.iter().map(|&v| (v / root_ys) / root_theta).collect());
        }
        CompactModel::new(basis)
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
