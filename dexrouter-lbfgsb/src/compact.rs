//! Small, pivoted direct solves for B/theta = I + U J U^T.
//! J has alternating -1/+1 entries. No dense n-by-n matrix is formed.
use std::sync::Mutex;

use num_traits::Float;

use crate::{all_finite, dot};

#[derive(Debug, Clone)]
struct ReducedSystem<T> {
    free: Vec<usize>,
    // Cache failures too: the same matrix should not be repeatedly factored.
    factor: Option<Lu<T>>,
}

#[derive(Debug)]
pub(crate) struct CompactModel<T> {
    basis: Vec<Vec<T>>,
    gram: Vec<T>,
    reduced: Mutex<Option<ReducedSystem<T>>>,
}

impl<T: Clone> Clone for CompactModel<T> {
    fn clone(&self) -> Self {
        Self {
            basis: self.basis.clone(),
            gram: self.gram.clone(),
            reduced: Mutex::new(self.reduced.lock().ok().and_then(|value| value.clone())),
        }
    }
}

impl<T: Float> CompactModel<T> {
    pub(crate) fn new(basis: Vec<Vec<T>>) -> Option<Self> {
        let dimension = basis.first()?.len();
        if basis.len() % 2 != 0 || basis.iter().any(|v| v.len() != dimension || !all_finite(v)) {
            return None;
        }
        let rank = basis.len();
        let mut gram = vec![T::zero(); rank * rank];
        for i in 0..rank {
            for j in 0..=i {
                let value = dot(&basis[i], &basis[j]);
                if !value.is_finite() {
                    return None;
                }
                gram[i * rank + j] = value;
                gram[j * rank + i] = value;
            }
        }
        Some(Self {
            basis,
            gram,
            reduced: Mutex::new(None),
        })
    }

    fn factor(&self, free: &[usize]) -> Option<Lu<T>> {
        let rank = self.basis.len();
        let dimension = self.basis[0].len();
        // Use whichever side of the mask is smaller. When only one coordinate
        // is active, removing its outer product avoids scanning n free entries
        // for every matrix element. The full Gram belongs to the memory state.
        let active: Vec<_> = (0..dimension)
            .filter(|i| free.binary_search(i).is_err())
            .collect();
        let subtract = active.len() < free.len();
        let indices = if subtract { active.as_slice() } else { free };
        let mut matrix = vec![T::zero(); rank * rank];
        for i in 0..rank {
            for j in 0..=i {
                let restricted = indices.iter().fold(T::zero(), |sum, &k| {
                    sum + self.basis[i][k] * self.basis[j][k]
                });
                let mut value = if subtract {
                    self.gram[i * rank + j] - restricted
                } else {
                    restricted
                };
                if i == j {
                    value = value + if i % 2 == 0 { -T::one() } else { T::one() };
                }
                matrix[i * rank + j] = value;
                matrix[j * rank + i] = value;
            }
        }
        Lu::factor(matrix, rank)
    }

    /// Apply the Woodbury inverse on F. Inactive coordinates are exactly zero.
    /// The caller checks the residual against the original BFGS model.
    pub(crate) fn solve(&self, theta: T, rhs: &[T], free: &[usize]) -> Option<Vec<T>> {
        let dimension = self.basis[0].len();
        if rhs.len() != dimension
            || free.last().is_some_and(|&i| i >= dimension)
            || free.windows(2).any(|w| w[0] >= w[1])
            || !theta.is_finite()
            || theta <= T::zero()
        {
            return None;
        }
        let projected: Vec<_> = self
            .basis
            .iter()
            .map(|column| {
                free.iter()
                    .fold(T::zero(), |sum, &i| sum + column[i] * rhs[i])
            })
            .collect();
        let coefficients = {
            let mut cached = self.reduced.lock().ok()?;
            if cached.as_ref().is_none_or(|system| system.free != free) {
                *cached = Some(ReducedSystem {
                    free: free.to_vec(),
                    factor: self.factor(free),
                });
            }
            cached.as_ref()?.factor.as_ref()?.solve(&projected)?
        };
        let mut out = vec![T::zero(); dimension];
        for &i in free {
            let correction = self
                .basis
                .iter()
                .zip(&coefficients)
                .fold(T::zero(), |sum, (u, &z)| sum + u[i] * z);
            out[i] = (rhs[i] - correction) / theta;
        }
        all_finite(&out).then_some(out)
    }
}

/// Row-equilibrated LU with partial pivoting: the compact matrix is indefinite,
/// so an ordinary SPD Cholesky factorization is not appropriate here.
#[derive(Debug, Clone)]
struct Lu<T> {
    matrix: Vec<T>,
    scales: Vec<T>,
    pivots: Vec<usize>,
}

impl<T: Float> Lu<T> {
    fn factor(mut matrix: Vec<T>, n: usize) -> Option<Self> {
        if n == 0 || n.checked_mul(n)? != matrix.len() {
            return None;
        }
        let mut scales = Vec::with_capacity(n);
        for row in matrix.chunks_mut(n) {
            let scale = row.iter().fold(T::zero(), |s, &v| s.max(v.abs()));
            if !scale.is_finite() || scale == T::zero() || !all_finite(row) {
                return None;
            }
            scales.push(scale);
            for value in row {
                *value = *value / scale;
            }
        }
        let threshold = T::epsilon() * T::from(n)?;
        let mut pivots = Vec::with_capacity(n);
        for k in 0..n {
            let pivot = (k..n).max_by(|&i, &j| {
                matrix[i * n + k]
                    .abs()
                    .partial_cmp(&matrix[j * n + k].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;
            let value = matrix[pivot * n + k];
            if !value.is_finite() || value.abs() <= threshold {
                return None;
            }
            pivots.push(pivot);
            for j in 0..n {
                matrix.swap(k * n + j, pivot * n + j);
            }
            for i in k + 1..n {
                let multiplier = matrix[i * n + k] / matrix[k * n + k];
                matrix[i * n + k] = multiplier;
                for j in k + 1..n {
                    matrix[i * n + j] = matrix[i * n + j] - multiplier * matrix[k * n + j];
                }
            }
        }
        all_finite(&matrix).then_some(Self {
            matrix,
            scales,
            pivots,
        })
    }

    fn solve(&self, rhs: &[T]) -> Option<Vec<T>> {
        let n = self.scales.len();
        if rhs.len() != n {
            return None;
        }
        let mut x: Vec<_> = rhs.iter().zip(&self.scales).map(|(&r, &s)| r / s).collect();
        for (i, &pivot) in self.pivots.iter().enumerate() {
            x.swap(i, pivot);
        }
        for i in 0..n {
            for j in 0..i {
                x[i] = x[i] - self.matrix[i * n + j] * x[j];
            }
        }
        for i in (0..n).rev() {
            for j in i + 1..n {
                x[i] = x[i] - self.matrix[i * n + j] * x[j];
            }
            x[i] = x[i] / self.matrix[i * n + i];
        }
        all_finite(&x).then_some(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pivoted_lu_handles_indefinite_scaled_systems() {
        let matrix = vec![0_f64, 2e-12, 3e12, 4e12];
        let factor = Lu::factor(matrix, 2).unwrap();
        let result = factor.solve(&[4e-12, 11e12]).unwrap();
        assert!((result[0] - 1.).abs() < 1e-12);
        assert!((result[1] - 2.).abs() < 1e-12);
        assert!(Lu::factor(vec![1_f64, 2., 2., 4.], 2).is_none());
        assert!(Lu::factor(vec![f64::NAN], 1).is_none());
        assert!(Lu::factor(Vec::<f64>::new(), 0).is_none());
        assert!(Lu::factor(vec![1_f64], 2).is_none());

        let singular = CompactModel::new(vec![vec![1_f64], vec![0.]]).unwrap();
        for _ in 0..2 {
            assert!(singular.solve(1., &[1.], &[0]).is_none());
        }
        assert!(
            singular
                .reduced
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .factor
                .is_none()
        );
        assert_eq!(singular.solve(1., &[1.], &[]).unwrap(), [0.]);
    }

    #[test]
    fn factorization_is_reused_and_mask_changes_rebuild_it() {
        let model = CompactModel::new(vec![vec![0.5_f64, 0.2, 0.1], vec![1., 0.3, 0.2]]).unwrap();
        model.solve(1., &[1., 2., 3.], &[0, 1]).unwrap();
        let old = model
            .reduced
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .factor
            .as_ref()
            .unwrap()
            .matrix
            .as_ptr();
        model.solve(1., &[3., 2., 1.], &[0, 1]).unwrap();
        assert_eq!(
            old,
            model
                .reduced
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .factor
                .as_ref()
                .unwrap()
                .matrix
                .as_ptr()
        );
        model.solve(1., &[1., 2., 3.], &[1, 2]).unwrap();
        assert_eq!(model.reduced.lock().unwrap().as_ref().unwrap().free, [1, 2]);
        assert!(model.solve(1., &[1., 2., 3.], &[2, 1]).is_none());
    }
}
