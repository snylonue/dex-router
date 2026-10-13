//! Sparse graph operations used only for equivalent coordinates and step models.

/// Deterministic connected components, including singleton vertices.
pub(crate) fn components(
    n: usize,
    edges: impl IntoIterator<Item = (usize, usize)>,
) -> Vec<Vec<usize>> {
    let mut parent: Vec<_> = (0..n).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for (a, b) in edges {
        let a = root(&mut parent, a);
        let b = root(&mut parent, b);
        parent[a.max(b)] = a.min(b);
    }
    let mut groups = vec![Vec::new(); n];
    for i in 0..n {
        let r = root(&mut parent, i);
        groups[r].push(i);
    }
    groups.into_iter().filter(|g| !g.is_empty()).collect()
}

/// Jacobi-preconditioned CG. An approximate iterate is useful only if the caller
/// verifies it against the original objective; no matrix damping changes that objective.
pub(crate) fn pcg(
    diagonal: &[f64],
    rhs: &[f64],
    mut apply: impl FnMut(&[f64]) -> Vec<f64>,
) -> Option<Vec<f64>> {
    if diagonal.len() != rhs.len()
        || diagonal.iter().any(|d| !d.is_finite() || *d <= 0.0)
        || rhs.iter().any(|r| !r.is_finite())
    {
        return None;
    }
    let dot = |a: &[f64], b: &[f64]| {
        let mut sum = crate::numerics::Sum::default();
        for (a, b) in a.iter().zip(b) {
            sum.add(a * b);
        }
        sum.total()
    };
    let mut x = vec![0.0; rhs.len()];
    let mut r = rhs.to_vec();
    let mut z: Vec<_> = r.iter().zip(diagonal).map(|(r, d)| r / d).collect();
    let mut direction = z.clone();
    let mut rz = dot(&r, &z);
    if rz == 0.0 {
        return Some(x);
    }
    if !rz.is_finite() || rz < 0.0 {
        return None;
    }
    let target = rz * 1e-20;
    for _ in 0..rhs.len().saturating_mul(2).clamp(1, 1024) {
        let product = apply(&direction);
        let denominator = dot(&direction, &product);
        if !denominator.is_finite() || denominator <= 0.0 {
            return None;
        }
        let alpha = rz / denominator;
        for i in 0..x.len() {
            x[i] += alpha * direction[i];
            r[i] -= alpha * product[i];
            z[i] = r[i] / diagonal[i];
        }
        let next = dot(&r, &z);
        if !next.is_finite() || next < 0.0 {
            return None;
        }
        if next <= target {
            break;
        }
        let beta = next / rz;
        for i in 0..direction.len() {
            direction[i] = z[i] + beta * direction[i];
        }
        rz = next;
    }
    x.iter().all(|x| x.is_finite()).then_some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components_are_deterministic_and_keep_singletons() {
        assert_eq!(
            components(5, [(3, 1), (2, 0)]),
            vec![vec![0, 2], vec![1, 3], vec![4]]
        );
    }

    #[test]
    fn pcg_solves_coupled_system_and_rejects_bad_curvature() {
        let x = pcg(&[2.0, 2.0], &[1.0, 0.0], |x| {
            vec![2.0 * x[0] - x[1], 2.0 * x[1] - x[0]]
        })
        .unwrap();
        assert!((x[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!((x[1] - 1.0 / 3.0).abs() < 1e-12);
        assert!(pcg(&[0.0], &[1.0], |x| x.to_vec()).is_none());
    }
}
