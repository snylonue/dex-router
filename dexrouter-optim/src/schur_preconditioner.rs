//! Exact positive star-mesh elimination, followed by a bounded dense core.
//! Only an inverse PCG metric: floating gauges never change the objective.
use crate::numerics::Sum;
use std::{cell::RefCell, collections::VecDeque};

fn conductance(adj: &[(usize, f64)], node: usize) -> f64 {
    adj.binary_search_by_key(&node, |&(i, _)| i)
        .map_or(0.0, |i| adj[i].1)
}
fn set_edge(adj: &mut Vec<(usize, f64)>, node: usize, weight: f64) {
    match adj.binary_search_by_key(&node, |&(i, _)| i) {
        Ok(i) => adj[i].1 = weight,
        Err(i) => adj.insert(i, (node, weight)),
    }
}

struct Elimination {
    node: usize,
    pivot: f64,
    neighbors: [(usize, f64); 2],
}

pub(crate) struct SchurPreconditioner {
    normalizer: Vec<f64>,
    eliminated: Vec<Elimination>,
    core: Vec<usize>,
    core_scale: Vec<f64>,
    cholesky: Vec<Vec<f64>>,
    workspace: RefCell<(Vec<Sum>, Vec<f64>)>,
}
impl SchurPreconditioner {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        edges: &[(usize, usize, f64)],
        utility: &[f64],
        factors: &[f64],
        objective_scale: f64,
        metric: &[f64],
        free: &[bool],
        damping: f64,
    ) -> Option<Self> {
        let n = metric.len();
        if utility.len() != n
            || factors.len() != n
            || free.len() != n
            || !objective_scale.is_finite()
            || objective_scale <= 0.0
            || !damping.is_finite()
            || damping < 0.0
        {
            return None;
        }
        let normalizer: Vec<_> = metric.iter().map(|d| d.sqrt()).collect();
        let q: Vec<_> = factors
            .iter()
            .zip(&normalizer)
            .map(|(f, d)| f / d / objective_scale.sqrt())
            .collect();
        if metric.iter().any(|d| !d.is_finite() || *d <= 0.0)
            || q.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || utility.iter().any(|u| !u.is_finite() || *u < 0.0)
        {
            return None;
        }
        let scaled = |w: f64, i: usize| (w * q[i]) * q[i];
        let mut ground: Vec<_> = utility
            .iter()
            .enumerate()
            .map(|(i, &u)| scaled(u, i) + damping)
            .collect();
        let mut pairs = Vec::with_capacity(edges.len());
        for &(a, b, w) in edges {
            if !w.is_finite()
                || w < 0.0
                || (a >= n && a != usize::MAX)
                || (b >= n && b != usize::MAX)
            {
                return None;
            }
            if a == b || w == 0.0 {
                continue;
            }
            match (a < n && free[a], b < n && free[b]) {
                (true, true) => pairs.push((a.min(b), a.max(b), w)),
                (true, false) => ground[a] += scaled(w, a),
                (false, true) => ground[b] += scaled(w, b),
                _ => {}
            }
        }
        // Stable contiguous aggregation avoids a tree allocation for every edge.
        // Lexicographic pair order also builds each adjacency list in node order.
        pairs.sort_by_key(|&(a, b, _)| (a, b));
        let mut adjacent = vec![Vec::<(usize, f64)>::new(); n];
        let mut at = 0;
        while at < pairs.len() {
            let (a, b, _) = pairs[at];
            let mut sum = Sum::default();
            while at < pairs.len() && (pairs[at].0, pairs[at].1) == (a, b) {
                sum.add(pairs[at].2);
                at += 1;
            }
            let w = sum.total();
            if !w.is_finite() {
                return None;
            }
            adjacent[a].push((b, w));
            adjacent[b].push((a, w));
        }
        // A singular component needs an SPD metric, not a new economic bound.
        let mut seen = vec![false; n];
        let mut stack = Vec::with_capacity(n);
        for start in 0..n {
            if !free[start] || seen[start] {
                continue;
            }
            stack.push(start);
            seen[start] = true;
            let mut anchored = false;
            while let Some(i) = stack.pop() {
                anchored |= ground[i] > 0.0;
                for &(j, _) in &adjacent[i] {
                    if !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
            if !anchored {
                ground[start] = 1.0;
            }
        }
        if ground.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return None;
        }
        let mut live = free.to_vec();
        let mut queue = VecDeque::new();
        for i in 0..n {
            if live[i] && adjacent[i].len() <= 2 {
                queue.push_back(i);
            }
        }
        let mut eliminated = Vec::with_capacity(n);
        while let Some(i) = queue.pop_front() {
            if !live[i] || adjacent[i].len() > 2 {
                continue;
            }
            let degree = adjacent[i].len();
            let mut neighbors = [(usize::MAX, 0.0); 2];
            for (k, &(j, w)) in adjacent[i].iter().enumerate() {
                neighbors[k] = (j, w);
            }
            let mut pivot = Sum::default();
            pivot.add(ground[i]);
            for &(_, w) in &neighbors[..degree] {
                pivot.add(scaled(w, i));
            }
            let pivot = pivot.total();
            if !pivot.is_finite() || pivot <= 0.0 {
                return None;
            }
            let transport = neighbors.map(|(j, w)| {
                if j < n {
                    (j, (q[j] / q[i]) * (scaled(w, i) / pivot))
                } else {
                    (usize::MAX, 0.0)
                }
            });
            if transport.iter().any(|(_, t)| !t.is_finite()) {
                return None;
            }
            for &(j, w) in &neighbors[..degree] {
                let index = adjacent[j]
                    .binary_search_by_key(&i, |&(node, _)| node)
                    .ok()?;
                adjacent[j].remove(index);
                // Positive shunt update, not subtraction of two huge diagonals.
                ground[j] += scaled(w, j) * (ground[i] / pivot);
            }
            if degree == 2 {
                let [(a, wa), (b, wb)] = neighbors;
                let fill = wa * (scaled(wb, i) / pivot);
                let mut total = Sum::default();
                total.add(conductance(&adjacent[a], b));
                total.add(fill);
                let total = total.total();
                if !total.is_finite() || total <= 0.0 {
                    return None;
                }
                set_edge(&mut adjacent[a], b, total);
                set_edge(&mut adjacent[b], a, total);
            }
            live[i] = false;
            adjacent[i].clear();
            eliminated.push(Elimination {
                node: i,
                pivot,
                neighbors: transport,
            });
            for &(j, _) in &neighbors[..degree] {
                if adjacent[j].len() <= 2 {
                    queue.push_back(j);
                }
            }
        }
        let core: Vec<_> = (0..n).filter(|&i| live[i]).collect();
        if core.len() > 64 {
            return None;
        }
        let core_scale: Vec<_> = core
            .iter()
            .map(|&i| {
                let mut s = Sum::default();
                s.add(ground[i]);
                for &(_, w) in &adjacent[i] {
                    s.add(scaled(w, i));
                }
                s.total().sqrt()
            })
            .collect();
        if core_scale.iter().any(|d| !d.is_finite() || *d <= 0.0) {
            return None;
        }
        let m = core.len();
        let mut cholesky = vec![vec![0.0; m]; m];
        for a in 0..m {
            for b in 0..=a {
                let mut value = if a == b {
                    1.0
                } else {
                    -conductance(&adjacent[core[a]], core[b]) * q[core[a]] / core_scale[a]
                        * q[core[b]]
                        / core_scale[b]
                };
                for (&x, &y) in cholesky[a][..b].iter().zip(&cholesky[b][..b]) {
                    value -= x * y;
                }
                if a == b {
                    if !value.is_finite() || value <= 0.0 {
                        return None;
                    }
                    cholesky[a][b] = value.sqrt();
                } else {
                    cholesky[a][b] = value / cholesky[b][b];
                }
            }
        }
        Some(Self {
            normalizer,
            eliminated,
            core,
            core_scale,
            cholesky,
            workspace: RefCell::new((vec![Sum::default(); n], vec![0.0; n])),
        })
    }
}
impl dexrouter_newton::Preconditioner for SchurPreconditioner {
    fn apply(&self, residual: &[f64], solution: &mut [f64]) -> anyhow::Result<()> {
        anyhow::ensure!(
            residual.len() == self.normalizer.len() && solution.len() == self.normalizer.len(),
            "Schur metric dimensions"
        );
        let mut space = self.workspace.borrow_mut();
        let (rhs, y) = &mut *space;
        rhs.fill(Sum::default());
        y.fill(0.0);
        solution.fill(0.0);
        for i in 0..rhs.len() {
            rhs[i].add(residual[i] / self.normalizer[i]);
        }
        for step in &self.eliminated {
            let b = rhs[step.node].total();
            for &(j, t) in &step.neighbors {
                if j < rhs.len() {
                    rhs[j].add(t * b);
                }
            }
        }
        for a in 0..self.core.len() {
            let i = self.core[a];
            let mut b = rhs[i].total() / self.core_scale[a];
            for k in 0..a {
                b -= self.cholesky[a][k] * y[self.core[k]];
            }
            y[i] = b / self.cholesky[a][a];
        }
        for a in (0..self.core.len()).rev() {
            let i = self.core[a];
            let mut b = y[i];
            for k in a + 1..self.core.len() {
                b -= self.cholesky[k][a] * y[self.core[k]];
            }
            y[i] = b / self.cholesky[a][a];
        }
        for (a, &i) in self.core.iter().enumerate() {
            y[i] /= self.core_scale[a];
        }
        for step in self.eliminated.iter().rev() {
            let mut b = Sum::default();
            b.add(rhs[step.node].total() / step.pivot);
            for &(j, t) in &step.neighbors {
                if j < y.len() {
                    b.add(t * y[j]);
                }
            }
            y[step.node] = b.total();
        }
        for (i, out) in solution.iter_mut().enumerate() {
            *out = y[i] / self.normalizer[i];
        }
        anyhow::ensure!(
            solution.iter().all(|v| v.is_finite()),
            "nonfinite Schur application"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dexrouter_newton::Preconditioner;
    #[test]
    fn exact_inverse_handles_cycles_masks_scaling_and_damping() {
        for n in [4, 8, 16] {
            for mask in [false, true] {
                for damping in [0.0, 0.03] {
                    let mut edges = vec![(0, usize::MAX, 0.07)];
                    for i in 0..n {
                        for j in i + 1..n {
                            if (i + j) % 3 != 0 {
                                edges.push((i, j, 0.2 + ((i * 7 + j) % 11) as f64));
                            }
                        }
                    }
                    edges.push((0, 1, 2.0));
                    let utility = vec![0.03; n];
                    let factors: Vec<_> = (0..n).map(|i| 10.0_f64.powi(i as i32 % 5 - 2)).collect();
                    let free: Vec<_> = (0..n).map(|i| !mask || i % 4 != 1).collect();
                    let mut metric = utility.clone();
                    for &(a, b, w) in &edges {
                        metric[a] += w;
                        if b < n {
                            metric[b] += w;
                        }
                    }
                    for i in 0..n {
                        metric[i] *= factors[i] * factors[i] / 7.0;
                    }
                    let p = SchurPreconditioner::new(
                        &edges, &utility, &factors, 7.0, &metric, &free, damping,
                    )
                    .unwrap();
                    let rhs: Vec<_> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
                    let mut x = vec![0.0; n];
                    p.apply(&rhs, &mut x).unwrap();
                    let mut product = vec![0.0; n];
                    for i in 0..n {
                        product[i] = utility[i] * factors[i] * x[i];
                    }
                    for &(a, b, w) in &edges {
                        let delta =
                            w * (factors[a] * x[a] - if b < n { factors[b] * x[b] } else { 0.0 });
                        product[a] += delta;
                        if b < n {
                            product[b] -= delta;
                        }
                    }
                    for i in 0..n {
                        if free[i] {
                            let hx = product[i] * factors[i] / 7.0 + damping * metric[i] * x[i];
                            assert!(
                                (hx - rhs[i]).abs() < 2e-10,
                                "n={n},i={i}: {hx} vs {}",
                                rhs[i]
                            );
                        } else {
                            assert_eq!(x[i], 0.0);
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn positive_shunt_preserves_weak_grounding() {
        let p = SchurPreconditioner::new(
            &[(0, 1, 1e12)],
            &[1e-12, 0.0],
            &[1.0; 2],
            1.0,
            &[1e12; 2],
            &[true; 2],
            0.0,
        )
        .unwrap();
        let mut x = [0.0; 2];
        p.apply(&[1e-12, 0.0], &mut x).unwrap();
        for v in x {
            assert!((v - 1.0).abs() < 1e-14);
        }
    }
    #[test]
    fn floating_gauge_is_finite_and_only_in_metric() {
        let p = SchurPreconditioner::new(
            &[(0, 1, 2.0)],
            &[0.0; 3],
            &[1.0; 3],
            1.0,
            &[2.0, 2.0, 1.0],
            &[true; 3],
            0.0,
        )
        .unwrap();
        let mut x = [0.0; 3];
        p.apply(&[1.0, -1.0, 2.0], &mut x).unwrap();
        assert!(x.iter().all(|x| x.is_finite()));
        assert!((2.0 * (x[0] - x[1]) - 1.0).abs() < 1e-12);
        assert_eq!(x[2], 2.0);
    }
}
