//! Stable support-forest inverse metric; it never changes the objective or bounds.
use crate::numerics::Sum;
use std::{cell::RefCell, collections::BTreeMap};

pub(crate) struct TreePreconditioner {
    parent: Vec<usize>,
    order: Vec<usize>,
    diagonal: Vec<f64>,
    transport: Vec<f64>,
    normalizer: Vec<f64>,
    workspace: RefCell<(Vec<Sum>, Vec<f64>)>,
}

fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

impl TreePreconditioner {
    /// Endpoints absent from this block are usize::MAX. Active-bound endpoints
    /// are also grounded for this solve, not fixed in the original problem.
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
        if factors.len() != n
            || utility.len() != n
            || free.len() != n
            || !objective_scale.is_finite()
            || objective_scale <= 0.0
            || !damping.is_finite()
            || damping < 0.0
            || metric.iter().any(|d| !d.is_finite() || *d <= 0.0)
            || factors.iter().any(|f| !f.is_finite() || *f <= 0.0)
            || utility.iter().any(|u| !u.is_finite() || *u < 0.0)
        {
            return None;
        }
        let normalizer: Vec<_> = metric.iter().map(|d| d.sqrt()).collect();
        let q: Vec<_> = factors
            .iter()
            .zip(&normalizer)
            .map(|(f, d)| f / d / objective_scale.sqrt())
            .collect();
        if q.iter().any(|q| !q.is_finite() || *q <= 0.0) {
            return None;
        }
        let scaled = |w: f64, i: usize| (w * q[i]) * q[i];
        let mut ground: Vec<_> = utility
            .iter()
            .enumerate()
            .map(|(i, &u)| scaled(u, i) + damping)
            .collect();
        let mut pairs = BTreeMap::<(usize, usize), Sum>::new();
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
                (true, true) => pairs.entry((a.min(b), a.max(b))).or_default().add(w),
                (true, false) => ground[a] += scaled(w, a),
                (false, true) => ground[b] += scaled(w, b),
                _ => {}
            }
        }
        let mut edges: Vec<_> = pairs
            .into_iter()
            .map(|((a, b), w)| (a, b, w.total()))
            .collect();
        if edges.iter().any(|e| !e.2.is_finite()) || ground.iter().any(|d| !d.is_finite()) {
            return None;
        }
        edges.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        let mut union: Vec<_> = (0..n).collect();
        let mut adjacent = vec![Vec::new(); n];
        for (a, b, w) in edges {
            let ra = root(&mut union, a);
            let rb = root(&mut union, b);
            if ra != rb {
                union[ra] = rb;
                adjacent[a].push((b, w));
                adjacent[b].push((a, w));
            }
        }
        let mut groups = vec![Vec::new(); n];
        for i in 0..n {
            if free[i] {
                groups[root(&mut union, i)].push(i);
            }
        }
        let mut parent = vec![usize::MAX; n];
        let mut weight = vec![0.0; n];
        let mut order = Vec::with_capacity(n);
        for group in groups.iter().filter(|g| !g.is_empty()) {
            let mut anchor = group[0];
            for &i in group {
                if ground[i] > ground[anchor] {
                    anchor = i;
                }
            }
            parent[anchor] = anchor;
            let mut stack = vec![anchor];
            while let Some(i) = stack.pop() {
                order.push(i);
                for &(j, w) in &adjacent[i] {
                    if parent[j] == usize::MAX {
                        parent[j] = i;
                        weight[j] = w;
                        stack.push(j);
                    }
                }
            }
        }
        let mut diagonal = vec![0.0; n];
        let mut transport = vec![0.0; n];
        for &i in order.iter().rev() {
            if parent[i] == i {
                // A floating component needs an SPD metric, not a new price
                // constraint. A unit root spring is added ONLY to this metric.
                diagonal[i] = if ground[i] > 0.0 { ground[i] } else { 1.0 };
            } else {
                let p = parent[i];
                let a = scaled(weight[i], i);
                let ap = scaled(weight[i], p);
                diagonal[i] = a + ground[i];
                if diagonal[i] <= 0.0 {
                    return None;
                }
                transport[i] = (q[p] / q[i]) * (a / diagonal[i]);
                // Positive shunt propagation, never parent_diag - off^2/diag.
                // The latter erases weak grounding next to huge pool liquidity.
                ground[p] += ap * (ground[i] / diagonal[i]);
            }
        }
        if order
            .iter()
            .any(|&i| !diagonal[i].is_finite() || diagonal[i] <= 0.0 || !transport[i].is_finite())
        {
            return None;
        }
        Some(Self {
            parent,
            order,
            diagonal,
            transport,
            normalizer,
            workspace: RefCell::new((vec![Sum::default(); n], vec![0.0; n])),
        })
    }
}
impl dexrouter_newton::Preconditioner for TreePreconditioner {
    fn apply(&self, residual: &[f64], solution: &mut [f64]) -> anyhow::Result<()> {
        anyhow::ensure!(
            residual.len() == self.parent.len() && solution.len() == self.parent.len(),
            "tree metric dimensions"
        );
        let mut workspace = self.workspace.borrow_mut();
        let (rhs, y) = &mut *workspace;
        rhs.fill(Sum::default());
        y.fill(0.0);
        solution.fill(0.0);
        for &i in &self.order {
            rhs[i].add(residual[i] / self.normalizer[i]);
        }
        for &i in self.order.iter().rev() {
            let p = self.parent[i];
            if p != i {
                let contribution = self.transport[i] * rhs[i].total();
                rhs[p].add(contribution);
            }
        }
        for &i in &self.order {
            let p = self.parent[i];
            let local = rhs[i].total() / self.diagonal[i];
            y[i] = if p == i {
                local
            } else {
                self.transport[i].mul_add(y[p], local)
            };
            solution[i] = y[i] / self.normalizer[i];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dexrouter_newton::Preconditioner;
    #[test]
    fn weak_grounding_survives_twenty_four_orders_of_liquidity() {
        let w = 1e12;
        let weak = 1e-12;
        let p = TreePreconditioner::new(
            &[(0, 1, w)],
            &[weak, 0.0],
            &[1.0, 1.0],
            1.0,
            &[w, w],
            &[true, true],
            0.0,
        )
        .unwrap();
        assert_eq!(
            (w + weak) - w,
            0.0,
            "naive elimination loses the weak spring"
        );
        assert!((p.diagonal[0] / (weak / w) - 1.0).abs() < 1e-14);
        let mut z = [0.0; 2];
        p.apply(&[weak, 0.0], &mut z).unwrap();
        for x in z {
            assert!((x - 1.0).abs() < 1e-14, "{z:?}");
        }
        p.apply(&[1.0, -1.0], &mut z).unwrap();
        assert_eq!(z[0], 0.0);
        assert!((z[1] + 1.0 / w).abs() < 1e-26);
    }
    #[test]
    fn masked_damped_scaled_forest_solves_its_original_step_model() {
        let edges = [
            (0, 1, 300.0),
            (1, 2, 1e-4),
            (2, 3, 2.0),
            (0, usize::MAX, 1e-3),
        ];
        let utility = [1e-5, 0.0, 0.0, 0.0];
        let factors = [0.01, 10.0, 0.2, 1000.0];
        let scale = 7.0;
        let damping = 0.03;
        let mut diag = utility;
        for &(a, b, w) in &edges {
            if a < 4 {
                diag[a] += w;
            }
            if b < 4 {
                diag[b] += w;
            }
        }
        for i in 0..4 {
            diag[i] = diag[i] * factors[i] * factors[i] / scale;
        }
        for free in [[true; 4], [true, false, true, true]] {
            let p =
                TreePreconditioner::new(&edges, &utility, &factors, scale, &diag, &free, damping)
                    .unwrap();
            let rhs = [0.2, -0.7, 1.1, 0.03];
            let mut x = [0.0; 4];
            p.apply(&rhs, &mut x).unwrap();
            let relative: Vec<_> = x.iter().zip(factors).map(|(x, f)| x * f).collect();
            let mut product = [0.0; 4];
            for i in 0..4 {
                product[i] = utility[i] * relative[i];
            }
            for &(a, b, w) in &edges {
                let delta = w
                    * (if a < 4 { relative[a] } else { 0.0 }
                        - if b < 4 { relative[b] } else { 0.0 });
                if a < 4 {
                    product[a] += delta;
                }
                if b < 4 {
                    product[b] -= delta;
                }
            }
            for i in 0..4 {
                if free[i] {
                    let actual = product[i] * factors[i] / scale + damping * diag[i] * x[i];
                    assert!(
                        (actual - rhs[i]).abs() < 1e-9,
                        "i={i} {actual} vs {}",
                        rhs[i]
                    );
                } else {
                    assert_eq!(x[i], 0.0);
                }
            }
        }
    }
    #[test]
    fn parallel_edges_form_one_support_edge_and_cycle_metric_is_spd() {
        let edges = [
            (0, 1, 4.0),
            (0, 1, 6.0),
            (1, 2, 8.0),
            (2, 3, 7.0),
            (3, 0, 1.0),
            (0, 2, 0.5),
            (0, usize::MAX, 0.1),
        ];
        let factors = [0.1, 10.0, 0.2, 50.0];
        let mut metric = [0.0; 4];
        for &(a, b, w) in &edges {
            if a < 4 {
                metric[a] += w;
            }
            if b < 4 {
                metric[b] += w;
            }
        }
        for i in 0..4 {
            metric[i] = metric[i] * factors[i] * factors[i];
        }
        let p = TreePreconditioner::new(&edges, &[0.0; 4], &factors, 1.0, &metric, &[true; 4], 0.0)
            .unwrap();
        assert_eq!(p.parent, vec![0, 0, 1, 2]);
        let r = [0.2, -0.3, 0.7, 1.2];
        let s = [-0.5, 0.2, 0.6, -0.8];
        let mut pr = [0.0; 4];
        let mut ps = [0.0; 4];
        p.apply(&r, &mut pr).unwrap();
        p.apply(&s, &mut ps).unwrap();
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
        assert!(dot(&r, &pr) > 0.0 && dot(&s, &ps) > 0.0);
        assert!((dot(&r, &ps) - dot(&s, &pr)).abs() < 1e-10 * dot(&r, &ps).abs().max(1.0));
        let reversed: Vec<_> = edges.into_iter().rev().collect();
        let other = TreePreconditioner::new(
            &reversed, &[0.0; 4], &factors, 1.0, &metric, &[true; 4], 0.0,
        )
        .unwrap();
        assert_eq!(other.parent, p.parent);
    }

    #[test]
    fn floating_and_isolated_components_get_only_metric_gauges() {
        let p = TreePreconditioner::new(
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
        assert!(x.iter().all(|v| v.is_finite()));
        assert!(x[0] - x[1] > 0.0);
        assert_eq!(x[2], 2.0);
        assert!(
            TreePreconditioner::new(
                &[(0, 1, f64::NAN)],
                &[0.0; 2],
                &[1.0; 2],
                1.0,
                &[1.0; 2],
                &[true; 2],
                0.0
            )
            .is_none()
        );
    }
}
