//! Twofold physical prices. Low words must reach quoting AND final verification.
use crate::{Route, RouteEvaluation, market::PreciseMarket, utility::PreciseUtility};
use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Neg, Sub},
};

/// A normalized hi+lo price, not a rounded f64 price with decorative metadata.
/// Construct with from_parts; transport each word by its bits if exact replay matters.
#[derive(Clone, Copy, Debug, Default)]
pub struct Price {
    hi: f64,
    lo: f64,
}
impl Price {
    pub fn from_parts(hi: f64, lo: f64) -> Self {
        let s = hi + lo;
        let b = s - hi;
        Self {
            hi: s,
            lo: (hi - (s - b)) + (lo - b),
        }
    }
    pub fn hi(self) -> f64 {
        self.hi
    }
    pub fn lo(self) -> f64 {
        self.lo
    }
    pub fn rounded(self) -> f64 {
        self.hi + self.lo
    }
    pub fn is_finite(self) -> bool {
        self.hi.is_finite() && self.lo.is_finite()
    }
    pub(crate) fn sqrt(self) -> Self {
        if self.hi == 0.0 && self.lo == 0.0 {
            return self;
        }
        let x = Self::from(self.hi.sqrt());
        x + (self - x * x) / Self::from(2.0 * x.hi)
    }
    pub(crate) fn abs(self) -> Self {
        if self < Self::from(0.0) { -self } else { self }
    }
}
impl From<f64> for Price {
    fn from(hi: f64) -> Self {
        Self { hi, lo: 0.0 }
    }
}
impl PartialEq for Price {
    fn eq(&self, other: &Self) -> bool {
        self.hi == other.hi && self.lo == other.lo
    }
}
impl PartialOrd for Price {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.hi.partial_cmp(&other.hi)? {
            Ordering::Equal => self.lo.partial_cmp(&other.lo),
            v => Some(v),
        }
    }
}
impl Add for Price {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        let s = self.hi + b.hi;
        let z = s - self.hi;
        Self::from_parts(s, ((self.hi - (s - z)) + (b.hi - z)) + (self.lo + b.lo))
    }
}
impl Neg for Price {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}
impl Sub for Price {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        self + (-b)
    }
}
impl Mul for Price {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        let p = self.hi * b.hi;
        Self::from_parts(
            p,
            self.hi.mul_add(b.hi, -p) + (self.hi * b.lo + self.lo * b.hi) + self.lo * b.lo,
        )
    }
}
impl Div for Price {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let q = self.hi / b.hi;
        Self::from_parts(q, ((self - b * Self::from(q)).rounded()) / b.hi)
    }
}

/// Cancellation-free regional square-root support remainder, using full price differences.
pub(crate) fn support_remainder(v: [Price; 2], r: [Price; 2]) -> f64 {
    let roots = v.map(|p| p.hi().sqrt());
    let old = r.map(|p| p.hi().sqrt());
    let delta = [
        (v[0] - r[0]).rounded() / (roots[0] + old[0]),
        (v[1] - r[1]).rounded() / (roots[1] + old[1]),
    ];
    let ratio = (old[1] / old[0]).sqrt();
    (ratio * delta[0] - delta[1] / ratio).powi(2)
}
impl<U: PreciseUtility, M: PreciseMarket> Route<U, M> {
    /// Recover executable trades using both price words, without a lossy f64 conversion.
    pub fn arbitrage_precise(&self, p: &[Price]) -> (ndarray::Array2<f64>, ndarray::Array2<f64>) {
        assert_eq!(p.len(), self.tokens);
        let mut input = ndarray::Array2::zeros((self.markets.len(), self.tokens));
        let mut output = input.clone();
        for (i, (m, (a, b))) in self.markets.iter().enumerate() {
            let t = m.evaluate_precise([p[*a], p[*b]]);
            for (j, k) in [*a, *b].into_iter().enumerate() {
                input[[i, k]] += t.input[j];
                output[[i, k]] += t.output[j];
            }
        }
        (input, output)
    }

    /// Evaluate every original market using both price words; never validates rounded hi alone.
    pub fn evaluate_precise(&self, p: &[Price]) -> RouteEvaluation {
        assert_eq!(p.len(), self.tokens);
        let mut cost = crate::numerics::Sum::default();
        cost.add(self.objective.precise_value(p));
        let mut gradient = vec![crate::numerics::Sum::default(); self.tokens];
        for (s, g) in gradient.iter_mut().zip(self.objective.precise_grad(p)) {
            s.add(g);
        }
        let mut turnover = vec![crate::numerics::Sum::default(); self.tokens];
        let mut trades = Vec::with_capacity(self.markets.len());
        for (m, (a, b)) in &self.markets {
            let t = m.evaluate_precise([p[*a], p[*b]]);
            cost.add(t.value);
            for (j, i) in [*a, *b].into_iter().enumerate() {
                gradient[i].add(t.output[j] - t.input[j]);
                turnover[i].add(t.output[j] + t.input[j]);
            }
            trades.push(t);
        }
        RouteEvaluation {
            cost: cost.total(),
            gradient: gradient
                .into_iter()
                .map(crate::numerics::Sum::total)
                .collect(),
            turnover: turnover
                .into_iter()
                .map(crate::numerics::Sum::total)
                .collect(),
            market_evaluations: trades,
        }
    }
    pub(crate) fn precise_value_change(
        &self,
        p: &[Price],
        r: &[Price],
        e: &RouteEvaluation,
        re: &RouteEvaluation,
    ) -> f64 {
        let mut sum = crate::numerics::Sum::default();
        sum.add(self.objective.precise_value_change(p, r));
        for (i, (m, (a, b))) in self.markets.iter().enumerate() {
            if p[*a] != r[*a] || p[*b] != r[*b] {
                sum.add(m.value_change_precise(
                    [p[*a], p[*b]],
                    [r[*a], r[*b]],
                    e.market_evaluations[i],
                    re.market_evaluations[i],
                ));
            }
        }
        sum.total()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retains_sub_ulp_changes_and_boundary_order() {
        let one = Price::from(1.0);
        let p = one + Price::from(1e-20);
        assert_eq!(p.hi(), 1.0);
        assert_eq!((p - one).rounded(), 1e-20);
        assert!(p > one);
        assert!(one - Price::from(1e-20) < one);
        let x = Price::from(1.1);
        let reciprocal = one / (x * x);
        assert!(((reciprocal * x * x) - one).rounded().abs() < 1e-30);
        assert!((p.sqrt() * p.sqrt() - p).rounded().abs() < 1e-30);
    }
}
