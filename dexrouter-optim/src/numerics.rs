/// Neumaier summation, also preserving the ordinary propagation of infinities.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Sum {
    value: f64,
    correction: f64,
}

impl Sum {
    pub fn add(&mut self, term: f64) {
        let next = self.value + term;
        if next.is_finite() {
            self.correction += if self.value.abs() >= term.abs() {
                (self.value - next) + term
            } else {
                (term - next) + self.value
            };
        } else {
            self.correction = 0.0;
        }
        self.value = next;
    }

    pub fn total(self) -> f64 {
        self.value + self.correction
    }
}

#[cfg(test)]
mod tests {
    use super::Sum;
    #[test]
    fn retains_small_terms_and_infinities() {
        let mut sum = Sum::default();
        for x in [1e20, 1.0, -1e20] {
            sum.add(x);
        }
        assert_eq!(sum.total(), 1.0);
        sum.add(f64::INFINITY);
        assert_eq!(sum.total(), f64::INFINITY);
    }
}
