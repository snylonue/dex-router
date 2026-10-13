use num_traits::Float;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bound<T> {
    pub lower: Option<T>,
    pub upper: Option<T>,
}

impl<T: Float> Bound<T> {
    pub fn new(lower: Option<T>, upper: Option<T>) -> Self {
        Self { lower, upper }
    }

    pub fn unbounded() -> Self {
        Self::new(None, None)
    }

    pub fn lower(lower: T) -> Self {
        Self::new(Some(lower), None)
    }

    pub fn upper(upper: T) -> Self {
        Self::new(None, Some(upper))
    }

    pub fn both(lower: T, upper: T) -> Self {
        Self::new(Some(lower), Some(upper))
    }

    /// True only for a supplied finite singleton interval, never an active bound.
    pub fn is_fixed(&self) -> bool {
        matches!((self.lower, self.upper), (Some(l), Some(u)) if l.is_finite() && l == u)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if let Some(lower) = self.lower {
            anyhow::ensure!(lower.is_finite(), "lower bound must be finite");
        }
        if let Some(upper) = self.upper {
            anyhow::ensure!(upper.is_finite(), "upper bound must be finite");
        }
        if let (Some(lower), Some(upper)) = (self.lower, self.upper) {
            anyhow::ensure!(lower <= upper, "lower bound must not exceed upper bound");
        }
        Ok(())
    }

    pub fn project(&self, value: T) -> T {
        let mut out = value;
        if let Some(lower) = self.lower {
            out = out.max(lower);
        }
        if let Some(upper) = self.upper {
            out = out.min(upper);
        }
        out
    }

    pub(crate) fn is_at_lower(&self, value: T) -> bool {
        self.lower.is_some_and(|lower| value <= lower)
    }

    pub(crate) fn is_at_upper(&self, value: T) -> bool {
        self.upper.is_some_and(|upper| value >= upper)
    }
}

pub(crate) fn project<T: Float>(x: &mut [T], bounds: &[Bound<T>]) {
    for (xi, bound) in x.iter_mut().zip(bounds.iter()) {
        *xi = bound.project(*xi);
    }
}

pub(crate) fn projected_gradient<T: Float>(x: &[T], g: &[T], bounds: &[Bound<T>]) -> Vec<T> {
    x.iter()
        .zip(g.iter())
        .zip(bounds.iter())
        .map(|((&xi, &gi), bound)| {
            // Fortran projgr: clip the gradient by the distance to the bound
            // in its descent direction (equivalent to x - P(x - g), without
            // subtracting nearly equal unbounded coordinates).
            if gi < T::zero() {
                bound.upper.map_or(gi, |upper| gi.max(xi - upper))
            } else {
                bound.lower.map_or(gi, |lower| gi.min(xi - lower))
            }
        })
        .collect()
}

pub(crate) fn projected_steepest_direction<T: Float>(
    x: &[T],
    g: &[T],
    bounds: &[Bound<T>],
) -> Vec<T> {
    x.iter()
        .zip(g.iter())
        .zip(bounds.iter())
        .map(|((&xi, &gi), bound)| {
            let direction = -gi;
            if (direction < T::zero() && bound.is_at_lower(xi))
                || (direction > T::zero() && bound.is_at_upper(xi))
            {
                T::zero()
            } else {
                direction
            }
        })
        .collect()
}

pub(crate) fn feasible_step_max<T: Float>(x: &[T], direction: &[T], bounds: &[Bound<T>]) -> T {
    let mut step = T::infinity();
    for ((&xi, &di), bound) in x.iter().zip(direction.iter()).zip(bounds.iter()) {
        if di > T::zero() {
            if let Some(upper) = bound.upper {
                step = step.min((upper - xi) / di);
            }
        } else if di < T::zero() {
            if let Some(lower) = bound.lower {
                step = step.min((lower - xi) / di);
            }
        }
    }
    step
}

pub(crate) fn is_free<T: Float>(x: T, bound: &Bound<T>) -> bool {
    !bound.is_at_lower(x) && !bound.is_at_upper(x)
}
