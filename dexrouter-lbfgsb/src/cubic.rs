//! Safeguarded cubic trial proposals for a clipped zoom interpolation.
// Adapted from MINPACK-2 dcsrch/dcstep via L-BFGS-B-C, BSD-3-Clause.
// Copyright (c) 2015, Stephen Becker. See ../MORE_THUENTE_LICENSE.txt for the retained terms.
use super::ZoomBracket;
use crate::cast;
use num_traits::{Float, FromPrimitive};
#[derive(Clone, Copy)]
struct Point<T> {
    a: T,
    f: T,
    g: T,
}
fn gamma<T: Float + FromPrimitive>(theta: T, dx: T, dp: T) -> T {
    let scale = theta.abs().max(dx.abs()).max(dp.abs());
    scale
        * ((theta / scale) * (theta / scale) - (dx / scale) * (dp / scale))
            .max(T::zero())
            .sqrt()
}
fn update<T: Float + FromPrimitive>(
    x: &mut Point<T>,
    y: &mut Point<T>,
    p: Point<T>,
    bracket: &mut bool,
    min: T,
    max: T,
) -> T {
    let delta = p.a - x.a;
    let opposite = (p.g < T::zero()) != (x.g < T::zero());
    let mut next;
    if p.f > x.f {
        let theta = cast::<T>(3.) * ((x.f - p.f) / delta) + x.g + p.g;
        let mut gm = gamma(theta, x.g, p.g);
        if p.a < x.a {
            gm = -gm;
        }
        let c = x.a + ((gm - x.g + theta) / (gm - x.g + gm + p.g)) * delta;
        let q = x.a + (x.g / ((x.f - p.f) / delta + x.g)) / cast(2.) * delta;
        next = if !c.is_finite() || c == x.a {
            q
        } else if (c - x.a).abs() < (q - x.a).abs() {
            c
        } else {
            c + (q - c) / cast(2.)
        };
        *bracket = true;
    } else if opposite {
        let theta = cast::<T>(3.) * ((x.f - p.f) / delta) + x.g + p.g;
        let mut gm = gamma(theta, x.g, p.g);
        if p.a > x.a {
            gm = -gm;
        }
        let c = p.a + ((gm - p.g + theta) / (gm - p.g + gm + x.g)) * (-delta);
        let q = p.a + (p.g / (p.g - x.g)) * (-delta);
        next = if (c - p.a).abs() > (q - p.a).abs() {
            c
        } else {
            q
        };
        *bracket = true;
    } else if p.g.abs() < x.g.abs() {
        let theta = cast::<T>(3.) * ((x.f - p.f) / delta) + x.g + p.g;
        let mut gm = gamma(theta, x.g, p.g);
        if p.a > x.a {
            gm = -gm;
        }
        let r = (gm - p.g + theta) / (gm + (x.g - p.g) + gm);
        let c = if r < T::zero() && gm != T::zero() {
            p.a + r * (-delta)
        } else if p.a > x.a {
            max
        } else {
            min
        };
        let q = p.a + (p.g / (p.g - x.g)) * (-delta);
        if *bracket {
            next = if (c - p.a).abs() < (q - p.a).abs() {
                c
            } else {
                q
            };
            let guard = p.a + cast::<T>(0.66) * (y.a - p.a);
            next = if p.a > x.a {
                next.min(guard)
            } else {
                next.max(guard)
            };
        } else {
            next = if (c - p.a).abs() > (q - p.a).abs() {
                c
            } else {
                q
            };
            next = next.max(min).min(max);
        }
    } else if *bracket {
        let theta = cast::<T>(3.) * ((p.f - y.f) / (y.a - p.a)) + y.g + p.g;
        let mut gm = gamma(theta, y.g, p.g);
        if p.a > y.a {
            gm = -gm;
        }
        next = p.a + ((gm - p.g + theta) / (gm - p.g + gm + y.g)) * (y.a - p.a);
    } else {
        next = if p.a > x.a { max } else { min };
    }
    if p.f > x.f {
        *y = p;
    } else {
        if opposite {
            *y = *x;
        }
        *x = p;
    }
    if !next.is_finite() {
        next = if *bracket {
            x.a / cast(2.) + y.a / cast(2.)
        } else {
            max
        };
    }
    next
}
pub(super) fn zoom<T: Float + FromPrimitive>(b: &ZoomBracket<T>) -> Option<T> {
    let mut x = Point {
        a: b.low_step,
        f: b.low_phi,
        g: b.low_derivative,
    };
    let mut y = x;
    let mut bracket = false;
    let p = Point {
        a: b.high_step,
        f: b.high_phi,
        g: b.high_derivative,
    };
    let a = update(
        &mut x,
        &mut y,
        p,
        &mut bracket,
        b.low_step.min(b.high_step),
        b.low_step.max(b.high_step),
    );
    (a.is_finite() && a > b.low_step.min(b.high_step) && a < b.low_step.max(b.high_step))
        .then_some(a)
}
