//! Structured numerical property fuzzing, not arbitrary/nonconvex callbacks.
//! All generated problems have a planted, independently known box-constrained optimum.
use dexrouter_lbfgsb::{Bound, Options, Status, minimize};
use num_traits::{Float, FromPrimitive};
use proptest::prelude::*;
use proptest::test_runner::{Config, FileFailurePersistence, RngSeed, TestCaseError, TestRunner};
use std::cell::RefCell;

#[derive(Clone, Debug)]
struct Coordinate {
    target: i8,
    start: i8,
    bound: u8,
    margin: u8,
    normal: u8,
}

#[derive(Clone, Debug)]
struct Term {
    row: Vec<i8>,
    shift: i8,
    exponent: i8,
    hinge: bool,
}

#[derive(Clone, Debug)]
struct Case {
    coordinates: Vec<Coordinate>,
    terms: Vec<Term>,
    scale: i8,
    mu: i8,
    offset: Option<u8>,
    memory: usize,
}

fn c<T: Float + FromPrimitive>(x: f64) -> T {
    T::from_f64(x).unwrap()
}

impl Case {
    fn target(&self) -> Vec<f64> {
        self.coordinates
            .iter()
            .map(|v| f64::from(v.target) / 8.)
            .collect()
    }

    fn start<T: Float + FromPrimitive>(&self) -> Vec<T> {
        self.coordinates
            .iter()
            .map(|v| c(f64::from(v.start) / 4.))
            .collect()
    }

    fn geometry(&self) -> (Vec<Bound<f64>>, Vec<f64>) {
        self.coordinates
            .iter()
            .map(|v| {
                let t = f64::from(v.target) / 8.;
                let margin = f64::from(v.margin) / 8.;
                let normal = f64::from(v.normal) / 4.;
                match v.bound {
                    0 => (Bound::unbounded(), 0.),
                    1 => (Bound::lower(t), normal),
                    2 => (Bound::upper(t), -normal),
                    3 => (Bound::both(t - margin, t + margin), 0.),
                    4 => (Bound::both(t, t), normal),
                    5 => (Bound::lower(t - margin), 0.),
                    6 => (Bound::upper(t + margin), 0.),
                    7 => (Bound::both(t, t + margin), normal),
                    _ => (Bound::both(t - margin, t), -normal),
                }
            })
            .unzip()
    }

    fn bounds<T: Float + FromPrimitive>(&self) -> Vec<Bound<T>> {
        self.geometry()
            .0
            .iter()
            .map(|b| Bound::new(b.lower.map(c), b.upper.map(c)))
            .collect()
    }

    // Bregman divergence of h/2 * max(t+q,0)^2, evaluated without subtracting
    // large quadratic constants near the optimum. Its minimum is at q=0.
    fn evaluate<T: Float + FromPrimitive>(&self, x: &[T], g: &mut [T]) -> T {
        let target = self.target();
        let (_, normal) = self.geometry();
        let d: Vec<T> = x.iter().zip(target).map(|(&x, t)| x - c(t)).collect();
        let mu = c::<T>(2_f64.powi(i32::from(self.mu)));
        let scale = c::<T>(2_f64.powi(i32::from(self.scale)));
        let mut f = T::zero();
        for i in 0..x.len() {
            g[i] = mu * d[i] + c(normal[i]);
            f = f + c::<T>(0.5) * mu * d[i] * d[i] + c::<T>(normal[i]) * d[i];
        }
        for term in &self.terms {
            let q = term
                .row
                .iter()
                .zip(&d)
                .fold(T::zero(), |s, (&a, &d)| s + c::<T>(f64::from(a) / 4.) * d);
            let t = c::<T>(f64::from(term.shift) / 8.);
            let h = c::<T>(2_f64.powi(i32::from(term.exponent)));
            let (value, derivative) = if !term.hinge || (t >= T::zero() && t + q >= T::zero()) {
                (c::<T>(0.5) * q * q, q)
            } else if t <= T::zero() && t + q <= T::zero() {
                (T::zero(), T::zero())
            } else if t < T::zero() {
                (c::<T>(0.5) * (t + q) * (t + q), t + q)
            } else {
                (t * (-q - c::<T>(0.5) * t), -t)
            };
            f = f + h * value;
            for (gi, &a) in g.iter_mut().zip(&term.row) {
                *gi = *gi + c::<T>(f64::from(a) / 4.) * h * derivative;
            }
        }
        for gi in g {
            *gi = *gi * scale;
        }
        let offset = self
            .offset
            .map_or(T::zero(), |e| c(2_f64.powi(i32::from(e))));
        offset + scale * f
    }

    // Independent f64 derivative of the defining (uncentred) convex function.
    // Do not call the optimizer's projected-gradient routine or trust its status.
    fn oracle(&self, x: &[f64]) -> (Vec<f64>, f64, f64) {
        let target = self.target();
        let (_, normal) = self.geometry();
        let d: Vec<_> = x.iter().zip(target).map(|(x, t)| x - t).collect();
        let scale = 2_f64.powi(i32::from(self.scale));
        let mu = 2_f64.powi(i32::from(self.mu));
        let mut g: Vec<_> = d.iter().zip(normal).map(|(d, n)| mu * d + n).collect();
        let mut lipschitz = mu;
        let mut magnitude = 1. + g.iter().map(|v| v.abs()).sum::<f64>();
        for term in &self.terms {
            let a: Vec<_> = term.row.iter().map(|&v| f64::from(v) / 4.).collect();
            let q: f64 = a.iter().zip(&d).map(|(a, d)| a * d).sum();
            let t = f64::from(term.shift) / 8.;
            let h = 2_f64.powi(i32::from(term.exponent));
            let derivative = if term.hinge {
                (t + q).max(0.) - t.max(0.)
            } else {
                q
            };
            for (gi, ai) in g.iter_mut().zip(&a) {
                *gi += ai * h * derivative;
            }
            lipschitz += h * a.iter().map(|a| a * a).sum::<f64>();
            let q_magnitude: f64 = a.iter().zip(&d).map(|(a, d)| (a * d).abs()).sum();
            magnitude += h * (q_magnitude + 2. * t.abs()) * a.iter().map(|a| a.abs()).sum::<f64>();
        }
        for gi in &mut g {
            *gi *= scale;
        }
        (g, scale * lipschitz, scale * magnitude)
    }
}

fn cases(extreme: bool) -> impl Strategy<Value = Case> {
    let coordinate = (-24_i8..=24, -32_i8..=32, 0_u8..9, 1_u8..=16, 1_u8..=8).prop_map(
        |(target, start, bound, margin, normal)| Coordinate {
            target,
            start,
            bound,
            margin,
            normal,
        },
    );
    prop::collection::vec(coordinate, 1..=8).prop_flat_map(move |coordinates| {
        let n = coordinates.len();
        let term = (
            prop::collection::vec(-2_i8..=2, n),
            -8_i8..=8,
            if extreme { -12_i8..=12 } else { -1_i8..=1 },
            any::<bool>(),
        )
            .prop_map(|(row, shift, exponent, hinge)| Term {
                row,
                shift,
                exponent,
                hinge,
            });
        (
            Just(coordinates),
            prop::collection::vec(term, 0..=8),
            if extreme { -12_i8..=12 } else { -3_i8..=3 },
            if extreme { -6_i8..=6 } else { 0_i8..=0 },
            if extreme {
                prop_oneof![Just(None), Just(Some(24)), Just(Some(80))].boxed()
            } else {
                Just(None).boxed()
            },
            prop::sample::select(vec![1_usize, 3, 7, 17]),
        )
            .prop_map(|(coordinates, terms, scale, mu, offset, memory)| Case {
                coordinates,
                terms,
                scale,
                mu,
                offset,
                memory,
            })
    })
}

fn config() -> Config {
    let mut config = Config::default();
    // Fixed CI stream unless explicitly overridden; persistence replays every
    // previously found seed before generating additional random cases.
    if std::env::var_os("PROPTEST_RNG_SEED").is_none() {
        config.rng_seed = RngSeed::Fixed(0x434f_4e56_4558);
    }
    if std::env::var_os("PROPTEST_MAX_SHRINK_ITERS").is_none() {
        config.max_shrink_iters = 4096;
    }
    config.failure_persistence = Some(Box::new(FileFailurePersistence::Direct(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/convex_fuzz.proptest-regressions"
    ))));
    config
}

#[derive(Debug)]
struct Audit {
    status: Option<Status>,
    certified: bool,
    roundoff_limited: bool,
    evaluations: usize,
}

fn run<T: Float + FromPrimitive + std::fmt::Debug>(
    case: &Case,
    options: Options<T>,
    injected_error: Option<usize>,
    require_solution: bool,
) -> Result<Audit, TestCaseError> {
    let bounds = case.bounds::<T>();
    let mut calls = 0;
    let mut error_called = false;
    let mut overflow = false;
    let mut first_f = None;
    let mut evaluated = Vec::new();
    let outcome = minimize(case.start(), bounds.clone(), options.clone(), |x, g| {
        calls += 1;
        assert!(
            calls <= options.max_evaluations,
            "callback budget overshoot"
        );
        for (x, b) in x.iter().zip(&bounds) {
            assert!(
                x.is_finite() && b.lower.is_none_or(|l| *x >= l) && b.upper.is_none_or(|u| *x <= u),
                "infeasible callback: {x:?}, {b:?}"
            );
        }
        if injected_error == Some(calls) {
            error_called = true;
            anyhow::bail!("convex-fuzz injected callback error");
        }
        let f = case.evaluate(x, g);
        overflow |= !f.is_finite() || g.iter().any(|v| !v.is_finite());
        if first_f.is_none() {
            first_f = Some(f);
        }
        evaluated.push((x.to_vec(), f, g.to_vec()));
        Ok(f)
    });
    if error_called {
        prop_assert_eq!(calls, injected_error.unwrap());
        let err = outcome.expect_err("callback error must propagate");
        prop_assert!(
            err.to_string()
                .contains("convex-fuzz injected callback error")
        );
        return Ok(Audit {
            status: None,
            certified: false,
            roundoff_limited: false,
            evaluations: calls,
        });
    }
    if overflow && !require_solution {
        prop_assert!(outcome.is_err(), "nonfinite fg must be rejected");
        return Ok(Audit {
            status: None,
            certified: false,
            roundoff_limited: false,
            evaluations: calls,
        });
    }
    let r = outcome.map_err(|e| TestCaseError::fail(format!("unexpected error: {e}")))?;
    prop_assert_eq!(calls, r.evaluations);
    match r.status {
        Status::MaxEvaluations => prop_assert_eq!(calls, options.max_evaluations),
        Status::MaxIterations => prop_assert_eq!(r.iterations, options.max_iterations),
        Status::ConvergedProjectedGradient | Status::ConvergedFunctionReduction => {
            prop_assert_eq!(r.line_search_failure, None)
        }
        Status::LineSearchFailed => prop_assert!(r.line_search_failure.is_some()),
    }
    prop_assert!(
        r.evaluations <= options.max_evaluations && r.iterations <= options.max_iterations
    );
    // At most two searches (history and fallback) per accepted update, plus
    // one failed attempt; the initial callback is outside their trial budgets.
    prop_assert!(r.evaluations <= 1 + 2 * (r.iterations + 1) * options.max_line_search_steps);
    prop_assert!(r.f.is_finite() && r.f <= first_f.unwrap());
    for (x, b) in r.x.iter().zip(&bounds) {
        prop_assert!(
            x.is_finite() && b.lower.is_none_or(|l| *x >= l) && b.upper.is_none_or(|u| *x <= u)
        );
    }
    let mut replay_g = vec![T::zero(); r.x.len()];
    let replay_f = case.evaluate(&r.x, &mut replay_g);
    prop_assert_eq!(r.f, replay_f);
    prop_assert_eq!(&r.gradient, &replay_g);
    prop_assert!(
        evaluated
            .iter()
            .any(|(x, f, g)| *x == r.x && *f == r.f && *g == r.gradient),
        "returned state was never evaluated"
    );
    if r.iterations == 0 {
        // Not a last, merely improved, unaccepted search trial.
        let projected: Vec<_> = case
            .start::<T>()
            .iter()
            .zip(&bounds)
            .map(|(&x, b)| {
                let mut z = x;
                if let Some(l) = b.lower {
                    z = z.max(l);
                }
                if let Some(u) = b.upper {
                    z = z.min(u);
                }
                z
            })
            .collect();
        prop_assert_eq!(&r.x, &projected);
    }
    let x: Vec<_> = r.x.iter().map(|v| v.to_f64().unwrap()).collect();
    let (gradient, lipschitz, magnitude) = case.oracle(&x);
    let (box64, _) = case.geometry();
    let projection_residual = |gradient: &[f64]| {
        gradient
            .iter()
            .zip(&x)
            .zip(&box64)
            .map(|((&g, &x), b)| {
                let mut z = x - g;
                if let Some(l) = b.lower {
                    z = z.max(l);
                }
                if let Some(u) = b.upper {
                    z = z.min(u);
                }
                (x - z).abs()
            })
            .fold(0_f64, f64::max)
    };
    let scale = 2_f64.powi(i32::from(case.scale));
    let normalized: Vec<_> = gradient.iter().map(|g| g / scale).collect();
    let normalized_pg = projection_residual(&normalized);
    let goal = precision::<T>();
    let pg = gradient
        .iter()
        .zip(&x)
        .zip(box64)
        .map(|((&g, &x), b)| {
            // Independently implement x-P(x-g), not the solver's distance-clipping formula.
            let mut projected = x - g;
            if let Some(l) = b.lower {
                projected = projected.max(l);
            }
            if let Some(u) = b.upper {
                projected = projected.min(u);
            }
            (x - projected).abs()
        })
        .fold(0_f64, f64::max);
    let tol = options.pgtol.to_f64().unwrap();
    let roundoff = 64. * T::epsilon().to_f64().unwrap() * (1. + magnitude);
    let reported = r.projected_gradient_inf_norm.to_f64().unwrap();
    prop_assert!(
        (pg - reported).abs() <= roundoff,
        "projected-gradient mismatch: oracle={pg:e}, reported={reported:e}, error_bound={roundoff:e}"
    );
    if r.status == Status::ConvergedProjectedGradient {
        prop_assert!(
            reported <= tol && pg <= tol + roundoff,
            "false projected-gradient certificate: {r:?}, oracle={pg:e}"
        );
    }
    let coordinate_scale = 1. + x.iter().map(|x| x.abs()).fold(0_f64, f64::max);
    let reference_error = 64. * f64::EPSILON * (coordinate_scale + magnitude / scale);
    let certified = normalized_pg + reference_error <= goal;
    let normalized_roundoff = 64.
        * T::epsilon().to_f64().unwrap()
        * (coordinate_scale + magnitude / scale)
        + 64. * T::min_positive_value().to_f64().unwrap() * T::epsilon().to_f64().unwrap() / scale;
    let roundoff_limited = !certified && normalized_pg <= goal + normalized_roundoff;
    if require_solution {
        prop_assert!(
            certified,
            "requested normalized KKT goal not reached (candidate, not automatically a bug): {r:?}, oracle={normalized_pg:e}, goal={goal:e}, roundoff={:e}",
            roundoff / scale
        );
        let distance = x
            .iter()
            .zip(case.target())
            .map(|(x, t)| (x - t).abs())
            .fold(0_f64, f64::max);
        let mu = 2_f64.powi(i32::from(case.mu));
        let bound = (1. + (1. + lipschitz / scale) / mu)
            * (x.len() as f64).sqrt()
            * (goal + 64. * f64::EPSILON);
        prop_assert!(
            distance <= bound,
            "planted optimum mismatch: distance={distance:e}, bound={bound:e}"
        );
    }
    Ok(Audit {
        status: Some(r.status),
        certified,
        roundoff_limited,
        evaluations: r.evaluations,
    })
}

fn precision<T: Float + FromPrimitive>() -> f64 {
    if T::epsilon() > c(1e-10) { 1e-5 } else { 1e-7 }
}

fn options<T: Float + FromPrimitive>(case: &Case) -> Options<T> {
    // Stricter internal tolerance leaves room for an independently recomputed,
    // scale-normalized certificate; near bounds the mapping also has x units.
    let tol = 0.25 * precision::<T>() * 2_f64.powi(i32::from(case.scale)).min(1.);
    Options {
        memory: case.memory,
        pgtol: c(tol),
        ..Options::default()
    }
}

fn first_step<T: Float + FromPrimitive + std::fmt::Debug>(
    case: &Case,
    wolfe: (f64, f64),
) -> Result<(), TestCaseError> {
    let bounds = case.bounds::<T>();
    let mut x0 = case.start::<T>();
    for (x, b) in x0.iter_mut().zip(&bounds) {
        if let Some(l) = b.lower {
            *x = x.max(l);
        }
        if let Some(u) = b.upper {
            *x = x.min(u);
        }
    }
    let mut g0 = vec![T::zero(); x0.len()];
    let f0 = case.evaluate(&x0, &mut g0);
    let opts = Options {
        max_iterations: 1,
        wolfe_c1: c(wolfe.0),
        wolfe_c2: c(wolfe.1),
        ..options::<T>(case)
    };
    let (c1, c2) = (opts.wolfe_c1, opts.wolfe_c2);
    let r = minimize(case.start(), bounds.clone(), opts, |x, g| {
        Ok(case.evaluate(x, g))
    })
    .map_err(|e| TestCaseError::fail(e.to_string()))?;
    if r.iterations == 0 {
        prop_assert_eq!(r.x, x0);
        return Ok(());
    }
    prop_assert_eq!(r.iterations, 1);
    let mut d0 = T::zero();
    let mut d1 = T::zero();
    let mut magnitude = T::zero();
    let mut endpoint = false;
    for i in 0..x0.len() {
        let d = r.x[i] - x0[i];
        d0 = d0 + g0[i] * d;
        d1 = d1 + r.gradient[i] * d;
        magnitude = magnitude
            + (g0[i] * d).abs()
            + (r.gradient[i] * d).abs()
            + (g0[i].abs() + r.gradient[i].abs()) * (r.x[i].abs() + x0[i].abs());
        let eps = c::<T>(32.) * T::epsilon() * (r.x[i].abs() + x0[i].abs());
        endpoint |= (d > T::zero() && bounds[i].upper.is_some_and(|u| (u - r.x[i]).abs() <= eps))
            || (d < T::zero() && bounds[i].lower.is_some_and(|l| (r.x[i] - l).abs() <= eps));
    }
    let eps = c::<T>(64.) * T::epsilon();
    prop_assert!(d0 < T::zero(), "non-descent first update: {r:?}");
    prop_assert!(
        r.f <= f0 + c1 * d0 + eps * (f0.abs() + r.f.abs() + (c1 * d0).abs()),
        "unchecked Armijo update: {r:?}"
    );
    prop_assert!(
        endpoint || d1.abs() <= c2 * d0.abs() + eps * magnitude,
        "unchecked Wolfe update: {r:?}, d0={d0:?}, d1={d1:?}"
    );
    Ok(())
}

#[test]
fn large_offset_can_hide_progress_without_hiding_kkt_error() {
    // Automatically minimized by the KKT-hunting campaign. This tests the
    // oracle, not a requirement to preserve the solver's premature stopping.
    let case = Case {
        coordinates: vec![Coordinate {
            target: 0,
            start: 1,
            bound: 0,
            margin: 1,
            normal: 1,
        }],
        terms: vec![],
        scale: -4,
        mu: 0,
        offset: Some(24),
        memory: 1,
    };
    let mut g = [0_f32];
    assert_eq!(case.evaluate(&[0_f32], &mut g), 16777216.);
    assert_eq!(case.evaluate(&[0.21875_f32], &mut g), 16777216.);
    assert_eq!(g[0], 0.013671875);
    let (reference, _, _) = case.oracle(&[0.21875]);
    assert_eq!(reference[0] / 2_f64.powi(-4), 0.21875);
    assert!(reference[0] / 2_f64.powi(-4) > precision::<f32>());
}

proptest! {
    #![proptest_config(config())]
    #[test]
    fn first_update_satisfies_wolfe_or_feasible_endpoint(case in cases(false),
        c1 in prop::sample::select(vec![1e-4_f64,1e-3]),
        c2 in prop::sample::select(vec![0.1_f64,0.5,0.9])) {
        first_step::<f64>(&case,(c1,c2))?;
        first_step::<f32>(&case,(c1,c2))?;
    }
    #[test]
    fn generated_oracle_is_convex_and_differentiable(case in cases(false)) {
        let target=case.target();
        let mut g=vec![0.;target.len()];
        prop_assert_eq!(case.evaluate(&target,&mut g),0.);
        let (bounds,normal)=case.geometry();
        let scale=2_f64.powi(i32::from(case.scale));
        for i in 0..g.len() {
            prop_assert_eq!(g[i],normal[i]*scale);
            prop_assert!(g[i]==0. || (g[i]>0. && bounds[i].lower==Some(target[i]))
                || (g[i]<0. && bounds[i].upper==Some(target[i])));
        }
        let x=case.start::<f64>();
        let f=case.evaluate(&x,&mut g);
        let (_,lipschitz,_)=case.oracle(&x);
        let step=2_f64.powi(-18);
        for i in 0..x.len() {
            let mut plus=x.clone(); plus[i]+=step;
            let mut minus=x.clone(); minus[i]-=step;
            let fp=case.evaluate(&plus,&mut vec![0.;x.len()]);
            let fm=case.evaluate(&minus,&mut vec![0.;x.len()]);
            let numerical=(fp-fm)/(2.*step);
            let error=lipschitz*step+64.*f64::EPSILON*(1.+fp.abs()+fm.abs())/step;
            prop_assert!((numerical-g[i]).abs()<=error,"gradient generator error at coordinate {i}: {numerical} vs {}",g[i]);
        }
        let midpoint: Vec<_>=x.iter().zip(&target).map(|(x,t)|0.5*(x+t)).collect();
        let middle=case.evaluate(&midpoint,&mut vec![0.;x.len()]);
        prop_assert!(middle<=0.5*f+64.*f64::EPSILON*(1.+f.abs()));
    }
    #[test]
    fn ordinary_convex_f64(case in cases(false)) {
        run::<f64>(&case,options(&case),None,true)?;
    }
    #[test]
    fn ordinary_convex_f32(case in cases(false)) {
        run::<f32>(&case,options(&case),None,true)?;
    }
    #[test]
    fn budgets_and_callback_errors(case in cases(false), evaluations in 1_usize..33,
        iterations in 1_usize..13, trials in 1_usize..9, error in prop::option::of(1_usize..33)) {
        let opts=Options{max_iterations:iterations,max_evaluations:evaluations,max_line_search_steps:trials,..options(&case)};
        run::<f64>(&case,opts,error,false)?;
        let opts=Options{max_iterations:iterations,max_evaluations:evaluations,max_line_search_steps:trials,..options(&case)};
        run::<f32>(&case,opts,error,false)?;
    }
}

#[test]
#[ignore = "extended numerical exploration; reports non-convergence separately from invariant violations"]
fn extreme_convex_campaign() {
    let mut config = config();
    let search_mode = std::env::var("CONVEX_FUZZ_HUNT").unwrap_or_default();
    assert!(
        matches!(
            search_mode.as_str(),
            "" | "search-f64" | "search-f32" | "scale-f64" | "scale-f32"
        ),
        "unknown CONVEX_FUZZ_HUNT"
    );
    let hunt = std::env::var("CONVEX_FUZZ_REQUIRE_KKT").is_ok_and(|v| v == "1");
    assert!(
        !hunt || search_mode.is_empty(),
        "select only one discovery mode"
    );
    config.failure_persistence = Some(Box::new(if search_mode == "scale-f64" {
        FileFailurePersistence::Direct(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/convex_fuzz_scale_f64.proptest-regressions"
        ))
    } else if search_mode == "scale-f32" {
        FileFailurePersistence::Direct(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/convex_fuzz_scale_f32.proptest-regressions"
        ))
    } else if search_mode == "search-f64" {
        FileFailurePersistence::Direct(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/convex_fuzz_search_f64.proptest-regressions"
        ))
    } else if search_mode == "search-f32" {
        FileFailurePersistence::Direct(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/convex_fuzz_search_f32.proptest-regressions"
        ))
    } else if hunt {
        FileFailurePersistence::Direct(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/convex_fuzz_candidates.proptest-regressions"
        ))
    } else {
        FileFailurePersistence::Direct(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/convex_fuzz_extreme.proptest-regressions"
        ))
    }));
    if std::env::var_os("PROPTEST_CASES").is_none() {
        config.cases = 4096;
    }
    let mut runner = TestRunner::new(config);
    let counts = RefCell::new([0_usize; 6]);
    let statuses = RefCell::new(std::collections::BTreeMap::<String, usize>::new());
    runner
        .run(&cases(true).prop_map(|mut case| {
            if !search_mode.is_empty() { case.offset=None; }
            if search_mode.starts_with("scale-") { case.scale *=10; }
            case
        }), |case| {
            let mut audits=Vec::new();
            if !search_mode.ends_with("f32") { audits.push(run::<f64>(&case,options(&case),None,hunt)?); }
            if !search_mode.ends_with("f64") { audits.push(run::<f32>(&case,options(&case),None,hunt)?); }
            for audit in audits {
                prop_assert!(search_mode.is_empty() || audit.status!=Some(Status::LineSearchFailed) || audit.certified,
                    "{search_mode}: line search failed above independent goal (candidate): {audit:?}");
                let mut counts = counts.borrow_mut();
                counts[0] += 1;
                counts[1] += usize::from(audit.certified);
                counts[2] += audit.evaluations;
                counts[3] += usize::from(audit.status == Some(Status::ConvergedFunctionReduction));
                counts[4] += usize::from(audit.roundoff_limited);
                counts[5] += usize::from(audit.status.is_none());
                *statuses
                    .borrow_mut()
                    .entry(format!("{:?}", audit.status))
                    .or_default() += 1;
            }
            Ok(())
        })
        .unwrap();
    let [
        runs,
        certified,
        evaluations,
        function_stops,
        roundoff_limited,
        nonfinite_callbacks,
    ] = *counts.borrow();
    eprintln!(
        "extreme campaign: runs={runs}, independently_certified={certified}, roundoff_limited={roundoff_limited}, nonfinite_callbacks={nonfinite_callbacks}, function_stops={function_stops}, fg_calls={evaluations}, statuses={:?}",
        statuses.borrow()
    );
}
