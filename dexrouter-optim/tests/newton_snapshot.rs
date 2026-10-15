//! Opt-in benchmark, not a claim that every backend reaches every tolerance.
use dexrouter_optim::{
    Route, SolveDiagnostics, SolveOptions,
    market::{Market, UniswapV3},
    snapshot::{RawUniswapV3Pool, build_full_market_route_with_tvl},
    solve_price_with_newton, solve_price_with_options,
    utility::{NonnegativeLinear, UtilityConjugate},
};
use std::time::Instant;

#[path = "support/laplacian_analysis.rs"]
mod laplacian_analysis;
#[path = "support/newton_conditioning.rs"]
mod newton_conditioning;
#[path = "support/newton_diagnostics.rs"]
mod newton_diagnostics;
#[path = "support/newton_improvements.rs"]
mod newton_improvements;
#[path = "support/newton_profile.rs"]
mod newton_profile;
#[path = "support/newton_resolution.rs"]
mod newton_resolution;
fn add((s, c): &mut (f64, f64), v: f64) {
    let n = *s + v;
    *c += if s.abs() >= v.abs() {
        (*s - n) + v
    } else {
        (v - n) + *s
    };
    *s = n;
}
fn independent<S: std::fmt::Debug>(
    route: &Route<NonnegativeLinear, UniswapV3>,
    r: &SolveDiagnostics<S>,
    o: &SolveOptions,
) -> f64 {
    let mut net = vec![(0.0, 0.0); route.tokens];
    let mut turnover = net.clone();
    for (m, (a, b)) in &route.markets {
        let (input, output) = m.arbitrage([r.price[*a], r.price[*b]]);
        for (k, i) in [*a, *b].into_iter().enumerate() {
            add(&mut net[i], output[k] - input[k]);
            add(&mut turnover[i], output[k] + input[k]);
        }
    }
    let ug = route.objective.grad(&r.price);
    let lower = route.objective.lower_bounds();
    let upper = route.objective.upper_bounds();
    let mut worst = 0.0_f64;
    for i in 0..route.tokens {
        let p = r.price[i];
        assert!(p.is_finite() && p > 0.0 && p >= lower[i] && p <= upper[i]);
        add(&mut net[i], ug[i]);
        let g = net[i].0 + net[i].1;
        let kkt = if (p == lower[i] && g >= 0.0) || (p == upper[i] && g <= 0.0) {
            0.0
        } else {
            g.abs()
        };
        let allowed = o.absolute_residual_value
            + o.relative_residual_tolerance * (turnover[i].0 + turnover[i].1) * p;
        worst = worst.max(kkt * p / allowed);
        if r.is_usable() {
            assert!(kkt * p <= allowed, "false acceptance token={i}: {r:?}");
            assert!((-g * p).max(0.0) <= allowed);
        }
    }
    if !r.is_usable() {
        assert!(worst > 1.0 || r.max_deficit_ratio > 1.0);
    }
    assert!(r.evaluations <= o.optimizer.max_evaluations);
    assert!(r.iterations <= o.optimizer.max_iterations);
    worst
}
#[test]
#[ignore = "offline repeated backend comparison; run release with nocapture"]
fn compare_snapshot_backends() {
    let pools: Vec<RawUniswapV3Pool> =
        serde_json::from_str(include_str!("fixtures/uniswap_v3_pools.json")).unwrap();
    for (label, cutoff, tol) in [
        ("filtered", 1000.0, 1e-8),
        ("full", 0.0, 1e-6),
        ("full-strict", 0.0, 1e-8),
    ] {
        let full = build_full_market_route_with_tvl(pools.clone(), cutoff).unwrap();
        let o = SolveOptions {
            absolute_residual_value: tol,
            ..Default::default()
        };
        for repeat in 0..3 {
            // Alternate ordering so one backend does not always get the warm cache.
            for backend in if repeat % 2 == 0 {
                ["lbfgsb", "newton-matched", "newton-default"]
            } else {
                ["newton-default", "newton-matched", "lbfgsb"]
            } {
                let start = Instant::now();
                let (status, assessment, fg, hvp, models, kkt, cost, elapsed) =
                    if backend.starts_with("newton") {
                        let n = dexrouter_newton::Options {
                            pgtol: if backend == "newton-matched" {
                                o.optimizer.pgtol
                            } else {
                                dexrouter_newton::Options::default().pgtol
                            },
                            max_iterations: o.optimizer.max_iterations,
                            max_evaluations: o.optimizer.max_evaluations,
                            ..Default::default()
                        };
                        let r = solve_price_with_newton(
                            full.route.clone(),
                            full.initial_price.clone(),
                            o.clone(),
                            n.clone(),
                        )
                        .unwrap();
                        let elapsed = start.elapsed().as_secs_f64();
                        let kkt = independent(&full.route, &r, &o);
                        // Every regular dataset/backend must pass unchanged
                        // economic checks, including matched-PG full Newton.
                        // Strict failures are still independently diagnosed.
                        if label != "full-strict" {
                            assert!(r.is_usable(), "{label} {backend}: {r:?}");
                        }
                        assert!(r.hessian_products <= n.max_hessian_products);
                        (
                            format!("{:?}", r.status),
                            r.assessment,
                            r.evaluations,
                            r.hessian_products,
                            r.hessian_evaluations,
                            kkt,
                            r.cost,
                            elapsed,
                        )
                    } else {
                        let r = solve_price_with_options(
                            full.route.clone(),
                            full.initial_price.clone(),
                            o.clone(),
                        )
                        .unwrap();
                        let elapsed = start.elapsed().as_secs_f64();
                        let kkt = independent(&full.route, &r, &o);
                        // Every regular dataset/backend must pass unchanged
                        // economic checks, including matched-PG full Newton.
                        // Strict failures are still independently diagnosed.
                        if label != "full-strict" {
                            assert!(r.is_usable(), "{label} {backend}: {r:?}");
                        }
                        (
                            format!("{:?}", r.status),
                            r.assessment,
                            r.evaluations,
                            0,
                            0,
                            kkt,
                            r.cost,
                            elapsed,
                        )
                    };
                println!(
                    "snapshot={label} repeat={repeat} backend={backend} pools={} tokens={} status={status} assessment={assessment:?} FG={fg} HVP={hvp} models={models} independent_KKT={kkt:.6e} cost={cost:.12e} seconds={elapsed:.6}",
                    full.route.markets.len(),
                    full.route.tokens
                );
            }
        }
    }
}
