//! Paired, unchanged-objective comparison of Jacobi and support-forest PCG.
use super::*;
#[test]
#[ignore = "offline paired preconditioner comparison; use release and one test thread"]
fn compare_newton_preconditioners() {
    let pools: Vec<RawUniswapV3Pool> =
        serde_json::from_str(include_str!("../fixtures/uniswap_v3_pools.json")).unwrap();
    let repeats = std::env::var("NEWTON_CONDITIONING_REPEATS")
        .map(|v| v.parse::<usize>().unwrap())
        .unwrap_or(3);
    for (label, cutoff, tolerance) in [
        ("filtered", 1000.0, 1e-8),
        ("full", 0.0, 1e-6),
        ("full-strict", 0.0, 1e-8),
    ] {
        let full = build_full_market_route_with_tvl(pools.clone(), cutoff).unwrap();
        for repeat in 0..repeats {
            for tree in if repeat % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let options = SolveOptions {
                    absolute_residual_value: tolerance,
                    newton_tree_precondition: tree,
                    ..Default::default()
                };
                let newton = dexrouter_newton::Options {
                    pgtol: options.optimizer.pgtol,
                    max_iterations: options.optimizer.max_iterations,
                    max_evaluations: options.optimizer.max_evaluations,
                    ..Default::default()
                };
                let start = Instant::now();
                let r = solve_price_with_newton(
                    full.route.clone(),
                    full.initial_price.clone(),
                    options.clone(),
                    newton.clone(),
                )
                .unwrap();
                let seconds = start.elapsed().as_secs_f64();
                let kkt = independent(&full.route, &r, &options);
                assert!(r.hessian_products <= newton.max_hessian_products);
                if label != "full-strict" {
                    assert!(r.is_usable(), "{label}: {r:?}");
                }
                println!(
                    "conditioning snapshot={label} repeat={repeat} tree={tree} status={:?} assessment={:?} FG={} HVP={} models={} iterations={} independent_KKT={kkt:.9e} cost={:.12e} seconds={seconds:.6}",
                    r.status,
                    r.assessment,
                    r.evaluations,
                    r.hessian_products,
                    r.hessian_evaluations,
                    r.iterations,
                    r.cost
                );
                println!(
                    "CONDITIONING_RESULT {}",
                    serde_json::json!({
                        "snapshot":label,"repeat":repeat,"preconditioner":if tree {"forest"}else{"jacobi"},
                        "pools":full.route.markets.len(),"tokens":full.route.tokens,
                        "status":format!("{:?}",r.status),"assessment":format!("{:?}",r.assessment),
                        "fg":r.evaluations,"hvp":r.hessian_products,"models":r.hessian_evaluations,
                        "iterations":r.iterations,"independent_kkt_ratio":kkt,"cost":r.cost,"seconds":seconds,
                        "absolute_residual_value":options.absolute_residual_value,
                        "relative_residual_tolerance":options.relative_residual_tolerance,"pgtol":newton.pgtol
                    })
                );
            }
        }
    }
}
