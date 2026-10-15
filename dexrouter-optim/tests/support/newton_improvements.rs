//! Same inputs and economic checks for Jacobi, exact Schur, and hi+lo refinement.
use super::*;
use dexrouter_optim::{market::PreciseMarket, solve_price_with_precise_newton};
#[test]
#[ignore = "offline paired release benchmark for Schur and precise prices"]
fn compare_schur_and_precise_newton() {
    let pools: Vec<RawUniswapV3Pool> =
        serde_json::from_str(include_str!("../fixtures/uniswap_v3_pools.json")).unwrap();
    let repeats = std::env::var("NEWTON_IMPROVEMENT_REPEATS")
        .map(|v| v.parse::<usize>().unwrap())
        .unwrap_or(8);
    let mut records = Vec::new();
    for (label, cutoff, tolerance) in [
        ("filtered", 1000.0, 1e-8),
        ("full", 0.0, 1e-6),
        ("full-strict", 0.0, 1e-8),
    ] {
        let full = build_full_market_route_with_tvl(pools.clone(), cutoff).unwrap();
        for repeat in 0..repeats {
            for k in 0..4 {
                let mode = (k + repeat) % 4;
                let options = SolveOptions {
                    absolute_residual_value: tolerance,
                    newton_schur_precondition: mode == 1 || mode == 2,
                    ..Default::default()
                };
                let n = dexrouter_newton::Options {
                    pgtol: options.optimizer.pgtol,
                    max_iterations: options.optimizer.max_iterations,
                    max_evaluations: options.optimizer.max_evaluations,
                    ..Default::default()
                };
                let route = full.route.clone(); // identical input preparation outside timing
                let now = Instant::now();
                let record = if mode >= 2 {
                    let r = solve_price_with_precise_newton(
                        &route,
                        full.initial_price.clone(),
                        options.clone(),
                        n.clone(),
                    )
                    .unwrap();
                    let seconds = now.elapsed().as_secs_f64();
                    let mut sums = vec![(0.0, 0.0); full.route.tokens];
                    let mut turnover = sums.clone();
                    for (m, (a, b)) in &full.route.markets {
                        let t = m.evaluate_precise([r.price[*a], r.price[*b]]);
                        for (j, i) in [*a, *b].into_iter().enumerate() {
                            add(&mut sums[i], t.output[j] - t.input[j]);
                            add(&mut turnover[i], t.output[j] + t.input[j]);
                        }
                    }
                    let mut kkt = 0.0_f64;
                    for i in 1..sums.len() {
                        let g = sums[i].0 + sums[i].1;
                        let p = r.price[i].hi();
                        kkt = kkt.max(
                            g.abs() * p
                                / (tolerance
                                    + options.relative_residual_tolerance
                                        * (turnover[i].0 + turnover[i].1)
                                        * p),
                        );
                    }
                    assert!((kkt - r.max_kkt_residual_ratio).abs() < 1e-6);
                    assert!(
                        r.evaluations <= n.max_evaluations
                            && r.hessian_products <= n.max_hessian_products
                            && r.iterations <= n.max_iterations
                    );
                    assert!(r.refinement_objective_change <= 0.0);
                    assert!(r.is_usable(), "{label}: {r:?}");
                    serde_json::json!({"snapshot":label,"repeat":repeat,"mode":if mode==2{"schur-twofold"}else{"jacobi-warm-twofold"},"fg":r.evaluations,"hvp":r.hessian_products,"models":r.hessian_evaluations,"iterations":r.iterations,"seconds":seconds,"kkt":kkt,"usable":r.is_usable(),"status":format!("{:?}",r.status),"cost":r.evaluation.cost,"refinement_delta":r.refinement_objective_change,"warm_kkt":r.warm_start.max_kkt_residual_ratio,"warm_fg":r.warm_start.evaluations,"warm_prices":r.warm_start.price.iter().map(|p|format!("{:016x}",p.to_bits())).collect::<Vec<_>>(),"prices":r.price.iter().map(|p|serde_json::json!({"hi":format!("{:016x}",p.hi().to_bits()),"lo":format!("{:016x}",p.lo().to_bits())})).collect::<Vec<_>>()})
                } else {
                    let r = solve_price_with_newton(
                        route,
                        full.initial_price.clone(),
                        options.clone(),
                        n,
                    )
                    .unwrap();
                    let seconds = now.elapsed().as_secs_f64();
                    let kkt = independent(&full.route, &r, &options);
                    serde_json::json!({"snapshot":label,"repeat":repeat,"mode":if mode==0{"jacobi"}else{"schur"},"fg":r.evaluations,"hvp":r.hessian_products,"models":r.hessian_evaluations,"iterations":r.iterations,"seconds":seconds,"kkt":kkt,"usable":r.is_usable(),"status":format!("{:?}",r.status),"cost":r.cost})
                };
                println!("IMPROVEMENT_RESULT {record}");
                records.push(record);
            }
        }
    }
    std::fs::write(
        "../target/newton_improvements.json",
        serde_json::to_string_pretty(&records).unwrap(),
    )
    .unwrap();
}
