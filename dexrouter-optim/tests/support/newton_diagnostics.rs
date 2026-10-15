//! Opt-in convergence diagnostics; probes are not part of optimizer timings.
use super::*;

fn ratio(
    route: &Route<NonnegativeLinear, UniswapV3>,
    price: &ndarray::Array1<f64>,
    evaluation: &dexrouter_optim::RouteEvaluation,
    options: &SolveOptions,
    i: usize,
) -> f64 {
    let lower = route.objective.lower_bounds();
    let upper = route.objective.upper_bounds();
    let g = evaluation.gradient[i];
    if (price[i] == lower[i] && g >= 0.0) || (price[i] == upper[i] && g <= 0.0) {
        return 0.0;
    }
    g.abs() * price[i]
        / (options.absolute_residual_value
            + options.relative_residual_tolerance * evaluation.turnover[i] * price[i])
}

#[test]
#[ignore = "diagnose native convergence and neighboring representable prices"]
fn inspect_snapshot_stagnation() {
    let pools: Vec<RawUniswapV3Pool> =
        serde_json::from_str(include_str!("../fixtures/uniswap_v3_pools.json")).unwrap();
    for (label, cutoff, tol) in [
        ("filtered", 1000.0, 1e-8),
        ("full", 0.0, 1e-6),
        ("full-strict", 0.0, 1e-8),
    ] {
        let full = build_full_market_route_with_tvl(pools.clone(), cutoff).unwrap();
        let lower = full.route.objective.lower_bounds();
        let upper = full.route.objective.upper_bounds();
        let o = SolveOptions {
            absolute_residual_value: tol,
            ..Default::default()
        };
        for (backend, pg) in [("matched", o.optimizer.pgtol), ("default", 1e-8)] {
            let start = Instant::now();
            let r = solve_price_with_newton(
                full.route.clone(),
                full.initial_price.clone(),
                o.clone(),
                dexrouter_newton::Options {
                    pgtol: pg,
                    max_iterations: o.optimizer.max_iterations,
                    max_evaluations: o.optimizer.max_evaluations,
                    ..Default::default()
                },
            )
            .unwrap();
            let elapsed = start.elapsed().as_secs_f64();
            let worst = independent(&full.route, &r, &o);
            println!(
                "diagnostic snapshot={label} backend={backend} status={:?} assessment={:?} FG={} HVP={} models={} it={} restarts={} KKT={worst:.9e} seconds={elapsed:.6}",
                r.status,
                r.assessment,
                r.evaluations,
                r.hessian_products,
                r.hessian_evaluations,
                r.iterations,
                r.restarts
            );
            if r.is_usable() {
                continue;
            }
            let e = full.route.evaluate(&r.price);
            let mut indices: Vec<_> = (0..full.route.tokens).collect();
            indices.sort_by(|&a, &b| {
                ratio(&full.route, &r.price, &e, &o, b).total_cmp(&ratio(
                    &full.route,
                    &r.price,
                    &e,
                    &o,
                    a,
                ))
            });
            let i = indices[0];
            let (_, (_, (a, b))) = full
                .route
                .markets
                .iter()
                .enumerate()
                .filter(|(_, (_, (a, b)))| *a == i || *b == i)
                .max_by(|(_, (ma, (a, b))), (_, (mb, (c, d)))| {
                    ma.relative_curvature([r.price[*a], r.price[*b]])
                        .unwrap()
                        .total_cmp(&mb.relative_curvature([r.price[*c], r.price[*d]]).unwrap())
                })
                .unwrap();
            let j = if *a == i { *b } else { *a };
            let h = full
                .route
                .markets
                .iter()
                .filter(|(_, (a, b))| *a == i || *b == i)
                .map(|(m, (a, b))| m.relative_curvature([r.price[*a], r.price[*b]]).unwrap())
                .sum::<f64>()
                / r.price[i].powi(2);
            if h.is_finite() && h > 0.0 && i != j {
                let mut best = (f64::INFINITY, 0.0, 0, 0.0);
                for offset in -32_i32..=32 {
                    let mut p = r.price.clone();
                    for _ in 0..offset.abs() {
                        p[j] = if offset < 0 {
                            p[j].next_down()
                        } else {
                            p[j].next_up()
                        };
                    }
                    if p[j] < lower[j] || p[j] > upper[j] {
                        continue;
                    }
                    let center = ((p[j] - r.price[j]) * (r.price[i] / r.price[j])
                        - e.gradient[i] / h)
                        + r.price[i];
                    for pi in [center.next_down(), center, center.next_up()] {
                        if !pi.is_finite() || pi <= 0.0 || pi < lower[i] || pi > upper[i] {
                            continue;
                        }
                        p[i] = pi;
                        let ep = full.route.evaluate(&p);
                        if !ep.cost.is_finite() || ep.gradient.iter().any(|g| !g.is_finite()) {
                            continue;
                        }
                        let worst = (0..full.route.tokens)
                            .map(|k| ratio(&full.route, &p, &ep, &o, k))
                            .fold(0.0, f64::max);
                        let mut change = (0.0, 0.0);
                        add(&mut change, full.route.objective.value_change(&p, &r.price));
                        for (m, (a, b)) in &full.route.markets {
                            add(
                                &mut change,
                                m.value_change(
                                    [p[*a], p[*b]],
                                    [r.price[*a], r.price[*b]],
                                    m.evaluate([p[*a], p[*b]]),
                                    m.evaluate([r.price[*a], r.price[*b]]),
                                ),
                            );
                        }
                        let delta = change.0 + change.1;
                        if worst < best.0 {
                            best = (worst, delta, offset, pi - r.price[i]);
                        }
                    }
                }
                println!(
                    "  joint token=({i},{j}) best_KKT={:.9e} delta={:.9e} neighbor_j={} delta_i={:.9e}",
                    best.0, best.1, best.2, best.3
                );
            }
            for &i in indices.iter().take(3) {
                println!(
                    "  token={i} p={:.17e} g={:.9e} turnover={:.9e} ratio={:.9e}",
                    r.price[i],
                    e.gradient[i],
                    e.turnover[i],
                    ratio(&full.route, &r.price, &e, &o, i)
                );
                for candidate in [r.price[i].next_down(), r.price[i].next_up()] {
                    let mut p = r.price.clone();
                    p[i] = candidate;
                    let ep = full.route.evaluate(&p);
                    println!(
                        "    neighbor dp={:.3e} g={:.9e} ratio={:.9e}",
                        candidate - r.price[i],
                        ep.gradient[i],
                        ratio(&full.route, &p, &ep, &o, i)
                    );
                }
                for (j, (m, (a, b))) in full
                    .route
                    .markets
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, (a, b)))| *a == i || *b == i)
                {
                    let t = m.evaluate([r.price[*a], r.price[*b]]);
                    let local = usize::from(*b == i);
                    println!(
                        "    pool={j} pair=({a},{b}) net={:.9e} curvature={:.9e}",
                        t.output[local] - t.input[local],
                        m.relative_curvature([r.price[*a], r.price[*b]])
                            .unwrap_or(f64::NAN)
                    );
                }
            }
        }
    }
}
