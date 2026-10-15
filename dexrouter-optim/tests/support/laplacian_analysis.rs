//! Offline spectral/pruning experiments; never changes production pool selection.
use super::*;
use serde_json::{Value, json};
fn options(tolerance: f64) -> SolveOptions {
    SolveOptions {
        absolute_residual_value: tolerance,
        ..Default::default()
    }
}
fn newton_options(o: &SolveOptions) -> dexrouter_newton::Options {
    dexrouter_newton::Options {
        pgtol: o.optimizer.pgtol,
        max_iterations: o.optimizer.max_iterations,
        max_evaluations: o.optimizer.max_evaluations,
        ..Default::default()
    }
}
pub(super) fn check<M: Market>(
    route: &Route<NonnegativeLinear, M>,
    price: &ndarray::Array1<f64>,
    o: &SolveOptions,
) -> Value {
    let e = route.evaluate(price);
    let lower = route.objective.lower_bounds();
    let upper = route.objective.upper_bounds();
    let mut net = vec![(0.0, 0.0); route.tokens];
    let mut turnover = net.clone();
    for (m, (a, b)) in &route.markets {
        let (input, output) = m.arbitrage([price[*a], price[*b]]);
        for (k, i) in [*a, *b].into_iter().enumerate() {
            add(&mut net[i], output[k] - input[k]);
            add(&mut turnover[i], output[k] + input[k]);
        }
    }
    let utility = route.objective.grad(price);
    let mut kkt = 0.0_f64;
    let mut deficit = 0.0_f64;
    let mut worst = 0;
    for i in 0..route.tokens {
        assert!(
            price[i].is_finite() && price[i] > 0.0 && price[i] >= lower[i] && price[i] <= upper[i]
        );
        add(&mut net[i], utility[i]);
        let g = net[i].0 + net[i].1;
        let allowed = o.absolute_residual_value
            + o.relative_residual_tolerance * (turnover[i].0 + turnover[i].1) * price[i];
        let projected = if (price[i] == lower[i] && g >= 0.0) || (price[i] == upper[i] && g <= 0.0)
        {
            0.0
        } else {
            g.abs()
        };
        let r = projected * price[i] / allowed;
        if r > kkt {
            kkt = r;
            worst = i;
        }
        deficit = deficit.max((-g * price[i]).max(0.0) / allowed);
    }
    assert!(e.cost.is_finite() && e.gradient.iter().all(|g| g.is_finite()));
    json!({"kkt":kkt,"deficit":deficit,"usable":kkt<=1.0 && deficit<=1.0,"worst_token":worst,"cost":e.cost,"anchor_net_estimate":net[0].0+net[0].1})
}
fn state(
    full: &dexrouter_optim::snapshot::FullMarketRoute,
    label: &str,
    price: &ndarray::Array1<f64>,
    o: &SolveOptions,
) -> Value {
    let e = full.route.evaluate(price);
    let lower = full.route.objective.lower_bounds();
    let upper = full.route.objective.upper_bounds();
    let anchor = full.tokens.iter().position(|t| t == "USDC").unwrap();
    let free: Vec<_> = (0..full.route.tokens)
        .map(|i| {
            i != anchor
                && lower[i] != upper[i]
                && !((price[i] <= lower[i] && e.gradient[i] >= 0.0)
                    || (price[i] >= upper[i] && e.gradient[i] <= 0.0))
        })
        .collect();
    let edges:Vec<_>=full.route.markets.iter().enumerate().map(|(index,(m,(a,b)))|{
        let t=m.evaluate([price[*a],price[*b]]);let weight=m.relative_curvature([price[*a],price[*b]]).unwrap();
        assert!(weight.is_finite() && weight>=0.0);
        json!({"index":index,"a":a,"b":b,"pair":[&full.tokens[*a],&full.tokens[*b]],"weight":weight,
            "support":t.value,"input":t.input,"output":t.output,"parameters":serde_json::to_value(m).unwrap()})
    }).collect();
    json!({"state":label,"tokens":full.tokens,"price":price.to_vec(),"free":free,"anchor":anchor,
        "gradient":e.gradient.to_vec(),"edges":edges,"economic":check(&full.route,price,o)})
}
#[test]
#[ignore = "export exact f64 regional edge weights for high-precision offline spectral analysis"]
fn export_laplacian_inputs() {
    let pools: Vec<RawUniswapV3Pool> =
        serde_json::from_str(include_str!("../fixtures/uniswap_v3_pools.json")).unwrap();
    let mut cases = Vec::new();
    for (label, cutoff, tolerance) in [
        ("filtered", 1000.0, 1e-8),
        ("full", 0.0, 1e-6),
        ("full-strict", 0.0, 1e-8),
    ] {
        let full = build_full_market_route_with_tvl(pools.clone(), cutoff).unwrap();
        let o = options(tolerance);
        let r = solve_price_with_newton(
            full.route.clone(),
            full.initial_price.clone(),
            o.clone(),
            newton_options(&o),
        )
        .unwrap();
        independent(&full.route, &r, &o);
        let initial = state(&full, "initial", &full.initial_price, &o);
        let final_state = state(&full, "final", &r.price, &o);
        println!(
            "spectral-export {label} pools={} tokens={} assessment={:?}",
            full.route.markets.len(),
            full.route.tokens,
            r.assessment
        );
        cases.push(json!({"snapshot":label,"cutoff":cutoff,"tolerance":tolerance,"states":[initial,final_state]}));
    }
    std::fs::create_dir_all("../target").unwrap();
    std::fs::write(
        "../target/laplacian_inputs.json",
        serde_json::to_string_pretty(&cases).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "remove attributed pools and verify against the original complete route"]
fn inspect_pool_pruning() {
    let spectra: Value = serde_json::from_str(
        &std::fs::read_to_string("../target/laplacian_spectrum.json").unwrap(),
    )
    .unwrap();
    let final_spectrum = spectra["results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["snapshot"] == "full" && r["state"] == "final")
        .unwrap();
    let selected: Vec<usize> = final_spectrum["single_pool_deletions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["index"].as_u64().unwrap() as usize)
        .collect();
    let pools: Vec<RawUniswapV3Pool> =
        serde_json::from_str(include_str!("../fixtures/uniswap_v3_pools.json")).unwrap();
    let full = build_full_market_route_with_tvl(pools, 0.0).unwrap();
    let mut output = Vec::new();
    for tolerance in [1e-6, 1e-8] {
        let o = options(tolerance);
        let n = newton_options(&o);
        let base = solve_price_with_newton(
            full.route.clone(),
            full.initial_price.clone(),
            o.clone(),
            n.clone(),
        )
        .unwrap();
        let base_check = check(&full.route, &base.price, &o);
        let mut plans: Vec<(String, Vec<usize>)> = selected
            .iter()
            .map(|&i| (format!("single-{i}"), vec![i]))
            .collect();
        let no_trade = |p: &ndarray::Array1<f64>, i: usize| {
            let (m, (a, b)) = &full.route.markets[i];
            let t = m.evaluate([p[*a], p[*b]]);
            t.input == [0.0; 2] && t.output == [0.0; 2]
        };
        plans.push((
            "zero-curvature-final".into(),
            full.route
                .markets
                .iter()
                .enumerate()
                .filter(|(_, (m, (a, b)))| {
                    m.relative_curvature([base.price[*a], base.price[*b]])
                        .unwrap()
                        == 0.0
                })
                .map(|(i, _)| i)
                .collect(),
        ));
        plans.push((
            "no-trade-initial".into(),
            (0..full.route.markets.len())
                .filter(|&i| no_trade(&full.initial_price, i))
                .collect(),
        ));
        plans.push((
            "no-trade-final".into(),
            (0..full.route.markets.len())
                .filter(|&i| no_trade(&base.price, i))
                .collect(),
        ));
        plans.push((
            "all-wstETH-pools".into(),
            full.route
                .markets
                .iter()
                .enumerate()
                .filter(|(_, (_, (a, b)))| {
                    full.tokens[*a] == "wstETH" || full.tokens[*b] == "wstETH"
                })
                .map(|(i, _)| i)
                .collect(),
        ));
        for (label, omitted) in plans {
            let mut reduced = full.route.clone();
            reduced.markets = full
                .route
                .markets
                .iter()
                .enumerate()
                .filter(|(i, _)| !omitted.contains(i))
                .map(|(_, m)| m.clone())
                .collect();
            for warm in [false, true] {
                let start = Instant::now();
                let r = solve_price_with_newton(
                    reduced.clone(),
                    if warm {
                        base.price.clone()
                    } else {
                        full.initial_price.clone()
                    },
                    o.clone(),
                    n.clone(),
                )
                .unwrap();
                let seconds = start.elapsed().as_secs_f64();
                independent(&reduced, &r, &o);
                let original = check(&full.route, &r.price, &o);
                let pruned = check(&reduced, &r.price, &o);
                let mut support = (0.0, 0.0);
                let mut activated = 0;
                let mut change = (0.0, 0.0);
                let mut retained_change = (0.0, 0.0);
                let mut base_omitted = (0.0, 0.0);
                for (i, (m, (a, b))) in full.route.markets.iter().enumerate() {
                    let trial = m.evaluate([r.price[*a], r.price[*b]]);
                    let reference = m.evaluate([base.price[*a], base.price[*b]]);
                    let delta = m.value_change(
                        [r.price[*a], r.price[*b]],
                        [base.price[*a], base.price[*b]],
                        trial,
                        reference,
                    );
                    add(&mut change, delta);
                    if omitted.contains(&i) {
                        add(&mut base_omitted, reference.value);
                        add(&mut support, trial.value);
                        if trial.input != [0.0; 2] || trial.output != [0.0; 2] {
                            activated += 1;
                        }
                    } else {
                        add(&mut retained_change, delta);
                    }
                }
                let record = json!({"plan":label,"warm":warm,"tolerance":tolerance,"removed":omitted,"removed_count":omitted.len(),
                    "native_status":format!("{:?}",r.status),"assessment":format!("{:?}",r.assessment),"fg":r.evaluations,"hvp":r.hessian_products,"seconds":seconds,
                    "baseline":base_check,"pruned":pruned,"original":original,"retained_dual_change":retained_change.0+retained_change.1,"baseline_omitted_support":base_omitted.0+base_omitted.1,"full_dual_change":change.0+change.1,"omitted_support_estimate":support.0+support.1,"omitted_trading_pools":activated});
                println!("PRUNING_RESULT {record}");
                output.push(record);
            }
        }
    }
    std::fs::write(
        "../target/laplacian_pruning.json",
        serde_json::to_string_pretty(&output).unwrap(),
    )
    .unwrap();
}

#[test]
fn conditioning_cannot_certify_pool_removal_but_a_no_trade_solution_can() {
    use dexrouter_optim::market::UniswapV2;
    let objective = NonnegativeLinear {
        c: ndarray::arr1(&[1.0, 0.0]),
    };
    let o = options(1e-8);
    let initial = ndarray::arr1(&[1.0, 1.0]);
    let full = Route {
        objective: objective.clone(),
        markets: vec![
            (UniswapV2::new(1000.0, 2000.0, 1.0), (0, 1)),
            (UniswapV2::new(2000.0, 1000.0, 1.0), (0, 1)),
        ],
        tokens: 2,
    };
    let reduced = Route {
        objective,
        markets: vec![full.markets[1]],
        tokens: 2,
    };
    let r =
        solve_price_with_newton(reduced, initial.clone(), o.clone(), newton_options(&o)).unwrap();
    assert!(r.is_usable());
    assert_eq!(check(&full, &r.price, &o)["usable"], false);
    assert!(r.cost.abs() < 1e-8);
    let initially_inactive_pool = UniswapV2::new(1000.0, 1000.0, 0.997);
    let zero = initially_inactive_pool.evaluate([initial[0], initial[1]]);
    assert_eq!(zero.input, [0.0; 2]);
    assert_eq!(zero.output, [0.0; 2]);
    let mut needs_reactivation = full.clone();
    needs_reactivation
        .markets
        .push((initially_inactive_pool, (0, 1)));
    let mut extended = full.clone();
    extended
        .markets
        .push((UniswapV2::new(800.0, 900.0, 0.997), (0, 1)));
    let r = solve_price_with_newton(full, initial, o.clone(), newton_options(&o)).unwrap();
    assert!(r.is_usable());
    assert!((r.cost - 1000.0 / 3.0).abs() < 1e-8);
    assert_eq!(check(&needs_reactivation, &r.price, &o)["usable"], false);
    assert_eq!(check(&extended, &r.price, &o)["usable"], true);
    let t = extended.markets[2].0.evaluate([r.price[0], r.price[1]]);
    assert_eq!(t.input, [0.0; 2]);
    assert_eq!(t.output, [0.0; 2]);
    assert_eq!(t.value, 0.0);
}
