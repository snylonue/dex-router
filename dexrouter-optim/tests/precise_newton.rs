use dexrouter_optim::{
    Route, SolveOptions,
    market::{PreciseMarket, UniswapV2, UniswapV3},
    precision::Price,
    solve_price_with_precise_newton,
    utility::{BasketLiquidation, NonnegativeLinear, PreciseUtility},
};
use ndarray::arr1;

#[test]
fn sub_ulp_prices_reach_v2_v3_quotes_and_cross_region_differences() {
    for market in [0, 1] {
        let v = [Price::from_parts(1.0, 1e-20), 1.0.into()];
        let r = [1.0.into(); 2];
        let v3 = UniswapV3::new(1.0, vec![2.0, 0.5], vec![1e20], 1.0);
        let v2 = UniswapV2::new(1e20, 1e20, 1.0);
        let m: &dyn PreciseMarket = if market == 0 { &v3 } else { &v2 };
        let t = m.evaluate_precise(v);
        let re = m.evaluate_precise(r);
        assert!((t.input[1] - 0.5).abs() < 1e-14);
        assert!((t.output[0] - 0.5).abs() < 1e-14);
        assert_eq!(m.evaluate([1.0, 1.0]).value, 0.0);
        let d = m.value_change_precise(v, r, t, re);
        assert!(d > 0.0);
        assert!((d / t.value - 1.0).abs() < 1e-14);
    }
    let m = UniswapV3::new(1.0, vec![2.0, 0.5], vec![1e20], 0.997);
    let r = [0.997.into(), 1.0.into()];
    let v = [Price::from_parts(0.997, -1e-20), 1.0.into()];
    let t = m.evaluate_precise(v);
    let re = m.evaluate_precise(r);
    assert_eq!(re.value, 0.0);
    assert!(t.input[0] > 0.0);
    assert!((m.value_change_precise(v, r, t, re) / t.value - 1.0).abs() < 1e-14);
}
#[test]
fn sub_ulp_utility_bound_violations_are_not_hidden_by_rounded_hi() {
    let p = [Price::from_parts(1.0, -1e-20), 2.0.into()];
    let n = NonnegativeLinear {
        c: arr1(&[1.0, 0.0]),
    };
    let b = BasketLiquidation {
        out: 0,
        inputs: arr1(&[0.0, 2.0]),
    };
    assert!(n.precise_value(&p).is_infinite());
    assert!(b.precise_value(&p).is_infinite());
}
fn stiff_leaf() -> (Route<NonnegativeLinear, UniswapV3>, ndarray::Array1<f64>) {
    let route = Route {
        objective: NonnegativeLinear {
            c: arr1(&[1.0, 0.0]),
        },
        markets: vec![(UniswapV3::new(1.1, vec![2.0, 0.5], vec![1e16], 1.0), (0, 1))],
        tokens: 2,
    };
    (route, arr1(&[1.0, 1.0 / (1.1 * 1.1)]))
}
#[test]
fn precise_newton_clears_a_nonrepresentable_stiff_price_without_uphill_steps() {
    let (route, p) = stiff_leaf();
    let o = SolveOptions {
        newton_schur_precondition: true,
        ..Default::default()
    };
    let n = dexrouter_newton::Options {
        max_evaluations: 10_000,
        ..Default::default()
    };
    let r = solve_price_with_precise_newton(&route, p, o, n.clone()).unwrap();
    assert!(r.is_usable(), "{r:?}");
    assert!(!r.warm_start.is_usable());
    assert!(r.price.iter().any(|p| p.lo() != 0.0));
    assert!(r.refinement_objective_change < 0.0);
    assert!(r.evaluations <= n.max_evaluations);
    assert!(r.iterations <= n.max_iterations);
    assert!(r.hessian_products <= n.max_hessian_products);
    let cp = Price::from(1.1);
    assert!((r.price[1] * cp * cp - r.price[0]).rounded().abs() < 1e-28);
    let check = route.evaluate_precise(&r.price);
    assert_eq!(check.gradient, r.evaluation.gradient);
    assert!(check.gradient[1].abs() < 1e-8);
    let (input, output) = route.arbitrage_precise(&r.price);
    for i in 0..route.tokens {
        assert!(
            ((output.column(i).sum() - input.column(i).sum()) - check.gradient[i]).abs() < 1e-20
        );
    }
    // A rounded return would invalidate the economic result; keep the distinct API.
    let rounded = route.evaluate(&r.price.iter().map(|p| p.rounded()).collect());
    assert!(rounded.gradient[1].abs() > 1e-8);
}
#[test]
fn precise_warm_and_refinement_share_hard_work_limits() {
    let (route, p) = stiff_leaf();
    for budget in [2, 5, 20] {
        let n = dexrouter_newton::Options {
            max_evaluations: budget,
            max_hessian_products: 1,
            max_iterations: 2,
            ..Default::default()
        };
        let r =
            solve_price_with_precise_newton(&route, p.clone(), SolveOptions::default(), n.clone())
                .unwrap();
        assert!(r.evaluations <= budget);
        assert!(r.hessian_products <= 1);
        assert!(r.iterations <= 2);
        if budget == 2 {
            assert!(!r.is_usable());
        }
        if r.is_usable() {
            assert!(r.max_kkt_residual_ratio <= 1.0 && r.max_deficit_ratio <= 1.0);
            assert!(r.refinement_objective_change <= 0.0);
        }
    }
}
