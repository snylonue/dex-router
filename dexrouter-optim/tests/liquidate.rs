use dexrouter_optim::{
    Route,
    market::UniswapV2,
    solve_price,
    utility::BasketLiquidation,
};
use ndarray::{Array1, arr1, arr2};

#[test]
fn liquidate() {
    let route = Route {
        objective: BasketLiquidation {
            out: 0,
            inputs: arr1(&[0.0, 1e1, 1e2]),
        },
        markets: vec![
            (UniswapV2::new(1e3, 1e4, 0.997), (0, 1)),
            (UniswapV2::new(1e3, 1e2, 0.997), (1, 2)),
            (UniswapV2::new(1e3, 2e4, 0.997), (0, 2)),
        ],
        tokens: 3,
    };

    let p = solve_price(route.clone(), Array1::ones([3]) / 3.0);

    let (inputs, outpus) = route.arbitrage(p);

    assert!(inputs.abs_diff_eq(
        &arr2(&[
            [0.0, 765.668523465685],
            [0.0, 310.2106850987172],
            [10.654141916111685, 0.0]
        ]),
        1e-4
    ));

    assert!(outpus.abs_diff_eq(
        &arr2(&[
            [70.92308545013987, 0.0],
            [755.6685226743975, 0.0],
            [0.0, 210.21069408391486]
        ]),
        1e-4
    ));
}
