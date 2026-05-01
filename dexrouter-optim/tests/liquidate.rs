use dexrouter_optim::{
    Route,
    market::{UniswapV2, UniswapV3},
    solve_price,
    utility::BasketLiquidation,
};
use ndarray::{Axis, arr1, arr2};

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

    let p = solve_price(route.clone());

    let (inputs, outputs) = route.arbitrage(p);

    println!("{inputs}\n{outputs}");
    println!("{}", (&outputs - &inputs).sum_axis(Axis(0)));

    assert!(inputs.abs_diff_eq(
        &arr2(&[
            [0.0, 765.668523465685, 0.0],
            [0.0, 0.0, 310.2106850987172],
            [10.654141916111685, 0.0, 0.0]
        ]),
        1e-5
    ));

    assert!(outputs.abs_diff_eq(
        &arr2(&[
            [70.92308545013987, 0.0, 0.0],
            [0.0, 755.6685226743975, 0.0],
            [0.0, 0.0, 210.21069408391486]
        ]),
        1e-5
    ));
}

#[test]
fn liquidate_eth() -> anyhow::Result<()> {
    let markets: Vec<UniswapV3> = serde_json::from_str(include_str!("./markets.json"))?;

    let route = Route {
        objective: BasketLiquidation {
            out: 0,
            inputs: arr1(&[0.0, 1908.74, 2754.99]),
        },
        markets: vec![
            (markets[0].clone().scaled(1e-3, 1e-9), (1, 0)),
            (markets[1].clone().scaled(1e-9, 1e-3), (0, 2)),
        ],
        tokens: 3,
    };

    let p = solve_price(route.clone());

    println!("{p}");
    println!("{}", &p / p[1]);

    let (inputs, outputs) = route.arbitrage(p);

    println!("{inputs}\n{outputs}");
    let net = (&outputs - &inputs).sum_axis(Axis(0));
    println!("{} {} {}", net[0], net[1], net[2]);

    let buy = net[0];
    println!("{buy}");
    assert!(buy > 1.0 && buy < 2.0);

    Ok(())
}

#[test]
fn swap() -> anyhow::Result<()> {
    let markets: Vec<UniswapV3> = serde_json::from_str(include_str!("./markets.json"))?;
    let route = Route {
        objective: BasketLiquidation {
            out: 0,
            inputs: arr1(&[0.0, 1e6]),
        },
        markets: vec![(markets[2].clone().scaled(1e-3, 1e-3), (0, 1))],
        tokens: 2,
    };

    let p = solve_price(route.clone());

    let (inputs, outputs) = route.arbitrage(p);

    println!("{inputs}\n{outputs}");
    let net_flow = (&outputs - &inputs).sum_axis(Axis(0));
    println!("{net_flow}");

    assert!((net_flow[0] / -net_flow[1] - 1.0).abs() <= 1e-1);

    Ok(())
}
