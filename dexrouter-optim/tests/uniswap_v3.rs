mod uniswap {
    use std::sync::LazyLock;

    use dexrouter_optim::{
        Route,
        market::{Market, UniswapV3},
        solve_price,
        utility::NonnegativeLinear,
    };
    use ndarray::{Array1, Axis, arr1};

    static MARKET: LazyLock<UniswapV3> = LazyLock::new(|| {
        UniswapV3::new(
            3.872983346207417,
            vec![
                5.477225575051661,
                4.47213595499958,
                3.1622776601683795,
                2.23606797749979,
            ],
            vec![1.0, 1.4142135623730951, 1.224744871391589, 0.0],
            0.997,
        )
    });

    #[test]
    fn uniswap_v3_1() {
        let (inputs, outputs) = MARKET.arbitrage([10.0, 1.0]);

        assert!(arr1(&inputs).abs_diff_eq(&arr1(&[0.08163881601325118, 0.0]), 1e-8));
        assert!(arr1(&outputs).abs_diff_eq(&arr1(&[0.0, 0.9983662848277683]), 1e-8));
    }

    #[test]
    fn uniswap_v3_2() {
        let (inputs, outputs) = MARKET.arbitrage([25.0, 1.0]);

        assert!(arr1(&inputs).abs_diff_eq(&arr1(&[0.0, 1.3718035675347677]), 1e-8));
        assert!(arr1(&outputs).abs_diff_eq(&arr1(&[0.07222672671131006, 0.0]), 1e-8));
    }

    #[test]
    fn uniswap_v3_3() {
        let (inputs, outputs) = MARKET.arbitrage([15.0, 1.0]);

        assert!(arr1(&inputs).abs_diff_eq(&arr1(&[0.0, 0.0]), 1e-8));
        assert!(arr1(&outputs).abs_diff_eq(&arr1(&[0.0, 0.0]), 1e-8));
    }

    #[test]
    fn uniswap_onchain() {
        let markets: Vec<UniswapV3> = serde_json::from_str(include_str!("./markets.json")).unwrap();
        let (inputs, outputs) = markets[2].clone().scaled(1e-3, 1e-3).arbitrage([1.0, 1.0]);
        dbg!(inputs, outputs);

        let route = Route {
            objective: NonnegativeLinear {
                c: Array1::from_elem([3], 1.0),
            },
            markets: vec![
                (markets[0].clone().scaled(1e-3, 1e-9), (1, 0)),
                (markets[1].clone().scaled(1e-9, 1e-3), (0, 2)),
                (markets[2].clone().scaled(1e-3, 1e-3), (1, 2)),
            ],
            tokens: 3,
        };

        let p = solve_price(route.clone());
        println!("{p}");
        println!("{}", &p / p[1]);

        let (inputs, outputs) = route.arbitrage(p);

        println!("{inputs}\n{outputs}");
        let net = (&outputs - &inputs).sum_axis(Axis(0));
        println!("{net}");

        assert!(net.iter().all(|&v| v == 0.0));
    }
}
