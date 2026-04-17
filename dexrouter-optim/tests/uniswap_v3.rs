mod uniswap {
    use std::sync::LazyLock;

    use dexrouter_optim::market::{Market, UniswapV3};
    use ndarray::arr1;

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

        assert!(arr1(&inputs).abs_diff_eq(&arr1(&[0.08163881601325118, 0.0]), 1e-4));
        assert!(arr1(&outputs).abs_diff_eq(&arr1(&[0.0, 0.9983662848277683]), 1e-4));
    }

    #[test]
    fn uniswap_v3_2() {
        let (inputs, outputs) = MARKET.arbitrage([25.0, 1.0]);

        assert!(arr1(&inputs).abs_diff_eq(&arr1(&[0.0, 1.3718035675347677]), 1e-4));
        assert!(arr1(&outputs).abs_diff_eq(&arr1(&[0.07222672671131006, 0.0]), 1e-4));
    }

    #[test]
    fn uniswap_v3_3() {
        let (inputs, outputs) = MARKET.arbitrage([15.0, 1.0]);

        assert!(arr1(&inputs).abs_diff_eq(&arr1(&[0.0, 0.0]), 1e-4));
        assert!(arr1(&outputs).abs_diff_eq(&arr1(&[0.0, 0.0]), 1e-4));
    }
}
