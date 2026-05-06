pub mod uniswap;

pub use uniswap::v2::UniswapV2;
pub use uniswap::v3::UniswapV3;

pub trait Market {
    fn arbitrage(&self, v: [f64; 2]) -> ([f64; 2], [f64; 2]);
}

#[cfg(test)]
mod tests {
    use super::{Market, UniswapV2, UniswapV3};

    #[test]
    fn mixed_v2_v3_pools_can_create_complementary_arbitrage_flows() {
        let valuation = [1.0, 1.0];

        // This v2 pool values token0 cheaply relative to the external valuation,
        // so the optimal trade sells token0 into the pool and receives token1.
        let v2 = UniswapV2::new(5.0, 10.0, 0.997);
        let (v2_input, v2_output) = v2.arbitrage(valuation);
        assert!(v2_input[0] > 0.0);
        assert_eq!(v2_input[1], 0.0);
        assert_eq!(v2_output[0], 0.0);
        assert!(v2_output[1] > 0.0);

        // This single-range v3 pool values token0 expensively relative to the
        // same external valuation, so the optimal trade goes in the opposite
        // direction and receives token0 for token1 input.
        let v3 = UniswapV3::new(0.75, vec![1.0, 0.5], vec![10.0], 0.997);
        let (v3_input, v3_output) = v3.arbitrage(valuation);
        assert_eq!(v3_input[0], 0.0);
        assert!(v3_input[1] > 0.0);
        assert!(v3_output[0] > 0.0);
        assert_eq!(v3_output[1], 0.0);

        let net = [
            v2_output[0] + v3_output[0] - v2_input[0] - v3_input[0],
            v2_output[1] + v3_output[1] - v2_input[1] - v3_input[1],
        ];

        assert!(net[0] > 0.0, "expected net token0 gain, got {net:?}");
        assert!(net[1] > 0.0, "expected net token1 gain, got {net:?}");
    }
}
