use std::str::FromStr;

use alloy::{
    network::Network,
    primitives::{Bytes, address},
    providers::Provider,
    sol,
};
use alloy_sol_types::SolCall;
use bigdecimal::BigDecimal;

use crate::IUniswapV3Pool::{IUniswapV3PoolInstance, slot0Return};

sol! {
    #[sol(rpc)]
    interface IUniswapV3Pool {
        function slot0()
        external
        view
        returns (
            uint160 sqrtPriceX96,
            int24 tick,
            uint16 observationIndex,
            uint16 observationCardinality,
            uint16 observationCardinalityNext,
            uint8 feeProtocol,
            bool unlocked
        );

        function liquidity() external view returns (uint128);

        function ticks(int24 tick)
        external
        view
        returns (
            uint128 liquidityGross,
            int128 liquidityNet,
            uint256 feeGrowthOutside0X128,
            uint256 feeGrowthOutside1X128,
            int56 tickCumulativeOutside,
            uint160 secondsPerLiquidityOutsideX128,
            uint32 secondsOutside,
            bool initialized
        );

        function tickSpacing() external view returns (int24);
    }

    #[sol(rpc)]
    interface IMulticall3 {
        struct Call {
            address target;
            bytes callData;
        }

        function aggregate(Call[] calldata calls) external payable returns (uint256 blockNumber, bytes[] memory returnData);
    }
}

#[derive(Debug, Default)]
pub struct PoolData {
    pub sqrt_pl: BigDecimal,
    pub sqrt_pu: BigDecimal,
    pub sqrt_p: BigDecimal,
    pub reserve0: BigDecimal,
    pub reserve1: BigDecimal,
}

pub async fn fetch_pools<P: Provider<N>, N: Network>(
    pool: Vec<IUniswapV3PoolInstance<P, N>>,
    provider: P,
) -> anyhow::Result<Vec<PoolData>> {
    let mut calls = Vec::with_capacity(pool.len() * 3);

    calls.extend(
        pool.iter()
            .map(|p| {
                let slot0_calldata: Bytes = IUniswapV3Pool::slot0Call {}.abi_encode().into();
                let liquidity_calldata: Bytes =
                    IUniswapV3Pool::liquidityCall {}.abi_encode().into();
                let tick_spacing_calldata: Bytes =
                    IUniswapV3Pool::tickSpacingCall {}.abi_encode().into();

                let target = *p.address();

                [
                    IMulticall3::Call {
                        target,
                        callData: slot0_calldata,
                    },
                    IMulticall3::Call {
                        target,
                        callData: liquidity_calldata,
                    },
                    IMulticall3::Call {
                        target,
                        callData: tick_spacing_calldata,
                    },
                ]
            })
            .flatten(),
    );

    let multicall = IMulticall3::new(
        address!("0xcA11bde05977b3631167028862bE2a173976CA11"),
        provider,
    );

    // aggregate3 执行必定都在同一个区块，同一个交易执行上下文中完成
    let r = multicall.aggregate(calls).call().await?;

    let base = BigDecimal::from_str("1.0001")?;

    r.returnData
        .as_chunks()
        .0
        .into_iter()
        .map(
            |[slot0, liquidity, tick_spacing]| -> Result<_, anyhow::Error> {
                let slot0Return {
                    sqrtPriceX96: sqrt_price_x96,
                    tick,
                    ..
                } = IUniswapV3Pool::slot0Call::abi_decode_returns(slot0)?;
                let liquidity = IUniswapV3Pool::liquidityCall::abi_decode_returns(liquidity)?;
                let tick_spacing =
                    IUniswapV3Pool::tickSpacingCall::abi_decode_returns(tick_spacing)?;

                let tl = tick.div_euclid(tick_spacing) * tick_spacing;
                let tu = tl + tick_spacing;

                let pl = base.powi(tl.as_i64()).sqrt().unwrap();
                let pu = base.powi(tu.as_i64()).sqrt().unwrap();
                let p = BigDecimal::from(sqrt_price_x96) / BigDecimal::from(1_u128 << 96);

                let reserve0 = liquidity * (1.0 / &p - 1.0 / &pu);
                let reserve1 = liquidity * (&p - &pl);

                Ok(PoolData {
                    sqrt_pl: pl,
                    sqrt_pu: pu,
                    sqrt_p: p,
                    reserve0,
                    reserve1,
                })
            },
        )
        .collect()
}
