#![feature(f128)]

use std::str::FromStr;

use alloy::{
    network::Network,
    primitives::address,
    providers::{Provider, ProviderBuilder, WsConnect},
    sol_types::sol,
};
use bigdecimal::{BigDecimal, ToPrimitive};

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
}

#[derive(Debug, Default)]
struct Data {
    pub sqrt_pl: BigDecimal,
    pub sqrt_pu: BigDecimal,
    pub sqrt_p: BigDecimal,
    pub reserve0: BigDecimal,
    pub reserve1: BigDecimal,
}

async fn get_current_data<P: Provider<N>, N: Network>(
    pool: IUniswapV3PoolInstance<P, N>,
    provider: P,
) -> anyhow::Result<Data> {
    let (
        slot0Return {
            sqrtPriceX96: sqrt_price_x96,
            tick,
            ..
        },
        liquidity,
        tick_spacing,
    ) = provider
        .multicall()
        .add(pool.slot0())
        .add(pool.liquidity())
        .add(pool.tickSpacing())
        .aggregate()
        .await?;
    let tl = tick.div_euclid(tick_spacing) * tick_spacing;
    let tu = tl + tick_spacing;

    let base = BigDecimal::from_str("1.0001")?;
    let pl = base.powi(tl.as_i64()).sqrt().unwrap();
    let pu = base.powi(tu.as_i64()).sqrt().unwrap();
    let p = BigDecimal::from(sqrt_price_x96) / BigDecimal::from(1_u128 << 96);

    let reserve0 = liquidity * (1.0 / &p - 1.0 / &pu);
    let reserve1 = liquidity * (&p - &pl);

    Ok(Data {
        sqrt_pl: pl,
        sqrt_pu: pu,
        sqrt_p: p,
        reserve0,
        reserve1,
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let rpc_url = "wss://ethereum-rpc.publicnode.com";
    let provider = ProviderBuilder::new()
        .connect_ws(WsConnect::new(rpc_url))
        .await?;

    let latest_block = provider.get_block_number().await?;
    println!("Latest block number: {latest_block}");

    // Get chain ID.
    let chain_id = provider.get_chain_id().await?;
    println!("Chain ID: {chain_id}");

    let pool = IUniswapV3Pool::new(
        address!("0xdc212B831b9C47f413218355BfFC73830E741446"),
        provider.clone(),
    );

    let Data {
        sqrt_pl,
        sqrt_pu,
        sqrt_p,
        reserve0,
        reserve1,
    } = get_current_data(pool, provider.clone()).await?;

    println!("Price (WETH/USDT): {}", sqrt_p.square().to_f64().unwrap());
    println!(
        "Price Range {} - {}",
        sqrt_pl.square().to_f64().unwrap(),
        sqrt_pu.square().to_f64().unwrap()
    );
    println!("Active WETH: {}", reserve0.to_f64().unwrap() / 1e6_f64);
    println!("Active USDT: {}", reserve1.to_f64().unwrap() / 1e6_f64);

    Ok(())
}
