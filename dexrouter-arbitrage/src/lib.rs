use std::{collections::HashMap, str::FromStr};

use alloy::{
    network::Network,
    primitives::{Address, Bytes, address, aliases::I24},
    providers::Provider,
    sol,
};
use alloy_sol_types::SolCall;
use bigdecimal::BigDecimal;
use dexrouter_optim::market::UniswapV3;

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

        function fee() external view returns (uint24);

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

        function tickBitmap(int16 wordPosition) external view returns (uint256);
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

#[derive(Debug, Clone)]
pub struct UniswapV3PoolStaticData {
    pub address: Address,
    /// Uniswap v3 fee tier in hundredths of a bip, e.g. 3000 => 0.3%.
    pub fee: u32,
    /// Full set of initialized ticks for the pool, prefetched out-of-band.
    pub initialized_ticks: Vec<i32>,
}

impl UniswapV3PoolStaticData {
    pub fn new(address: Address, fee: u32, initialized_ticks: Vec<i32>) -> Self {
        Self {
            address,
            fee,
            initialized_ticks,
        }
    }
}

pub async fn fetch_uniswap_v3_static_data<P: Provider<N> + Clone, N: Network>(
    pools: Vec<Address>,
    provider: P,
) -> anyhow::Result<Vec<UniswapV3PoolStaticData>> {
    let mut metadata_calls = Vec::with_capacity(pools.len() * 2);

    for &pool in &pools {
        metadata_calls.push(IMulticall3::Call {
            target: pool,
            callData: IUniswapV3Pool::feeCall {}.abi_encode().into(),
        });
        metadata_calls.push(IMulticall3::Call {
            target: pool,
            callData: IUniswapV3Pool::tickSpacingCall {}.abi_encode().into(),
        });
    }

    let multicall = IMulticall3::new(
        address!("0xcA11bde05977b3631167028862bE2a173976CA11"),
        provider.clone(),
    );
    let metadata = multicall.aggregate(metadata_calls).call().await?;

    let mut pool_metadata = Vec::with_capacity(pools.len());
    let mut bitmap_calls = Vec::new();

    for (&pool, [fee, tick_spacing]) in pools.iter().zip(metadata.returnData.as_chunks().0.iter()) {
        dbg!(pool);
        let fee = IUniswapV3Pool::feeCall::abi_decode_returns(fee)?;
        let tick_spacing = IUniswapV3Pool::tickSpacingCall::abi_decode_returns(tick_spacing)?;
        let tick_spacing = tick_spacing.as_i32();

        anyhow::ensure!(tick_spacing > 0, "invalid tick spacing for pool {pool:#x}");

        let (min_word, max_word) = bitmap_word_range(tick_spacing);
        let mut word_positions = Vec::with_capacity((max_word - min_word + 1) as usize);

        for word_position in min_word..=max_word {
            word_positions.push(word_position);
            bitmap_calls.push(IMulticall3::Call {
                target: pool,
                callData: IUniswapV3Pool::tickBitmapCall {
                    wordPosition: word_position.into(),
                }
                .abi_encode()
                .into(),
            });
        }

        pool_metadata.push((pool, fee, tick_spacing, word_positions));
    }

    let bitmaps = multicall.aggregate(bitmap_calls).call().await?;
    dbg!();
    let mut cursor = 0usize;
    let mut out = Vec::with_capacity(pool_metadata.len());

    for (address, fee, tick_spacing, word_positions) in pool_metadata {
        dbg!(address);
        let mut initialized_ticks = Vec::new();

        for word_position in word_positions {
            let bitmap =
                IUniswapV3Pool::tickBitmapCall::abi_decode_returns(&bitmaps.returnData[cursor])?;
            cursor += 1;
            collect_initialized_ticks(word_position, tick_spacing, bitmap, &mut initialized_ticks);
        }

        out.push(UniswapV3PoolStaticData::new(
            address,
            fee.try_into()?,
            initialized_ticks,
        ));
    }

    Ok(out)
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

pub async fn fetch_uniswap_v3_markets<P: Provider<N>, N: Network>(
    pools: Vec<UniswapV3PoolStaticData>,
    provider: P,
) -> anyhow::Result<Vec<UniswapV3>> {
    let mut calls = Vec::new();

    for pool in &pools {
        let slot0_calldata: Bytes = IUniswapV3Pool::slot0Call {}.abi_encode().into();
        calls.push(IMulticall3::Call {
            target: pool.address,
            callData: slot0_calldata,
        });

        for &tick in &pool.initialized_ticks {
            let tick_calldata: Bytes = IUniswapV3Pool::ticksCall {
                tick: I24::try_from(tick)?,
            }
            .abi_encode()
            .into();
            calls.push(IMulticall3::Call {
                target: pool.address,
                callData: tick_calldata,
            });
        }
    }

    let multicall = IMulticall3::new(
        address!("0xcA11bde05977b3631167028862bE2a173976CA11"),
        provider,
    );
    let response = multicall.aggregate(calls).call().await?;

    let mut cursor = 0usize;
    let mut markets = Vec::with_capacity(pools.len());

    for pool in pools {
        let slot0 = &response.returnData[cursor];
        cursor += 1;

        let slot0Return {
            sqrtPriceX96: sqrt_price_x96,
            ..
        } = IUniswapV3Pool::slot0Call::abi_decode_returns(slot0)?;

        let mut liquidity_net_by_tick = HashMap::with_capacity(pool.initialized_ticks.len());
        for &tick in &pool.initialized_ticks {
            let tick_info =
                IUniswapV3Pool::ticksCall::abi_decode_returns(&response.returnData[cursor])?;
            cursor += 1;
            if !tick_info.initialized {
                anyhow::bail!(
                    "pool {:#x} tick {} is no longer initialized",
                    pool.address,
                    tick
                );
            }
            liquidity_net_by_tick.insert(tick, tick_info.liquidityNet);
        }

        markets.push(build_uniswap_v3_market(
            sqrt_price_x96_to_f64(sqrt_price_x96)?,
            pool.fee,
            &pool.initialized_ticks,
            &liquidity_net_by_tick,
        )?);
    }

    Ok(markets)
}

fn build_uniswap_v3_market(
    current_price: f64,
    fee: u32,
    initialized_ticks: &[i32],
    liquidity_net_by_tick: &HashMap<i32, i128>,
) -> anyhow::Result<UniswapV3> {
    anyhow::ensure!(
        initialized_ticks.len() >= 2,
        "need at least two initialized ticks to construct a UniswapV3 market"
    );

    let mut ticks = initialized_ticks.to_vec();
    ticks.sort_unstable();

    let lower_prices = ticks
        .iter()
        .rev()
        .map(|&tick| sqrt_price_from_tick(tick))
        .collect::<Vec<_>>();

    let highest = lower_prices[0];
    let lowest = *lower_prices.last().unwrap();
    anyhow::ensure!(
        current_price <= highest && current_price >= lowest,
        "current price {current_price} is outside initialized tick range [{lowest}, {highest}]"
    );

    let mut running_liquidity = 0i128;
    let mut interval_liquidity = Vec::with_capacity(ticks.len() - 1);

    for (idx, tick) in ticks.iter().enumerate() {
        let liquidity_net = *liquidity_net_by_tick
            .get(tick)
            .ok_or_else(|| anyhow::anyhow!("missing liquidityNet for initialized tick {}", tick))?;
        running_liquidity += liquidity_net;
        anyhow::ensure!(
            running_liquidity >= 0,
            "reconstructed liquidity became negative at tick {}",
            tick
        );
        if idx + 1 < ticks.len() {
            interval_liquidity.push(running_liquidity as f64);
        }
    }

    let mut liquidity = interval_liquidity.into_iter().rev().collect::<Vec<_>>();
    liquidity.push(0.0);

    Ok(UniswapV3::new(
        current_price,
        lower_prices,
        liquidity,
        fee_factor(fee),
    ))
}

fn fee_factor(fee: u32) -> f64 {
    1.0 - (fee as f64) / 1_000_000.0
}

const MIN_TICK: i32 = -887_272;
const MAX_TICK: i32 = 887_272;

fn bitmap_word_range(tick_spacing: i32) -> (i16, i16) {
    let min_compressed = MIN_TICK.div_euclid(tick_spacing);
    let max_compressed = MAX_TICK.div_euclid(tick_spacing);
    ((min_compressed >> 8) as i16, (max_compressed >> 8) as i16)
}

fn collect_initialized_ticks(
    word_position: i16,
    tick_spacing: i32,
    bitmap: alloy::primitives::U256,
    out: &mut Vec<i32>,
) {
    if bitmap.is_zero() {
        return;
    }

    let base = (word_position as i32) << 8;
    for bit in 0..256 {
        if bitmap.bit(bit) {
            out.push((base + bit as i32) * tick_spacing);
        }
    }
}

fn sqrt_price_x96_to_f64(sqrt_price_x96: alloy::primitives::U160) -> anyhow::Result<f64> {
    let q96 = BigDecimal::from(1_u128 << 96);
    let p = BigDecimal::from(sqrt_price_x96) / q96;
    Ok(p.to_string().parse()?)
}

fn sqrt_price_from_tick(tick: i32) -> f64 {
    1.0001_f64.powf((tick as f64) / 2.0)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use alloy::primitives::U256;
    use ndarray::arr1;

    use super::{build_uniswap_v3_market, collect_initialized_ticks};
    use dexrouter_optim::market::{Market, UniswapV3};

    #[test]
    fn reconstructs_market_from_initialized_ticks() {
        let ticks = vec![100, 200, 300, 400];
        let liquidity_net_by_tick =
            HashMap::from([(100, 5_i128), (200, 3_i128), (300, -4_i128), (400, -4_i128)]);
        let current_price = 1.0001_f64.powf(250.0 / 2.0);

        let market =
            build_uniswap_v3_market(current_price, 3_000, &ticks, &liquidity_net_by_tick).unwrap();

        let expected = UniswapV3::new(
            current_price,
            vec![
                1.0001_f64.powf(400.0 / 2.0),
                1.0001_f64.powf(300.0 / 2.0),
                1.0001_f64.powf(200.0 / 2.0),
                1.0001_f64.powf(100.0 / 2.0),
            ],
            vec![4.0, 8.0, 5.0, 0.0],
            0.997,
        );

        let (actual_in, actual_out) = market.arbitrage([1.0, 0.5]);
        let (expected_in, expected_out) = expected.arbitrage([1.0, 0.5]);

        assert!(arr1(&actual_in).abs_diff_eq(&arr1(&expected_in), 1e-9));
        assert!(arr1(&actual_out).abs_diff_eq(&arr1(&expected_out), 1e-9));
    }

    #[test]
    fn decodes_tick_bitmap() {
        let mut ticks = Vec::new();
        let bitmap = U256::from(1_u64 << 1) | U256::from(1_u64 << 7);

        collect_initialized_ticks(0, 10, bitmap, &mut ticks);

        assert_eq!(ticks, vec![10, 70]);
    }
}
