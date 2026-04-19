// #[tokio::main]
// async fn _main() -> anyhow::Result<()> {
//     let rpc_url = "wss://ethereum-rpc.publicnode.com";
//     let provider = ProviderBuilder::new()
//         .connect_ws(WsConnect::new(rpc_url))
//         .await?;

//     let latest_block = provider.get_block_number().await?;
//     println!("Latest block number: {latest_block}");

//     // Get chain ID.
//     let chain_id = provider.get_chain_id().await?;
//     println!("Chain ID: {chain_id}");

//     let pool = IUniswapV3Pool::new(
//         address!("0xdc212B831b9C47f413218355BfFC73830E741446"),
//         provider.clone(),
//     );

//     let _Data {
//         sqrt_pl,
//         sqrt_pu,
//         sqrt_p,
//         reserve0,
//         reserve1,
//     } = _get_current_data(pool, provider.clone()).await?;

//     println!("Price (WETH/USDT): {}", sqrt_p.square().to_f64().unwrap());
//     println!(
//         "Price Range {} - {}",
//         sqrt_pl.square().to_f64().unwrap(),
//         sqrt_pu.square().to_f64().unwrap()
//     );
//     println!("Active WETH: {}", reserve0.to_f64().unwrap() / 1e6_f64);
//     println!("Active USDT: {}", reserve1.to_f64().unwrap() / 1e6_f64);

//     Ok(())
// }

use alloy::{
    primitives::address,
    providers::{ProviderBuilder, WsConnect},
};
use dexrouter_arbitrage::{IUniswapV3Pool::IUniswapV3PoolInstance, fetch_pools};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let rpc_url = "wss://ethereum-rpc.publicnode.com";
    let provider = ProviderBuilder::new()
        .connect_ws(WsConnect::new(rpc_url))
        .await?;
    let pools = vec![
        IUniswapV3PoolInstance::new(
            address!("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"),
            provider.clone(),
        ),
        IUniswapV3PoolInstance::new(
            address!("0xc7bBeC68d12a0d1830360F8Ec58fA599bA1b0e9b"),
            provider.clone(),
        ),
    ];

    let pools = fetch_pools(pools, provider.clone()).await?;

    println!("{:#?}", pools);

    Ok(())
}
