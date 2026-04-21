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

use std::fs;

use alloy::{
    primitives::{Address, address},
    providers::{Provider, ProviderBuilder, WsConnect},
};
use dexrouter_arbitrage::{fetch_uniswap_v3_markets, fetch_uniswap_v3_static_data};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let rpc_url = "";
    let provider = ProviderBuilder::new()
        // .connect("http://127.0.0.1:8545").await
        // .connect_anvil_with_wallet_and_config(|a| a.fork(rpc_url))
        .connect_ws(WsConnect::new(rpc_url)).await
        ?;

    dbg!(provider.get_block_number().await?);

    let pools: Vec<Address> = vec![
        address!("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"),
        address!("0xc7bBeC68d12a0d1830360F8Ec58fA599bA1b0e9b"),
    ];

    let static_data = fetch_uniswap_v3_static_data(pools, provider.clone()).await?;
    dbg!();
    let markets = fetch_uniswap_v3_markets(static_data, provider.clone()).await?;

    println!("{:#?}", markets);

    let json = serde_json::to_string(&markets)?;
    fs::write("markets.json", &json)?;

    Ok(())
}
