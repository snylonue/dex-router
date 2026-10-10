use std::fs;

use alloy::{
    primitives::{Address, address},
    providers::{ProviderBuilder, WsConnect},
};
use dexrouter_arbitrage::{fetch_uniswap_v3_markets, fetch_uniswap_v3_static_data};
use dexrouter_optim::{
    SolveOptions, snapshot::load_full_market_route, solve_price_with_options,
    utility::UtilityConjugate,
};
use ndarray::Array1;

const DEFAULT_POOL_FILE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../dexrouter-optim/tests/fixtures/uniswap_v3_pools.json",
);
fn main() -> anyhow::Result<()> {
    env_logger::init();

    let full_market = load_full_market_route(DEFAULT_POOL_FILE)?;
    println!(
        "loaded pools: {} kept, {} skipped / {} total",
        full_market.stats.kept_pools,
        full_market.stats.skipped_pools(),
        full_market.stats.total_pools
    );
    println!(
        "skips: invalid_price={} invalid_fee={} invalid_shape={} invalid_order={} outside_range={} no_liquidity={} negative_liquidity={} outlier={} low_active_tvl={}",
        full_market.stats.invalid_price,
        full_market.stats.invalid_fee,
        full_market.stats.invalid_shape,
        full_market.stats.invalid_tick_order,
        full_market.stats.current_price_outside_range,
        full_market.stats.no_positive_liquidity,
        full_market.stats.negative_liquidity,
        full_market.stats.price_outlier,
        full_market.stats.low_active_tvl,
    );
    println!(
        "active route: {} tokens, {} pools",
        full_market.tokens.len(),
        full_market.route.markets.len()
    );

    let absolute_residual_value: f64 = std::env::var("DEXROUTER_ABSOLUTE_RESIDUAL_USD")
        .unwrap_or_else(|_| "0.000001".to_owned())
        .parse()?;
    // USDC is the application anchor. This policy belongs here, not in the optimizer.
    let diagnostics = solve_price_with_options(
        full_market.route.clone(),
        full_market.initial_price.clone(),
        SolveOptions {
            absolute_residual_value,
            ..SolveOptions::default()
        },
    )?;
    // The joint gradient uses compensated summation; ordinary matrix summation
    // can lose thousands of cheap tokens against much larger opposing flows.
    let net_flow = &diagnostics.gradient - &full_market.route.objective.grad(&diagnostics.price);
    let max_abs_net_flow = net_flow.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let lower_bounds = full_market.route.objective.lower_bounds();
    let active_lower_bounds = diagnostics
        .price
        .iter()
        .zip(lower_bounds.iter())
        .filter(|(price, lower)| (*price - *lower).abs() <= 1e-8 * lower.abs().max(1.0))
        .count();

    println!(
        "solver: status={:?} assessment={:?} restarts={} kkt_ratio={:.3e} deficit_ratio={:.3e} max_deficit_value={:.3e} iterations={} evals={} cost={:.12e} max_abs_gradient={:.12e} max_projected_gradient={:.12e} max_scaled_projected_gradient={:.12e} max_abs_net_flow={:.12e} active_lower_bounds={}/{}",
        diagnostics.status,
        diagnostics.assessment,
        diagnostics.restarts,
        diagnostics.max_kkt_residual_ratio,
        diagnostics.max_deficit_ratio,
        diagnostics.max_deficit_value,
        diagnostics.iterations,
        diagnostics.evaluations,
        diagnostics.cost,
        diagnostics.max_abs_gradient,
        diagnostics.max_projected_gradient,
        diagnostics.max_scaled_projected_gradient,
        max_abs_net_flow,
        active_lower_bounds,
        full_market.tokens.len()
    );
    anyhow::ensure!(
        diagnostics.is_usable(),
        "solver result rejected: {:?}; KKT ratio={}, deficit ratio={}",
        diagnostics.assessment,
        diagnostics.max_kkt_residual_ratio,
        diagnostics.max_deficit_ratio
    );
    print_anchor_prices(&full_market.tokens, &diagnostics.price);
    print_top_net_flows(&full_market.tokens, &net_flow, 10);

    Ok(())
}

fn print_anchor_prices(tokens: &[String], price: &Array1<f64>) {
    for symbol in ["USDC", "WETH", "USDT", "DAI", "WBTC"] {
        if let Some(index) = tokens.iter().position(|token| token == symbol) {
            println!("price[{symbol}]={:.12e}", price[index]);
        }
    }
}

fn print_top_net_flows(tokens: &[String], net_flow: &Array1<f64>, limit: usize) {
    let mut flows = net_flow.iter().copied().enumerate().collect::<Vec<_>>();
    flows.sort_by(|a, b| a.1.total_cmp(&b.1));

    println!("top negative net flows:");
    for (index, value) in flows.iter().take(limit) {
        println!("  {:>8}: {:.12e}", tokens[*index], value);
    }

    println!("top positive net flows:");
    for (index, value) in flows.iter().rev().take(limit) {
        println!("  {:>8}: {:.12e}", tokens[*index], value);
    }
}

#[allow(dead_code)]
async fn fetch_live_markets_to_json() -> anyhow::Result<()> {
    let rpc_url = "wss://rpc.ankr.com/eth/ws/693e2a5be7ed47deca6e34f79bdfcd49385153c81ff593ab0abc272fcf415548";
    let provider = ProviderBuilder::new()
        .connect_ws(WsConnect::new(rpc_url))
        .await?;

    let pools: Vec<Address> = vec![
        address!("0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"),
        address!("0xc7bBeC68d12a0d1830360F8Ec58fA599bA1b0e9b"),
        address!("0x3416cF6C708Da44DB2624D63ea0AAef7113527C6"),
    ];

    let static_data = fetch_uniswap_v3_static_data(pools, provider.clone()).await?;
    let markets = fetch_uniswap_v3_markets(static_data, provider.clone()).await?;
    let json = serde_json::to_string(&markets)?;
    fs::write("markets.json", &json)?;

    Ok(())
}
