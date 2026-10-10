use std::{
    collections::{HashMap, VecDeque},
    fs,
    path::Path,
};

use alloy::{
    primitives::{Address, address},
    providers::{ProviderBuilder, WsConnect},
};
use dexrouter_arbitrage::{fetch_uniswap_v3_markets, fetch_uniswap_v3_static_data};
use dexrouter_optim::{
    Route, SolveOptions,
    market::UniswapV3,
    solve_price_with_options,
    utility::{NonnegativeLinear, UtilityConjugate},
};
use ndarray::Array1;
use serde::Deserialize;

const DEFAULT_POOL_FILE: &str = "./uniswap_v3_pools.json";
const ANCHOR_TOKEN: &str = "USDC";
const OUTLIER_LOG_TOLERANCE: f64 = 4.0;
const MIN_LIQUIDITY: f64 = 1e-12;
const DEFAULT_MIN_ACTIVE_TVL_USD: f64 = 1_000.0;

#[derive(Debug, Clone, Deserialize)]
struct RawUniswapV3Pool {
    token0_symbol: String,
    token1_symbol: String,
    current_price: f64,
    lower_prices: Vec<f64>,
    liquidity: Vec<f64>,
    fee: f64,
}

#[derive(Debug, Clone)]
struct AcceptedPool {
    market: UniswapV3,
    token0_symbol: String,
    token1_symbol: String,
    log_price: f64,
    current_price: f64,
    lower_prices: Vec<f64>,
    liquidity: Vec<f64>,
}

#[derive(Debug, Clone)]
struct PriceEdge {
    token0: usize,
    token1: usize,
    log_price: f64,
}

#[derive(Debug, Default, Clone)]
struct LoadStats {
    total_pools: usize,
    kept_pools: usize,
    invalid_price: usize,
    invalid_fee: usize,
    invalid_shape: usize,
    invalid_tick_order: usize,
    current_price_outside_range: usize,
    no_positive_liquidity: usize,
    negative_liquidity: usize,
    price_outlier: usize,
    low_active_tvl: usize,
}

impl LoadStats {
    fn skipped_pools(&self) -> usize {
        self.invalid_price
            + self.invalid_fee
            + self.invalid_shape
            + self.invalid_tick_order
            + self.current_price_outside_range
            + self.no_positive_liquidity
            + self.negative_liquidity
            + self.price_outlier
            + self.low_active_tvl
    }

    fn record_skip(&mut self, reason: SkipReason) {
        match reason {
            SkipReason::InvalidPrice => self.invalid_price += 1,
            SkipReason::InvalidFee => self.invalid_fee += 1,
            SkipReason::InvalidShape => self.invalid_shape += 1,
            SkipReason::InvalidTickOrder => self.invalid_tick_order += 1,
            SkipReason::CurrentPriceOutsideRange => self.current_price_outside_range += 1,
            SkipReason::NoPositiveLiquidity => self.no_positive_liquidity += 1,
            SkipReason::NegativeLiquidity => self.negative_liquidity += 1,
            SkipReason::PriceOutlier => self.price_outlier += 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SkipReason {
    InvalidPrice,
    InvalidFee,
    InvalidShape,
    InvalidTickOrder,
    CurrentPriceOutsideRange,
    NoPositiveLiquidity,
    NegativeLiquidity,
    PriceOutlier,
}

#[derive(Debug, Clone)]
struct FullMarketRoute {
    tokens: Vec<String>,
    route: Route<NonnegativeLinear, UniswapV3>,
    initial_price: Array1<f64>,
    stats: LoadStats,
}

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

fn load_full_market_route(path: impl AsRef<Path>) -> anyhow::Result<FullMarketRoute> {
    let pools: Vec<RawUniswapV3Pool> = serde_json::from_reader(fs::File::open(path)?)?;
    build_full_market_route(pools)
}

fn build_full_market_route(pools: Vec<RawUniswapV3Pool>) -> anyhow::Result<FullMarketRoute> {
    build_full_market_route_with_tvl(pools, min_active_tvl_usd())
}

fn build_full_market_route_with_tvl(
    pools: Vec<RawUniswapV3Pool>,
    min_active_tvl_usd: f64,
) -> anyhow::Result<FullMarketRoute> {
    let pair_medians = pair_median_log_prices(&pools);
    let mut stats = LoadStats {
        total_pools: pools.len(),
        ..LoadStats::default()
    };
    let mut accepted = Vec::new();

    for pool in pools {
        if !valid_sqrt_price(pool.current_price) {
            stats.record_skip(SkipReason::InvalidPrice);
            continue;
        }
        if is_price_outlier(&pool, &pair_medians) {
            stats.record_skip(SkipReason::PriceOutlier);
            continue;
        }

        match convert_pool(&pool) {
            Ok(market) => accepted.push(AcceptedPool {
                market,
                token0_symbol: pool.token0_symbol,
                token1_symbol: pool.token1_symbol,
                log_price: sqrt_price_to_log_price(pool.current_price),
                current_price: pool.current_price,
                lower_prices: pool.lower_prices,
                liquidity: pool.liquidity,
            }),
            Err(reason) => stats.record_skip(reason),
        }
    }

    anyhow::ensure!(!accepted.is_empty(), "no usable pools after filtering");

    let (tokens, token_index, _, edges) = indexed_route_parts(&accepted);
    let anchor = *token_index
        .get(ANCHOR_TOKEN)
        .ok_or_else(|| anyhow::anyhow!("anchor token {ANCHOR_TOKEN} is absent after filtering"))?;
    let initial_price = initial_prices(tokens.len(), anchor, &edges);

    let before_tvl_filter = accepted.len();
    accepted.retain(|pool| {
        let token0 = token_index[&pool.token0_symbol];
        let token1 = token_index[&pool.token1_symbol];
        active_pool_value(pool, initial_price[token0], initial_price[token1])
            .is_some_and(|value| value >= min_active_tvl_usd)
    });
    stats.low_active_tvl = before_tvl_filter - accepted.len();
    anyhow::ensure!(!accepted.is_empty(), "no pools above active TVL filter");

    let (tokens, token_index, markets, edges) = indexed_route_parts(&accepted);
    let anchor = *token_index.get(ANCHOR_TOKEN).ok_or_else(|| {
        anyhow::anyhow!("anchor token {ANCHOR_TOKEN} is absent after TVL filtering")
    })?;
    let initial_price = initial_prices(tokens.len(), anchor, &edges);
    let mut c = Array1::zeros(tokens.len());
    c[anchor] = 1.0;

    stats.kept_pools = markets.len();
    Ok(FullMarketRoute {
        tokens,
        route: Route {
            objective: NonnegativeLinear { c },
            markets,
            tokens: initial_price.len(),
        },
        initial_price,
        stats,
    })
}

fn min_active_tvl_usd() -> f64 {
    std::env::var("DEXROUTER_MIN_ACTIVE_TVL_USD")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(DEFAULT_MIN_ACTIVE_TVL_USD)
}

fn indexed_route_parts(
    accepted: &[AcceptedPool],
) -> (
    Vec<String>,
    HashMap<String, usize>,
    Vec<(UniswapV3, (usize, usize))>,
    Vec<PriceEdge>,
) {
    let mut token_index = HashMap::<String, usize>::new();
    let mut tokens = Vec::<String>::new();
    let mut markets = Vec::with_capacity(accepted.len());
    let mut edges = Vec::with_capacity(accepted.len());

    for pool in accepted {
        let token0 = token_id(&pool.token0_symbol, &mut tokens, &mut token_index);
        let token1 = token_id(&pool.token1_symbol, &mut tokens, &mut token_index);
        edges.push(PriceEdge {
            token0,
            token1,
            log_price: pool.log_price,
        });
        markets.push((pool.market.clone(), (token0, token1)));
    }

    (tokens, token_index, markets, edges)
}

fn active_pool_value(pool: &AcceptedPool, token0_price: f64, token1_price: f64) -> Option<f64> {
    let tick = pool
        .lower_prices
        .partition_point(|price| *price <= pool.current_price)
        .checked_sub(1)?;
    if tick + 1 >= pool.lower_prices.len() {
        return None;
    }

    let lower = pool.lower_prices[tick];
    let upper = pool.lower_prices[tick + 1];
    let liquidity = pool.liquidity[tick].max(0.0);
    let reserve0 = liquidity * (1.0 / pool.current_price - 1.0 / upper).max(0.0);
    let reserve1 = liquidity * (pool.current_price - lower).max(0.0);
    let value = reserve0 * token0_price + reserve1 * token1_price;

    value.is_finite().then_some(value)
}

fn token_id(token: &str, tokens: &mut Vec<String>, index: &mut HashMap<String, usize>) -> usize {
    if let Some(&id) = index.get(token) {
        id
    } else {
        let id = tokens.len();
        tokens.push(token.to_owned());
        index.insert(token.to_owned(), id);
        id
    }
}

fn convert_pool(pool: &RawUniswapV3Pool) -> Result<UniswapV3, SkipReason> {
    if !valid_sqrt_price(pool.current_price) {
        return Err(SkipReason::InvalidPrice);
    }
    if !pool.fee.is_finite() || pool.fee <= 0.0 || pool.fee > 1.0 {
        return Err(SkipReason::InvalidFee);
    }
    if pool.lower_prices.len() < 2 || pool.lower_prices.len() != pool.liquidity.len() {
        return Err(SkipReason::InvalidShape);
    }
    if pool
        .lower_prices
        .iter()
        .any(|price| !valid_sqrt_price(*price))
        || pool
            .liquidity
            .iter()
            .any(|liquidity| !liquidity.is_finite())
    {
        return Err(SkipReason::InvalidShape);
    }
    if pool
        .lower_prices
        .windows(2)
        .any(|prices| prices[0] > prices[1])
    {
        return Err(SkipReason::InvalidTickOrder);
    }

    let lowest = pool.lower_prices[0];
    let highest = *pool.lower_prices.last().unwrap();
    if pool.current_price < lowest || pool.current_price > highest {
        return Err(SkipReason::CurrentPriceOutsideRange);
    }

    let mut liquidity = Vec::with_capacity(pool.liquidity.len());
    for &value in pool.liquidity[..pool.liquidity.len() - 1].iter().rev() {
        if value < -MIN_LIQUIDITY {
            return Err(SkipReason::NegativeLiquidity);
        }
        liquidity.push(value.max(0.0));
    }
    if !liquidity.iter().any(|&value| value > MIN_LIQUIDITY) {
        return Err(SkipReason::NoPositiveLiquidity);
    }
    liquidity.push(0.0);

    let lower_prices = pool.lower_prices.iter().rev().copied().collect();
    Ok(UniswapV3::new(
        pool.current_price,
        lower_prices,
        liquidity,
        pool.fee,
    ))
}

fn valid_sqrt_price(price: f64) -> bool {
    price.is_finite() && price > 0.0
}

fn sqrt_price_to_log_price(sqrt_price: f64) -> f64 {
    2.0 * sqrt_price.ln()
}

fn canonical_pair_log_price(pool: &RawUniswapV3Pool) -> ((String, String), f64) {
    let log_price = sqrt_price_to_log_price(pool.current_price);
    if pool.token0_symbol <= pool.token1_symbol {
        (
            (pool.token0_symbol.clone(), pool.token1_symbol.clone()),
            log_price,
        )
    } else {
        (
            (pool.token1_symbol.clone(), pool.token0_symbol.clone()),
            -log_price,
        )
    }
}

fn pair_median_log_prices(pools: &[RawUniswapV3Pool]) -> HashMap<(String, String), f64> {
    let mut pair_prices = HashMap::<(String, String), Vec<f64>>::new();
    for pool in pools {
        if !valid_sqrt_price(pool.current_price) {
            continue;
        }
        let (pair, log_price) = canonical_pair_log_price(pool);
        pair_prices.entry(pair).or_default().push(log_price);
    }

    pair_prices
        .into_iter()
        .map(|(pair, mut values)| {
            values.sort_by(|a, b| a.total_cmp(b));
            (pair, values[values.len() / 2])
        })
        .collect()
}

fn is_price_outlier(
    pool: &RawUniswapV3Pool,
    pair_medians: &HashMap<(String, String), f64>,
) -> bool {
    let (pair, log_price) = canonical_pair_log_price(pool);
    pair_medians
        .get(&pair)
        .is_some_and(|median| (log_price - median).abs() > OUTLIER_LOG_TOLERANCE)
}

fn initial_prices(tokens: usize, anchor: usize, edges: &[PriceEdge]) -> Array1<f64> {
    let pair_prices = median_token_pair_prices(edges);
    let mut adjacency = vec![Vec::<(usize, f64)>::new(); tokens];
    for ((token0, token1), log_price) in pair_prices {
        adjacency[token0].push((token1, -log_price));
        adjacency[token1].push((token0, log_price));
    }

    for neighbors in &mut adjacency {
        neighbors.sort_by_key(|(token, _)| *token);
    }
    let mut log_prices = vec![f64::NAN; tokens];
    let mut queue = VecDeque::new();
    log_prices[anchor] = 0.0;
    queue.push_back(anchor);

    while let Some(token) = queue.pop_front() {
        for &(next, delta) in &adjacency[token] {
            if log_prices[next].is_nan() {
                log_prices[next] = log_prices[token] + delta;
                queue.push_back(next);
            }
        }
    }

    Array1::from_iter(log_prices.into_iter().enumerate().map(|(idx, log_price)| {
        let lower_bound = if idx == anchor {
            1.0
        } else {
            f64::MIN_POSITIVE
        };
        if log_price.is_finite() {
            log_price.exp().max(lower_bound)
        } else {
            1.0_f64.max(lower_bound)
        }
    }))
}

fn median_token_pair_prices(edges: &[PriceEdge]) -> HashMap<(usize, usize), f64> {
    let mut pair_prices = HashMap::<(usize, usize), Vec<f64>>::new();
    for edge in edges {
        let (pair, log_price) = if edge.token0 <= edge.token1 {
            ((edge.token0, edge.token1), edge.log_price)
        } else {
            ((edge.token1, edge.token0), -edge.log_price)
        };
        pair_prices.entry(pair).or_default().push(log_price);
    }

    pair_prices
        .into_iter()
        .map(|(pair, mut values)| {
            values.sort_by(|a, b| a.total_cmp(b));
            (pair, values[values.len() / 2])
        })
        .collect()
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

#[cfg(test)]
mod convergence_tests;
#[cfg(test)]
mod real_pool_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use dexrouter_optim::{market::Market, solve_price_with_init_p_checked};
    use ndarray::arr1;

    fn raw_pool(
        token0: &str,
        token1: &str,
        current_price: f64,
        lower_prices: Vec<f64>,
        liquidity: Vec<f64>,
    ) -> RawUniswapV3Pool {
        RawUniswapV3Pool {
            token0_symbol: token0.to_owned(),
            token1_symbol: token1.to_owned(),
            current_price,
            lower_prices,
            liquidity,
            fee: 0.997,
        }
    }

    #[test]
    fn converts_json_pool_to_optim_convention() {
        let raw = raw_pool("A", "B", 2.5, vec![1.0, 2.0, 3.0], vec![5.0, 7.0, 0.0]);
        let actual = convert_pool(&raw).unwrap();
        let expected = UniswapV3::new(2.5, vec![3.0, 2.0, 1.0], vec![7.0, 5.0, 0.0], 0.997);

        assert_eq!(actual.lower_prices(), expected.lower_prices());
        let (actual_input, actual_output) = actual.arbitrage([10.0, 1.0]);
        let (expected_input, expected_output) = expected.arbitrage([10.0, 1.0]);
        assert!(arr1(&actual_input).abs_diff_eq(&arr1(&expected_input), 1e-12));
        assert!(arr1(&actual_output).abs_diff_eq(&arr1(&expected_output), 1e-12));
    }

    #[test]
    fn rejects_current_price_outside_range() {
        let raw = raw_pool("A", "B", 4.0, vec![1.0, 2.0, 3.0], vec![5.0, 7.0, 0.0]);
        assert_eq!(
            convert_pool(&raw).unwrap_err(),
            SkipReason::CurrentPriceOutsideRange
        );
    }

    #[test]
    fn detects_orientation_aware_pair_outliers() {
        let pools = vec![
            raw_pool("A", "B", 2.0, vec![1.0, 3.0], vec![1.0, 0.0]),
            raw_pool("B", "A", 0.5, vec![0.1, 1.0], vec![1.0, 0.0]),
            raw_pool("A", "B", 100.0, vec![1.0, 101.0], vec![1.0, 0.0]),
        ];
        let medians = pair_median_log_prices(&pools);

        assert!(!is_price_outlier(&pools[0], &medians));
        assert!(!is_price_outlier(&pools[1], &medians));
        assert!(is_price_outlier(&pools[2], &medians));
    }

    #[test]
    fn initial_prices_use_usdc_anchor() {
        let weth_usdc = PriceEdge {
            token0: 0,
            token1: 1,
            log_price: 2000.0_f64.ln(),
        };
        let dai_usdc = PriceEdge {
            token0: 2,
            token1: 1,
            log_price: 1.0_f64.ln(),
        };

        let prices = initial_prices(3, 1, &[weth_usdc, dai_usdc]);

        assert!((prices[0] - 2000.0).abs() < 1e-6);
        assert!((prices[2] - 1.0).abs() < 1e-8);
        assert!(prices[1] >= 1.0);
    }

    #[test]
    fn checked_solver_runs_on_small_full_market() {
        let pools = vec![
            raw_pool(
                "USDC",
                "WETH",
                (1.0_f64 / 2000.0).sqrt(),
                vec![0.01, 0.1],
                vec![100_000.0, 0.0],
            ),
            raw_pool(
                "WETH",
                "DAI",
                2000.0_f64.sqrt(),
                vec![10.0, 100.0],
                vec![100.0, 0.0],
            ),
            raw_pool("USDC", "DAI", 1.0, vec![0.5, 2.0], vec![100_000.0, 0.0]),
        ];
        let full_market = build_full_market_route(pools).unwrap();
        let diagnostics =
            solve_price_with_init_p_checked(full_market.route, full_market.initial_price).unwrap();

        assert!(diagnostics.evaluations > 0);
        assert!(diagnostics.price.iter().all(|value| value.is_finite()));
        assert!(diagnostics.cost.is_finite());
    }
}
