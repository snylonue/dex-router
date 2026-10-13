//! Offline Uniswap V3 snapshot conversion, filtering, and USDC-anchored routes.

use std::{
    collections::{HashMap, VecDeque},
    fs,
    path::Path,
};

use crate::{Route, market::UniswapV3, utility::NonnegativeLinear};
use ndarray::Array1;
use serde::Deserialize;

pub const ANCHOR_TOKEN: &str = "USDC";
const OUTLIER_LOG_TOLERANCE: f64 = 4.0;
const MIN_LIQUIDITY: f64 = 1e-12;
const DEFAULT_MIN_ACTIVE_TVL_USD: f64 = 1_000.0;

#[derive(Debug, Clone, Deserialize)]
pub struct RawUniswapV3Pool {
    pub token0_symbol: String,
    pub token1_symbol: String,
    pub current_price: f64,
    pub lower_prices: Vec<f64>,
    pub liquidity: Vec<f64>,
    pub fee: f64,
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
pub struct LoadStats {
    pub total_pools: usize,
    pub kept_pools: usize,
    pub invalid_price: usize,
    pub invalid_fee: usize,
    pub invalid_shape: usize,
    pub invalid_tick_order: usize,
    pub current_price_outside_range: usize,
    pub no_positive_liquidity: usize,
    pub negative_liquidity: usize,
    pub price_outlier: usize,
    pub low_active_tvl: usize,
}

impl LoadStats {
    pub fn skipped_pools(&self) -> usize {
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
pub struct FullMarketRoute {
    pub tokens: Vec<String>,
    pub route: Route<NonnegativeLinear, UniswapV3>,
    pub initial_price: Array1<f64>,
    pub stats: LoadStats,
}

impl FullMarketRoute {
    /// Reuse a previous solution by token identity, not token ordering. New tokens
    /// retain graph-fitted initial prices. This changes no pool or utility bound.
    pub fn warm_start_prices(
        &self,
        previous_tokens: &[String],
        previous_prices: &Array1<f64>,
    ) -> anyhow::Result<Array1<f64>> {
        anyhow::ensure!(
            previous_tokens.len() == previous_prices.len(),
            "warm-start length mismatch"
        );
        let mut previous = HashMap::new();
        for (token, &price) in previous_tokens.iter().zip(previous_prices) {
            anyhow::ensure!(price.is_finite() && price > 0.0, "invalid warm-start price");
            anyhow::ensure!(
                previous.insert(token.as_str(), price).is_none(),
                "duplicate warm-start token"
            );
        }
        let mut price = self.initial_price.clone();
        for (i, token) in self.tokens.iter().enumerate() {
            if token == ANCHOR_TOKEN {
                price[i] = 1.0;
            } else if let Some(&old) = previous.get(token.as_str()) {
                price[i] = old;
            }
        }
        Ok(price)
    }
}

pub fn load_full_market_route(path: impl AsRef<Path>) -> anyhow::Result<FullMarketRoute> {
    let pools: Vec<RawUniswapV3Pool> = serde_json::from_reader(fs::File::open(path)?)?;
    build_full_market_route(pools)
}

pub fn build_full_market_route(pools: Vec<RawUniswapV3Pool>) -> anyhow::Result<FullMarketRoute> {
    build_full_market_route_with_tvl(pools, min_active_tvl_usd())
}

pub fn build_full_market_route_with_tvl(
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
    // Preserve the existing pool-selection policy. The improved initializer must
    // not change TVL filtering, hence cannot remove or add economic opportunities.
    let initial_price = initial_prices_bfs(tokens.len(), anchor, &edges);

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

fn initial_prices_bfs(tokens: usize, anchor: usize, edges: &[PriceEdge]) -> Array1<f64> {
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

// Fit q_a - q_b to pair-median log prices. Pair observation counts supply
// deterministic confidence weights. Logs are used for initialization only;
// the economic objective is still optimized in affine price coordinates.
fn initial_prices(tokens: usize, anchor: usize, edges: &[PriceEdge]) -> Array1<f64> {
    let medians = median_token_pair_prices(edges);
    let mut counts = HashMap::<(usize, usize), usize>::new();
    for edge in edges {
        *counts
            .entry((edge.token0.min(edge.token1), edge.token0.max(edge.token1)))
            .or_default() += 1;
    }
    let mut pairs: Vec<_> = medians
        .into_iter()
        .filter(|((a, b), _)| a != b)
        .map(|((a, b), r)| (a, b, r, counts[&(a, b)] as f64))
        .collect();
    pairs.sort_by_key(|&(a, b, _, _)| (a, b));
    let groups = crate::graph::components(tokens, pairs.iter().map(|&(a, b, _, _)| (a, b)));
    let mut fixed = vec![false; tokens];
    let mut adjacency = vec![Vec::new(); tokens];
    for &(a, b, r, _) in &pairs {
        adjacency[a].push((b, -r));
        adjacency[b].push((a, r));
    }
    let mut logs = vec![f64::NAN; tokens];
    for group in groups {
        // A gauge for the initializer only, not a new solver/economic constraint.
        let root = if group.contains(&anchor) {
            anchor
        } else {
            group[0]
        };
        fixed[root] = true;
        logs[root] = 0.0;
        let mut queue = VecDeque::from([root]);
        while let Some(a) = queue.pop_front() {
            for &(b, delta) in &adjacency[a] {
                if logs[b].is_nan() {
                    logs[b] = logs[a] + delta;
                    queue.push_back(b);
                }
            }
        }
    }
    let mut diagonal = vec![0.0; tokens];
    let mut rhs = vec![0.0; tokens];
    for &(a, b, r, w) in &pairs {
        let residual = w * (r - (logs[a] - logs[b]));
        rhs[a] += residual;
        rhs[b] -= residual;
        diagonal[a] += w;
        diagonal[b] += w;
    }
    for i in 0..tokens {
        if fixed[i] {
            diagonal[i] = 1.0;
            rhs[i] = 0.0;
        }
    }
    let correction = crate::graph::pcg(&diagonal, &rhs, |x| {
        let mut out = vec![0.0; tokens];
        for &(a, b, _, w) in &pairs {
            let delta = w * (if fixed[a] { 0.0 } else { x[a] } - if fixed[b] { 0.0 } else { x[b] });
            out[a] += delta;
            out[b] -= delta;
        }
        for i in 0..tokens {
            if fixed[i] {
                out[i] = x[i];
            }
        }
        out
    });
    if let Some(correction) = correction {
        let candidate: Vec<_> = logs.iter().zip(correction).map(|(q, d)| q + d).collect();
        let error = |q: &[f64]| {
            pairs
                .iter()
                .map(|&(a, b, r, w)| w * (q[a] - q[b] - r).powi(2))
                .sum::<f64>()
        };
        if candidate.iter().all(|q| q.is_finite()) && error(&candidate) <= error(&logs) {
            logs = candidate;
        }
    }
    Array1::from_iter(logs.into_iter().enumerate().map(|(i, q)| {
        if i == anchor {
            1.0
        } else {
            q.clamp(-700.0, 700.0).exp()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{market::Market, solve_price_with_init_p_checked};
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
    fn graph_fit_uses_cycles_and_is_independent_of_pool_order() {
        let edges = vec![
            PriceEdge {
                token0: 0,
                token1: 1,
                log_price: 0.0,
            },
            PriceEdge {
                token0: 1,
                token1: 2,
                log_price: 0.0,
            },
            PriceEdge {
                token0: 0,
                token1: 2,
                log_price: 3.0,
            },
            PriceEdge {
                token0: 3,
                token1: 4,
                log_price: 2.0,
            },
        ];
        let fitted = initial_prices(6, 0, &edges);
        assert!((fitted[1].ln() + 1.0).abs() < 1e-10);
        assert!((fitted[2].ln() + 2.0).abs() < 1e-10);
        assert!((fitted[3] / fitted[4] - 2.0_f64.exp()).abs() < 1e-10);
        assert_eq!(fitted[0], 1.0);
        assert_eq!(fitted[5], 1.0);
        let mut reversed = edges;
        reversed.reverse();
        assert_eq!(fitted, initial_prices(6, 0, &reversed));
        let bfs = initial_prices_bfs(6, 0, &reversed);
        let error = |p: &Array1<f64>| {
            reversed
                .iter()
                .map(|edge| (p[edge.token0].ln() - p[edge.token1].ln() - edge.log_price).powi(2))
                .sum::<f64>()
        };
        assert!(error(&fitted) < error(&bfs));
    }

    #[test]
    fn warm_start_matches_token_identity_and_preserves_route() {
        let pools = vec![
            raw_pool("USDC", "A", 1.0, vec![0.5, 2.0], vec![100_000.0, 0.0]),
            raw_pool("USDC", "B", 1.0, vec![0.5, 2.0], vec![100_000.0, 0.0]),
        ];
        let full = build_full_market_route_with_tvl(pools, 0.0).unwrap();
        let previous_tokens = vec!["A".into(), "USDC".into(), "removed".into()];
        let prices = full
            .warm_start_prices(&previous_tokens, &arr1(&[1.2, 8.0, 3.0]))
            .unwrap();
        for (i, token) in full.tokens.iter().enumerate() {
            match token.as_str() {
                "A" => assert_eq!(prices[i], 1.2),
                "USDC" => assert_eq!(prices[i], 1.0),
                _ => assert_eq!(prices[i], full.initial_price[i]),
            }
        }
        assert_eq!(full.route.markets.len(), 2);
        assert!(
            full.warm_start_prices(&previous_tokens, &arr1(&[1.0]))
                .is_err()
        );
        assert!(
            full.warm_start_prices(&previous_tokens, &arr1(&[f64::NAN, 1.0, 1.0]))
                .is_err()
        );
        assert!(
            full.warm_start_prices(&["A".into(), "A".into()], &arr1(&[1.0, 1.0]))
                .is_err()
        );
    }

    #[test]
    fn fitted_initializer_does_not_change_tvl_pool_selection() {
        let pools = vec![
            raw_pool("USDC", "A", 1.0, vec![0.5, 2.0], vec![100_000.0, 0.0]),
            raw_pool("A", "B", 2.0, vec![0.5, 3.0], vec![2000.0, 0.0]),
            raw_pool("USDC", "B", 1.0, vec![0.5, 2.0], vec![4000.0, 0.0]),
        ];
        let accepted: Vec<_> = pools
            .iter()
            .map(|pool| AcceptedPool {
                market: convert_pool(pool).unwrap(),
                token0_symbol: pool.token0_symbol.clone(),
                token1_symbol: pool.token1_symbol.clone(),
                log_price: sqrt_price_to_log_price(pool.current_price),
                current_price: pool.current_price,
                lower_prices: pool.lower_prices.clone(),
                liquidity: pool.liquidity.clone(),
            })
            .collect();
        let (tokens, indices, _, edges) = indexed_route_parts(&accepted);
        let old_prices = initial_prices_bfs(tokens.len(), indices[ANCHOR_TOKEN], &edges);
        let cutoff = 3000.0;
        let expected: Vec<_> = accepted
            .iter()
            .filter(|pool| {
                active_pool_value(
                    pool,
                    old_prices[indices[&pool.token0_symbol]],
                    old_prices[indices[&pool.token1_symbol]],
                )
                .is_some_and(|value| value >= cutoff)
            })
            .map(|pool| (&pool.token0_symbol, &pool.token1_symbol))
            .collect();
        let full = build_full_market_route_with_tvl(pools, cutoff).unwrap();
        let actual: Vec<_> = full
            .route
            .markets
            .iter()
            .map(|(_, (a, b))| (&full.tokens[*a], &full.tokens[*b]))
            .collect();
        assert_eq!(actual, expected);
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
