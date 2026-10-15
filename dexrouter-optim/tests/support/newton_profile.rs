//! Opt-in callback attribution. Per-call clock overhead is included in wall time;
//! use compare_snapshot_backends (without instrumentation) for speed comparisons.
use super::*;
use dexrouter_optim::market::MarketEvaluation;
use std::{cell::Cell, rc::Rc, time::Duration};

type Counters = Rc<[Cell<(u64, Duration)>; 3]>;

#[derive(Clone)]
struct TimedMarket {
    market: UniswapV3,
    counters: Counters,
}
impl TimedMarket {
    fn timed<T>(&self, kind: usize, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        let (calls, time) = self.counters[kind].get();
        self.counters[kind].set((calls + 1, time + elapsed));
        result
    }
}
impl Market for TimedMarket {
    fn arbitrage(&self, v: [f64; 2]) -> ([f64; 2], [f64; 2]) {
        let t = self.evaluate(v);
        (t.input, t.output)
    }
    fn evaluate(&self, v: [f64; 2]) -> MarketEvaluation {
        self.timed(0, || self.market.evaluate(v))
    }
    fn value_change(
        &self,
        v: [f64; 2],
        reference_v: [f64; 2],
        trade: MarketEvaluation,
        reference: MarketEvaluation,
    ) -> f64 {
        self.timed(1, || {
            self.market.value_change(v, reference_v, trade, reference)
        })
    }
    fn relative_curvature(&self, v: [f64; 2]) -> Option<f64> {
        self.timed(2, || self.market.relative_curvature(v))
    }
}

#[test]
#[ignore = "callback attribution; run release with nocapture and one test thread"]
fn profile_snapshot_callbacks() {
    let pools: Vec<RawUniswapV3Pool> =
        serde_json::from_str(include_str!("../fixtures/uniswap_v3_pools.json")).unwrap();
    for (label, cutoff, tolerance) in [("filtered", 1000.0, 1e-8), ("full", 0.0, 1e-6)] {
        let full = build_full_market_route_with_tvl(pools.clone(), cutoff).unwrap();
        for backend in ["lbfgsb", "newton"] {
            let counters: Counters = Rc::new(std::array::from_fn(|_| Cell::default()));
            let route = Route {
                objective: full.route.objective.clone(),
                tokens: full.route.tokens,
                markets: full
                    .route
                    .markets
                    .iter()
                    .map(|(market, pair)| {
                        (
                            TimedMarket {
                                market: market.clone(),
                                counters: counters.clone(),
                            },
                            *pair,
                        )
                    })
                    .collect(),
            };
            let options = SolveOptions {
                absolute_residual_value: tolerance,
                ..Default::default()
            };
            let start = Instant::now();
            let (fg, hvp, models, kkt, assessment, elapsed) = if backend == "newton" {
                let r = solve_price_with_newton(
                    route,
                    full.initial_price.clone(),
                    options.clone(),
                    dexrouter_newton::Options {
                        pgtol: options.optimizer.pgtol,
                        max_iterations: options.optimizer.max_iterations,
                        max_evaluations: options.optimizer.max_evaluations,
                        ..Default::default()
                    },
                )
                .unwrap();
                let elapsed = start.elapsed();
                (
                    r.evaluations,
                    r.hessian_products,
                    r.hessian_evaluations,
                    independent(&full.route, &r, &options),
                    r.assessment,
                    elapsed,
                )
            } else {
                let r =
                    solve_price_with_options(route, full.initial_price.clone(), options.clone())
                        .unwrap();
                let elapsed = start.elapsed();
                (
                    r.evaluations,
                    0,
                    0,
                    independent(&full.route, &r, &options),
                    r.assessment,
                    elapsed,
                )
            };
            println!(
                "profile snapshot={label} backend={backend} wall={:.6} FG={fg} HVP={hvp} models={models} KKT={kkt:.6e} assessment={assessment:?}",
                elapsed.as_secs_f64()
            );
            for (kind, counter) in ["evaluate", "value_change", "curvature"]
                .iter()
                .zip(counters.iter())
            {
                let (calls, time) = counter.get();
                println!(
                    "  {kind}: calls={calls} seconds={:.6} wall_percent={:.1}",
                    time.as_secs_f64(),
                    100.0 * time.as_secs_f64() / elapsed.as_secs_f64()
                );
            }
        }
    }
}
