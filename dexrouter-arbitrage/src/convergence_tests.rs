//! Opt-in offline real-data diagnosis; production routing is unchanged.
use super::*;
use argmin::core::Gradient;
use dexrouter_optim::market::Market;
#[path = "../../dexrouter-optim/tests/support/convergence.rs"]
mod convergence;

fn inspect_conditioning<U: UtilityConjugate>(
    route: &Route<U, UniswapV3>,
    price: &Array1<f64>,
    names: &[String],
) {
    let mut positive_diagonal = Vec::new();
    for i in 0..route.tokens {
        let mut plus = price.clone();
        let mut minus = price.clone();
        plus[i] *= 1.0 + 1e-5;
        minus[i] *= 1.0 - 1e-5;
        if minus[i] < route.objective.lower_bounds()[i] {
            continue;
        }
        let diagonal = (route.gradient(&plus).unwrap()[i] - route.gradient(&minus).unwrap()[i])
            * price[i]
            / (2e-5);
        if diagonal > 1e-10 {
            positive_diagonal.push((diagonal, names[i].clone()));
        }
    }
    positive_diagonal.sort_by(|a, b| a.0.total_cmp(&b.0));
    if let (Some(min), Some(max)) = (positive_diagonal.first(), positive_diagonal.last()) {
        println!(
            "curvature positive_diagonal_min={min:?} max={max:?} diagonal_ratio={:.3e}",
            max.0 / min.0
        );
    }
    let mut worst = (0.0_f64, 0usize, 0usize, 0.0, 0.0);
    for (index, (pool, (a, b))) in route.markets.iter().enumerate() {
        let v = [price[*a], price[*b]];
        let (input, output) = pool.arbitrage(v);
        let value = |v: [f64; 2]| {
            let (input, output) = pool.arbitrage(v);
            output[0] * v[0] + output[1] * v[1] - input[0] * v[0] - input[1] * v[1]
        };
        for j in 0..2 {
            let analytic = (output[j] - input[j]) * v[j];
            let mut best = (f64::INFINITY, 0.0);
            for h in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9] {
                let mut plus = v;
                let mut minus = v;
                plus[j] *= 1.0 + h;
                minus[j] *= 1.0 - h;
                let fd = (value(plus) - value(minus)) / (2.0 * h);
                let error = (fd - analytic).abs() / fd.abs().max(analytic.abs()).max(1.0);
                if error < best.0 {
                    best = (error, fd);
                }
            }
            if best.0 > worst.0 {
                worst = (best.0, index, j, best.1, analytic);
            }
        }
    }
    let (_, (a, b)) = &route.markets[worst.1];
    println!(
        "pool_gradient best-step-worst-relative-error={:.3e} pool={} pair={}/{} component={} fd={:.6e} analytic={:.6e}",
        worst.0, worst.1, names[*a], names[*b], worst.2, worst.3, worst.4
    );
}

#[derive(Clone)]
struct FloorUtility {
    inner: NonnegativeLinear,
    floor: f64,
}
impl UtilityConjugate for FloorUtility {
    fn value(&self, v: &Array1<f64>) -> f64 {
        self.inner.value(v)
    }
    fn grad(&self, v: &Array1<f64>) -> Array1<f64> {
        self.inner.grad(v)
    }
    fn lower_bounds(&self) -> Array1<f64> {
        self.inner
            .c
            .mapv(|c| if c > 0.0 { c + 1e-8 } else { self.floor })
    }
    fn upper_bounds(&self) -> Array1<f64> {
        self.inner.upper_bounds()
    }
}

#[test]
#[ignore = "real-data curvature, local market gradient, warm-rescaling and price-floor diagnosis"]
fn diagnose_real_conditioning() {
    use convergence::{Config, check_gradient, run};
    let full = load_full_market_route(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../uniswap_v3_pools.json"),
    )
    .unwrap();
    println!(
        "condition dataset pools={} tokens={}",
        full.route.markets.len(),
        full.route.tokens
    );
    inspect_conditioning(
        &full.route,
        &full.initial_price.mapv(|p| p * 1.001),
        &full.tokens,
    );
    let warm = run(
        &full.route,
        &full.initial_price,
        Config {
            label: "warm-5000",
            iterations: 5000,
            ..Config::default()
        },
        false,
    );
    inspect_conditioning(&full.route, &warm.price.mapv(|p| p * 1.001), &full.tokens);
    check_gradient(&full.route, &warm.price, "real-warm");
    let ratios = &warm.price / &full.initial_price;
    println!(
        "price_change initial_to_warm_min={:.3e} max={:.3e}",
        ratios.iter().copied().fold(f64::INFINITY, f64::min),
        ratios.iter().copied().fold(0.0, f64::max)
    );
    let config = Config {
        label: "warm-rescaled-1000",
        ..Config::default()
    };
    run(&full.route, &warm.price, config, false);
    run(&full.route, &warm.price, config, true);
    for floor in [1e-12, 1e-16] {
        let route = Route {
            objective: FloorUtility {
                inner: full.route.objective.clone(),
                floor,
            },
            markets: full.route.markets.clone(),
            tokens: full.route.tokens,
        };
        println!("price_floor={floor:.0e}");
        run(
            &route,
            &warm.price,
            Config {
                label: "reduced-floor-warm-1000",
                ..Config::default()
            },
            false,
        );
        run(
            &route,
            &warm.price,
            Config {
                label: "reduced-floor-warm-1000",
                ..Config::default()
            },
            true,
        );
    }
}

#[test]
#[ignore = "requires local pool snapshot; matched reference/normalization/budget experiments"]
fn diagnose_real_convergence() {
    use convergence::{Config, check_gradient, run};
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../uniswap_v3_pools.json");
    let full = load_full_market_route(path).unwrap();
    let route = &full.route;
    let initial = &full.initial_price;
    let anchor = full
        .tokens
        .iter()
        .position(|token| token == ANCHOR_TOKEN)
        .unwrap();
    let cost_scale = route.utility(initial).max(1.0);
    let min_price = initial.iter().copied().fold(f64::INFINITY, f64::min);
    let max_price = initial.iter().copied().fold(0.0, f64::max);
    println!(
        "real pools={} tokens={} initial_cost={cost_scale:.12e} initial_price_range=[{min_price:.3e},{max_price:.3e}] anchor={anchor}",
        route.markets.len(),
        route.tokens
    );
    check_gradient(route, initial, "real-initial");
    for config in [
        Config::default(),
        Config {
            label: "normalized",
            cost_scale,
            ..Config::default()
        },
        Config {
            label: "fixed-anchor",
            anchor: Some(anchor),
            ..Config::default()
        },
        Config {
            label: "normalized-fixed-anchor",
            cost_scale,
            anchor: Some(anchor),
            ..Config::default()
        },
        Config {
            label: "stable-sum",
            stable_sum: true,
            ..Config::default()
        },
        Config {
            label: "budget-5000",
            iterations: 5000,
            ..Config::default()
        },
    ] {
        let rust = run(route, initial, config, false);
        let reference = run(route, initial, config, true);
        println!(
            "real comparison label={} relative_cost_difference={:.3e} rust_pg={:.3e} reference_pg={:.3e}",
            config.label,
            (rust.cost - reference.cost).abs() / rust.cost.abs().max(reference.cost.abs()).max(1.0),
            rust.pg,
            reference.pg
        );
        if config.label == "default" {
            check_gradient(route, &rust.price, "real-rust-last");
        }
    }
}
