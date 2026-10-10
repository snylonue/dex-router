//! Test-only controlled Rust solver scaling and precision diagnostics.
use argmin::core::Gradient;
use dexrouter_lbfgsb::{Bound, Options, minimize};
use dexrouter_optim::{Route, market::Market, utility::UtilityConjugate};
use ndarray::{Array1, arr1};
use std::time::Instant;

#[derive(Clone, Copy)]
pub struct Config {
    pub label: &'static str,
    pub cost_scale: f64,
    pub anchor: Option<usize>,
    pub iterations: usize,
    pub maxls: usize,
    pub factr: f64,
    pub stable_sum: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            label: "default",
            cost_scale: 1.0,
            anchor: None,
            iterations: 1000,
            maxls: 20,
            factr: 0.0,
            stable_sum: false,
        }
    }
}

fn stable_cost<U: UtilityConjugate, M: Market>(route: &Route<U, M>, price: &Array1<f64>) -> f64 {
    let mut sum = route.objective.value(price);
    let mut correction = 0.0;
    for (pool, (a, b)) in &route.markets {
        let (input, output) = pool.arbitrage([price[*a], price[*b]]);
        for term in [
            (output[0] - input[0]) * price[*a],
            (output[1] - input[1]) * price[*b],
        ] {
            let next = sum + term;
            correction += if sum.abs() >= term.abs() {
                (sum - next) + term
            } else {
                (term - next) + sum
            };
            sum = next;
        }
    }
    sum + correction
}

// x - P(x - g), evaluated without cancellation for unbounded coordinates.
fn pg(x: &[f64], g: &[f64], bounds: &[Bound<f64>]) -> f64 {
    x.iter()
        .zip(g)
        .zip(bounds)
        .map(|((&x, &g), bound)| {
            if g < 0.0 {
                bound.upper.map_or(g, |u| g.max(x - u))
            } else {
                bound.lower.map_or(g, |l| g.min(x - l))
            }
            .abs()
        })
        .fold(0.0, f64::max)
}

pub fn run<U: UtilityConjugate, M: Market>(
    route: &Route<U, M>,
    initial: &Array1<f64>,
    config: Config,
) -> Array1<f64> {
    let lower = route.objective.lower_bounds();
    let upper = route.objective.upper_bounds();
    let mut bounds: Vec<_> = lower
        .iter()
        .zip(&upper)
        .zip(initial)
        .map(|((&l, &u), &s)| Bound::new(Some((l / s).max(1e-12)), u.is_finite().then_some(u / s)))
        .collect();
    if let Some(anchor) = config.anchor {
        let fixed = lower[anchor] / initial[anchor];
        bounds[anchor] = Bound::both(fixed, fixed);
    }
    let cost = |price: &Array1<f64>| {
        if config.stable_sum {
            stable_cost(route, price)
        } else {
            route.utility(price)
        }
    };
    let fg = |x: &[f64], g: &mut [f64]| -> anyhow::Result<f64> {
        let price = arr1(x) * initial;
        let gradient = route.gradient(&price)? * initial / config.cost_scale;
        g.copy_from_slice(gradient.as_slice().unwrap());
        Ok(cost(&price) / config.cost_scale)
    };
    let start = Instant::now();
    let result = minimize(
        vec![1.0; route.tokens],
        bounds.clone(),
        Options {
            max_iterations: config.iterations,
            max_evaluations: 50_000,
            max_line_search_steps: config.maxls,
            pgtol: 1e-5 / config.cost_scale,
            factr: config.factr,
            ..Options::default()
        },
        fg,
    )
    .unwrap();
    let (x, status, iterations, evaluations) = (
        result.x,
        format!("{:?}", result.status),
        result.iterations,
        result.evaluations,
    );
    let price = arr1(&x) * initial;
    let g = (route.gradient(&price).unwrap() * initial).to_vec();
    let original_cost = route.utility(&price);
    let raw_pg = pg(&x, &g, &bounds);
    let worst_negative_value = route
        .gradient(&price)
        .unwrap()
        .iter()
        .zip(&price)
        .map(|(&g, &p)| (-g * p).max(0.0))
        .fold(0.0, f64::max);
    println!(
        "diagnose label={} solver=rust scale={:.3e} anchor={:?} status={status} iterations={iterations} evaluations={evaluations} seconds={:.3} cost={original_cost:.12e} raw_scaled_pg={raw_pg:.3e} max_negative_value={worst_negative_value:.3e}",
        config.label,
        config.cost_scale,
        config.anchor,
        start.elapsed().as_secs_f64()
    );
    price
}

pub fn check_gradient<U: UtilityConjugate, M: Market>(
    route: &Route<U, M>,
    initial: &Array1<f64>,
    label: &str,
) {
    let lower = route.objective.lower_bounds();
    let point = Array1::from_iter(initial.iter().zip(&lower).map(|(&p, &l)| p.max(l) * 1.001));
    let gradient = route.gradient(&point).unwrap() * initial;
    for h in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8] {
        let mut worst = (0.0_f64, 0usize, 0.0, 0.0);
        for i in 0..route.tokens {
            let mut plus = point.clone();
            let mut minus = point.clone();
            plus[i] += h * initial[i];
            minus[i] -= h * initial[i];
            let fd = (route.utility(&plus) - route.utility(&minus)) / (2.0 * h);
            let error = (fd - gradient[i]).abs() / fd.abs().max(gradient[i].abs()).max(1.0);
            if error > worst.0 {
                worst = (error, i, fd, gradient[i]);
            }
        }
        println!(
            "gradient label={label} h={h:.0e} worst_relative_error={:.3e} token={} fd={:.6e} analytic={:.6e}",
            worst.0, worst.1, worst.2, worst.3
        );
    }
}
