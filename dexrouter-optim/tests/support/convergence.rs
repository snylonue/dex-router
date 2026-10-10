//! Test-only controlled new/reference solver comparisons. No production changes.
use argmin::core::Gradient;
use dexrouter_lbfgsb::{Bound, Options, minimize};
use dexrouter_optim::{Route, market::Market, utility::UtilityConjugate};
use ndarray::{Array1, arr1};
use std::{sync::Mutex, time::Instant};

static REFERENCE_LOCK: Mutex<()> = Mutex::new(());

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

pub struct Outcome {
    pub price: Array1<f64>,
    pub cost: f64,
    pub pg: f64,
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
    reference: bool,
) -> Outcome {
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
    let (x, status, iterations, evaluations) = if reference {
        reference_run(vec![1.0; route.tokens], &bounds, config, fg)
    } else {
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
        (
            result.x,
            format!("{:?}", result.status),
            result.iterations,
            result.evaluations,
        )
    };
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
        "compare label={} solver={} scale={:.3e} anchor={:?} status={status} iterations={iterations} evaluations={evaluations} seconds={:.3} cost={original_cost:.12e} raw_scaled_pg={raw_pg:.3e} max_negative_value={worst_negative_value:.3e}",
        config.label,
        if reference { "reference" } else { "rust" },
        config.cost_scale,
        config.anchor,
        start.elapsed().as_secs_f64()
    );
    Outcome {
        price,
        cost: original_cost,
        pg: raw_pg,
    }
}

fn reference_run<F: FnMut(&[f64], &mut [f64]) -> anyhow::Result<f64>>(
    mut x: Vec<f64>,
    bounds: &[Bound<f64>],
    config: Config,
    mut fg: F,
) -> (Vec<f64>, String, usize, usize) {
    // The public legacy wrapper discards the termination code and has no
    // iteration cap. Use its public reverse-communication entry point so
    // comparisons can match the iteration budget and inspect actual status.
    // Codes and workspace sizes follow lbfgsb 0.1.1's lbfgsb.h and Rust wrapper.
    let _guard = REFERENCE_LOCK.lock().unwrap();
    let n = x.len() as i64;
    let m = 17_i64;
    let l: Vec<_> = bounds.iter().map(|b| b.lower.unwrap_or(0.0)).collect();
    let u: Vec<_> = bounds.iter().map(|b| b.upper.unwrap_or(0.0)).collect();
    let nbd: Vec<i64> = bounds
        .iter()
        .map(|b| match (b.lower, b.upper) {
            (None, None) => 0,
            (Some(_), None) => 1,
            (Some(_), Some(_)) => 2,
            (None, Some(_)) => 3,
        })
        .collect();
    for (xi, bound) in x.iter_mut().zip(bounds) {
        *xi = bound.project(*xi);
    }
    let mut accepted = x.clone();
    let mut f = 0.0;
    let mut g = vec![0.0; n as usize];
    let mut wa = vec![0.0; (2 * m * n + 5 * n + 11 * m * m + 8 * m) as usize];
    let mut iwa = vec![0_i64; 3 * n as usize];
    let mut task = 1_i64;
    let iprint = -1_i64;
    let mut csave = [0_i64; 60];
    let mut lsave = [0_i64; 4];
    let mut isave = [0_i64; 44];
    let mut dsave = [0.0; 29];
    let pgtol = 1e-5 / config.cost_scale;
    let mut evaluations = 0;
    let mut iterations = 0;
    let status = loop {
        // SAFETY: all arrays have the legacy wrapper's required lengths and
        // scalar ABI types; they remain alive and uniquely mutable for the
        // call. The non-reentrant C implementation is serialized above.
        unsafe {
            lbfgsb::setulb(
                &n,
                &m,
                x.as_mut_ptr(),
                l.as_ptr(),
                u.as_ptr(),
                nbd.as_ptr(),
                &mut f,
                g.as_mut_ptr(),
                &config.factr,
                &pgtol,
                wa.as_mut_ptr(),
                iwa.as_mut_ptr(),
                &mut task,
                &iprint,
                csave.as_mut_ptr(),
                lsave.as_mut_ptr(),
                isave.as_mut_ptr(),
                dsave.as_mut_ptr(),
            );
        }
        match task {
            10..=15 => {
                if evaluations >= 50_000 {
                    x = accepted;
                    break "MaxEvaluations".into();
                }
                f = fg(&x, &mut g).unwrap();
                assert!(f.is_finite() && g.iter().all(|v| v.is_finite()));
                evaluations += 1;
            }
            2 => {
                accepted = x.clone();
                iterations += 1;
                if iterations >= config.iterations {
                    break "MaxIterations".into();
                }
            }
            21 => break "ConvergedProjectedGradient".into(),
            22 => break "ConvergedFunctionReduction".into(),
            3 => break "LineSearchFailed".into(),
            other => break format!("Task({other})"),
        }
    };
    (x, status, iterations, evaluations)
}

#[test]
fn reference_driver_recovers_known_bounded_quadratic() {
    let (x, status, _, _) = reference_run(
        vec![0.0, 0.0],
        &[Bound::both(-5.0, 5.0), Bound::both(-5.0, 5.0)],
        Config::default(),
        |x, g| {
            g[0] = x[0] - 2.0;
            g[1] = x[1] + 1.0;
            Ok(0.5 * (g[0] * g[0] + g[1] * g[1]))
        },
    );
    assert_eq!(status, "ConvergedProjectedGradient");
    assert!((x[0] - 2.0).abs() < 1e-8 && (x[1] + 1.0).abs() < 1e-8);
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
