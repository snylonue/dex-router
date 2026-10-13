use super::cg_solve;
use crate::memory::LbfgsMemory;

#[test]
#[ignore = "release-only compact vs CG benchmark, including fresh-history construction costs"]
fn benchmark_compact_vs_cg() {
    use std::{hint::black_box, time::Instant};
    let n = 438;
    let free: Vec<_> = (1..n).collect();
    let mut mask = vec![true; n];
    mask[0] = false;
    let mut rhs: Vec<_> = (0..n).map(|i| (i as f64).cos()).collect();
    rhs[0] = 0.;
    for condition in [10_f64, 1e4] {
        let mut base = LbfgsMemory::new(17);
        let mut seed = 0x123456789abcdef_u64;
        for _ in 0..17 {
            let s: Vec<_> = (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed >> 11) as f64 / (1_u64 << 53) as f64 - 0.5
                })
                .collect();
            let y: Vec<_> = s
                .iter()
                .enumerate()
                .map(|(i, &v)| v * condition.powf((i % 17) as f64 / 16.))
                .collect();
            assert!(base.push(s, y));
        }
        let check = base.clone();
        let compact = check
            .solve_reduced(&rhs, &free)
            .expect("benchmark must exercise direct solve");
        let cg = cg_solve(&rhs, &free, &mask, &check);
        assert!(compact.iter().zip(cg).all(|(&a, b)| (a - b).abs() < 1e-6));
        for run in 0..3 {
            for hot in [false, true] {
                let solves = 500;
                let mut elapsed = [0_f64; 2];
                for direct in [false, true] {
                    let histories: Vec<_> = (0..if hot { 1 } else { solves })
                        .map(|_| base.clone())
                        .collect();
                    let solve = |m: &LbfgsMemory<f64>| {
                        if direct {
                            m.solve_reduced(black_box(&rhs), black_box(&free)).unwrap()
                        } else {
                            cg_solve(black_box(&rhs), black_box(&free), black_box(&mask), m)
                        }
                    };
                    if hot {
                        black_box(solve(&histories[0]));
                    }
                    let start = Instant::now();
                    for i in 0..solves {
                        black_box(solve(&histories[if hot { 0 } else { i }]));
                    }
                    elapsed[usize::from(direct)] = start.elapsed().as_secs_f64();
                }
                println!(
                    "run={run} n={n} m=17 free=437 condition={condition} cache={} solves={solves} cg_seconds={:.6} compact_seconds={:.6} speedup={:.2}",
                    if hot { "reused" } else { "fresh" },
                    elapsed[0],
                    elapsed[1],
                    elapsed[0] / elapsed[1]
                );
            }
        }
    }
}
