use super::*;
use num_traits::FromPrimitive;

// The pre-cache implementation, deliberately independent of cached theta,
// cached curvatures, and the production update helper.
fn uncached_product<T: Float>(memory: &LbfgsMemory<T>, v: &[T]) -> Vec<T> {
    let theta = memory
        .pairs
        .last()
        .map_or(T::one(), |p| dot(&p.y, &p.y) / p.ys);
    let update = |out: &mut [T], original: &[T], pair: &Correction<T>, bs: &[T]| {
        let sbs = dot(&pair.s, bs);
        if sbs > T::zero() {
            let alpha = dot(&pair.s, out) / sbs;
            axpy(out, -alpha, bs);
        }
        if pair.ys > T::zero() {
            let alpha = dot(&pair.y, original) / pair.ys;
            axpy(out, alpha, &pair.y);
        }
    };
    let mut cache: Vec<Vec<T>> = Vec::with_capacity(memory.pairs.len());
    for pair in &memory.pairs {
        let mut bs: Vec<_> = pair.s.iter().map(|&x| theta * x).collect();
        for (previous, previous_bs) in memory.pairs.iter().zip(&cache) {
            update(&mut bs, &pair.s, previous, previous_bs);
        }
        cache.push(bs);
    }
    let mut out: Vec<_> = v.iter().map(|&x| theta * x).collect();
    for (pair, bs) in memory.pairs.iter().zip(&cache) {
        update(&mut out, v, pair, bs);
    }
    out
}

fn correction<T: Float + FromPrimitive>(n: usize, k: usize) -> (Vec<T>, Vec<T>) {
    let s: Vec<_> = (0..n)
        .map(|j| T::from_f64(((k * 7 + j * 3 + 1) as f64).sin()).unwrap())
        .collect();
    let y = s
        .iter()
        .enumerate()
        .map(|(j, &s)| s * T::from_usize(1 + (j + k) % 11).unwrap())
        .collect();
    (s, y)
}

fn compare_history<T: Float + FromPrimitive>(bits: impl Fn(T) -> u64) {
    for capacity in [1, 3, 17] {
        let mut memory = LbfgsMemory::new(capacity);
        for k in 0..40 {
            if k == 0 || k == 27 {
                memory.clear();
                assert_eq!(bits(memory.theta()), bits(T::one()));
                assert!(memory.hessian_cache.get().is_none());
                let v = vec![T::one(); 23];
                assert!(memory.hessian_product(&v) == v);
            }
            let (s, y) = correction(23, k);
            assert!(memory.push(s, y));
            assert!(memory.hessian_cache.get().is_none());
            // Inverse products need theta, but must not build the Hessian cache.
            memory.inverse_hessian_product(&vec![T::one(); 23]);
            assert!(memory.hessian_cache.get().is_none());
            let theta = memory.pairs.last().map(|p| dot(&p.y, &p.y) / p.ys).unwrap();
            assert_eq!(bits(memory.theta()), bits(theta));
            for probe in 0..5 {
                let (v, _) = correction(23, probe + k);
                let expected = uncached_product(&memory, &v);
                let actual = memory.hessian_product(&v);
                assert_eq!(
                    actual.into_iter().map(&bits).collect::<Vec<_>>(),
                    expected.into_iter().map(&bits).collect::<Vec<_>>()
                );
            }
            let cache = memory.hessian_cache.get().unwrap();
            assert_eq!(cache.len(), memory.pairs.len());
            let address = cache.as_ptr();
            // Rejected curvature must preserve both scale and cached history.
            assert!(!memory.push(vec![T::one(); 23], vec![-T::one(); 23]));
            assert!(!memory.push(vec![T::zero(); 23], vec![T::zero(); 23]));
            assert!(!memory.push(vec![T::one(); 23], vec![T::nan(); 23]));
            assert!(!memory.push(vec![T::one(); 23], vec![T::infinity(); 23]));
            assert_eq!(bits(memory.theta()), bits(theta));
            assert_eq!(memory.hessian_cache.get().unwrap().as_ptr(), address);
            memory.hessian_product(&vec![T::one(); 23]);
            assert_eq!(memory.hessian_cache.get().unwrap().as_ptr(), address);
        }
    }
}

#[test]
fn cached_products_match_original_bitwise_f64() {
    compare_history(f64::to_bits);
}

#[test]
fn cached_products_match_original_bitwise_f32() {
    compare_history(|value: f32| u64::from(value.to_bits()));
}

#[test]
fn empty_and_zero_capacity_memory_are_identity() {
    let mut memory = LbfgsMemory::<f64>::new(0);
    for v in [vec![], vec![2., -3.], vec![1., 2., 3.]] {
        assert_eq!(memory.hessian_product(&v), v);
        assert_eq!(memory.inverse_hessian_product(&v), v);
    }
    assert!(!memory.push(vec![1.], vec![2.]));
    assert!(memory.is_empty());
    assert_eq!(memory.theta(), 1.);
    memory.clear();
    assert!(memory.hessian_cache.get().is_none());
}

#[test]
fn cloned_caches_and_memory_resets_are_independent() {
    let mut original = LbfgsMemory::new(3);
    let (s, y) = correction::<f64>(4, 0);
    assert!(original.push(s, y));
    let mut uninitialized_clone = original.clone();
    let v = [1., 2., 3., 4.];
    let expected = original.hessian_product(&v);
    assert!(uninitialized_clone.hessian_cache.get().is_none());
    assert_eq!(uninitialized_clone.hessian_product(&v), expected);
    let mut initialized_clone = original.clone();
    original.clear();
    assert_eq!(original.hessian_product(&v), v);
    for clone in [&mut uninitialized_clone, &mut initialized_clone] {
        assert_eq!(clone.hessian_product(&v), expected);
        let (s, y) = correction(4, 1);
        assert!(clone.push(s, y));
        assert!(clone.hessian_cache.get().is_none());
        assert_eq!(clone.hessian_product(&v), uncached_product(clone, &v));
    }
}

#[test]
fn lazy_cache_supports_concurrent_shared_products() {
    let mut memory = LbfgsMemory::new(17);
    for k in 0..17 {
        let (s, y) = correction::<f64>(23, k);
        assert!(memory.push(s, y));
    }
    let v = vec![1.; 23];
    let expected = uncached_product(&memory, &v);
    assert!(memory.hessian_cache.get().is_none());
    let barrier = std::sync::Barrier::new(8);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                barrier.wait();
                for _ in 0..20 {
                    assert_eq!(memory.hessian_product(&v), expected);
                }
            });
        }
    });
    assert_eq!(memory.hessian_cache.get().unwrap().len(), 17);
}

#[test]
#[ignore = "release-only Hessian cache microbenchmark against original implementation"]
fn benchmark_repeated_hessian_products() {
    use std::{hint::black_box, time::Instant};
    let mut memory = LbfgsMemory::new(17);
    for k in 0..17 {
        let (s, y) = correction::<f64>(438, k);
        assert!(memory.push(s, y));
    }
    let v: Vec<_> = (0..438).map(|j| (j as f64).cos()).collect();
    let expected = uncached_product(&memory, &v);
    let products = 62_854;
    for run in 0..3 {
        memory.hessian_cache.take();
        let mut rebuilds = 0;
        let start = Instant::now();
        for _ in 0..products {
            rebuilds += usize::from(memory.hessian_cache.get().is_none());
            black_box(memory.hessian_product(black_box(&v)));
        }
        let cached_seconds = start.elapsed().as_secs_f64();
        assert_eq!(rebuilds, 1);
        assert_eq!(memory.hessian_product(&v), expected);
        let start = Instant::now();
        for _ in 0..products {
            black_box(uncached_product(black_box(&memory), black_box(&v)));
        }
        let uncached_seconds = start.elapsed().as_secs_f64();
        println!(
            "run={run} n=438 m=17 products={products} cached_rebuilds={rebuilds} uncached_rebuilds={products} cached_seconds={cached_seconds:.6} uncached_seconds={uncached_seconds:.6} speedup={:.2}",
            uncached_seconds / cached_seconds
        );
    }
}
