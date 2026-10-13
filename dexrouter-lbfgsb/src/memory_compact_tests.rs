use super::*;
use num_traits::FromPrimitive;

fn correction<T: Float + FromPrimitive>(n: usize, k: usize, scale: f64) -> (Vec<T>, Vec<T>) {
    let s: Vec<_> = (0..n)
        .map(|j| T::from_f64(scale * ((k * 7 + j * 3 + 1) as f64).sin()).unwrap())
        .collect();
    let y = s
        .iter()
        .enumerate()
        .map(|(j, &v)| v * T::from_usize(1 + (j + k) % 11).unwrap())
        .collect();
    (s, y)
}

// Independent dense principal-matrix solve, assembled from original BFGS
// products. This checks both sides of the free/active Gram assembly shortcut.
fn dense_solve<T: Float>(memory: &LbfgsMemory<T>, rhs: &[T], free: &[usize]) -> Vec<f64> {
    let n = free.len();
    let mut matrix = vec![vec![0_f64; n]; n];
    for (j, &column) in free.iter().enumerate() {
        let mut unit = vec![T::zero(); rhs.len()];
        unit[column] = T::one();
        let product = memory.hessian_product(&unit);
        for (i, &row) in free.iter().enumerate() {
            matrix[i][j] = product[row].to_f64().unwrap();
        }
    }
    let mut answer: Vec<_> = free.iter().map(|&i| rhs[i].to_f64().unwrap()).collect();
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| matrix[i][k].abs().total_cmp(&matrix[j][k].abs()))
            .unwrap();
        matrix.swap(k, pivot);
        answer.swap(k, pivot);
        for i in k + 1..n {
            let ratio = matrix[i][k] / matrix[k][k];
            for j in k + 1..n {
                matrix[i][j] -= ratio * matrix[k][j];
            }
            answer[i] -= ratio * answer[k];
        }
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            answer[i] -= matrix[i][j] * answer[j];
        }
        answer[i] /= matrix[i][i];
    }
    answer
}

fn check_models<T: Float + FromPrimitive + std::fmt::Debug>() {
    let n = 23;
    for capacity in [1, 3, 17] {
        for scale in [1e-12, 1., 1e12] {
            let mut memory = LbfgsMemory::new(capacity);
            for k in 0..28 {
                let (s, y) = correction::<T>(n, k, scale);
                assert!(memory.push(s, y));
                assert!(memory.compact_cache.get().is_none());
                let masks = vec![
                    vec![0],
                    vec![0, 2, 4],
                    (1..n).collect(),
                    (0..n).step_by(2).collect(),
                ];
                for free in masks {
                    let rhs: Vec<_> = (0..n)
                        .map(|j| T::from_f64(((k + j) as f64).cos()).unwrap())
                        .collect();
                    let actual = memory
                        .solve_reduced(&rhs, &free)
                        .expect("well-conditioned history must use direct solve");
                    let expected = dense_solve(&memory, &rhs, &free);
                    let product = memory.hessian_product(&actual);
                    for (local, &i) in free.iter().enumerate() {
                        let tolerance = T::epsilon().sqrt().to_f64().unwrap();
                        assert!(
                            (actual[i].to_f64().unwrap() - expected[local]).abs()
                                <= tolerance * expected[local].abs().max(1.)
                        );
                        assert!((product[i] - rhs[i]).abs() <= T::epsilon().sqrt());
                    }
                    for i in 0..n {
                        if !free.contains(&i) {
                            assert_eq!(actual[i], T::zero());
                        }
                    }
                }
                let address = memory.compact_cache.get().unwrap().as_ref().unwrap() as *const _;
                assert!(!memory.push(vec![T::one(); n], vec![-T::one(); n]));
                assert_eq!(
                    memory.compact_cache.get().unwrap().as_ref().unwrap() as *const _,
                    address
                );
            }
            memory.clear();
            assert!(memory.compact_cache.get().is_none());
            assert_eq!(
                memory.solve_reduced(&vec![T::one(); n], &[0, 2]).unwrap()[0],
                T::one()
            );
        }
    }
}

#[test]
fn compact_matches_dense_reduced_models_f64() {
    check_models::<f64>();
}
#[test]
fn compact_matches_dense_reduced_models_f32() {
    check_models::<f32>();
}

#[test]
fn compact_cache_clones_and_shared_solves_are_independent() {
    let mut memory = LbfgsMemory::new(17);
    for k in 0..20 {
        let (s, y) = correction::<f64>(23, k, 1.);
        assert!(memory.push(s, y));
    }
    let rhs = vec![1.; 23];
    let masks: Vec<Vec<usize>> = vec![(1..23).collect(), (0..23).step_by(2).collect()];
    let expected: Vec<_> = masks
        .iter()
        .map(|free| memory.solve_reduced(&rhs, free).unwrap())
        .collect();
    let mut cloned = memory.clone();
    std::thread::scope(|scope| {
        for worker in 0..8 {
            let masks = &masks;
            let expected = &expected;
            let memory = &memory;
            let rhs = &rhs;
            scope.spawn(move || {
                for repeat in 0..20 {
                    let i = (worker + repeat) % 2;
                    assert_eq!(memory.solve_reduced(rhs, &masks[i]).unwrap(), expected[i]);
                }
            });
        }
    });
    memory.clear();
    assert!(memory.compact_cache.get().is_none());
    assert_eq!(cloned.solve_reduced(&rhs, &masks[0]).unwrap(), expected[0]);
    let (s, y) = correction(23, 22, 1.);
    assert!(cloned.push(s, y));
    assert!(cloned.compact_cache.get().is_none());
    assert!(cloned.solve_reduced(&rhs, &masks[0]).is_some());
}

#[test]
fn invalid_compact_model_falls_back_to_cg_and_preserves_model_decrease() {
    // Check both an unavailable factor and a finite but incorrect low-rank
    // reconstruction. The latter must fail the original-model residual check,
    // including refinement, instead of silently accepting its internal solve.
    for compact in [None, CompactModel::new(vec![vec![0_f64; 2]; 2])] {
        let mut memory = LbfgsMemory::new(1);
        assert!(memory.push(vec![1_f64, 0.], vec![1., 2.]));
        memory.compact_cache.set(compact).unwrap();
        assert!(memory.solve_reduced(&[0., -4.5], &[1]).is_none());
        let result = crate::subspace::subspace_minimize(
            &[0., 0.],
            &[0., 9.],
            &[0., -0.5],
            &[crate::Bound::both(0., 0.), crate::Bound::unbounded()],
            &memory,
        );
        assert_eq!(result, [0., -1.]);
    }
}

#[test]
fn free_space_shortcut_and_zero_rhs_do_not_build_compact_cache() {
    let mut memory = LbfgsMemory::new(1);
    assert!(memory.push(vec![1_f64, 0.], vec![1., 2.]));
    assert_eq!(memory.solve_reduced(&[0., 0.], &[1]).unwrap(), [0., 0.]);
    let bounds = [crate::Bound::unbounded(); 2];
    let cauchy = crate::cauchy::generalized_cauchy_point(&[0., 0.], &[1., 2.], &bounds, &memory);
    crate::subspace::subspace_minimize(&[0., 0.], &[1., 2.], &cauchy, &bounds, &memory);
    assert!(memory.compact_cache.get().is_none());
    assert!(memory.solve_reduced(&[1., 1.], &[2]).is_none());
    assert!(memory.solve_reduced(&[1., 1.], &[1, 0]).is_none());
    assert!(memory.solve_reduced(&[1., f64::NAN], &[0]).is_none());
}
