//! Internal invariants; SciPy floatround fixture adapted under tests/SCIPY_LICENSE.txt.
use crate::{
    Bound,
    bounds::{feasible_step_max, project, projected_gradient},
    cauchy::generalized_cauchy_point,
    dot,
    memory::LbfgsMemory,
    sub,
    subspace::subspace_minimize,
};

#[test]
fn projected_gradient_clips_to_distance_from_bound() {
    // L-BFGS-B projgr uses the unit projected displacement, not only active-set masking.
    let x = [0.1_f64, 0.9, 0.5, 2., 3.];
    let g = [5., -5., 0.2, 5., -5.];
    let b = [
        Bound::both(0., 1.),
        Bound::both(0., 1.),
        Bound::both(0., 1.),
        Bound::lower(1.),
        Bound::upper(4.),
    ];
    let pg = projected_gradient(&x, &g, &b);
    for (v, e) in pg.iter().zip([0.1, -0.1, 0.2, 1., -1.]) {
        assert!((v - e).abs() < 1e-15, "{pg:?}");
    }
}

#[test]
fn bfgs_hessian_products_match_dense_updates_and_secant_equation() {
    for capacity in [1, 3, 8] {
        let mut memory = LbfgsMemory::new(capacity);
        let mut pairs = Vec::new();
        for k in 0..12 {
            let s: Vec<f64> = (0..4).map(|j| ((k * 7 + j * 3 + 1) as f64).sin()).collect();
            let y: Vec<f64> = s
                .iter()
                .enumerate()
                .map(|(j, &v)| v * (j + 1) as f64)
                .collect();
            assert!(memory.push(s.clone(), y.clone()));
            pairs.push((s.clone(), y.clone()));
            if pairs.len() > capacity {
                pairs.remove(0);
            }
            let mut dense = vec![vec![0_f64; 4]; 4];
            for (j, row) in dense.iter_mut().enumerate() {
                row[j] = memory.theta();
            }
            for (s, y) in &pairs {
                let bs: Vec<f64> = dense.iter().map(|row| dot(row, s)).collect();
                let sbs = dot(s, &bs);
                let ys = dot(y, s);
                for i in 0..4 {
                    for j in 0..4 {
                        dense[i][j] += y[i] * y[j] / ys - bs[i] * bs[j] / sbs;
                    }
                }
            }
            for v in [vec![1., 2., 3., 4.], s] {
                let actual = memory.hessian_product(&v);
                for (a, row) in actual.iter().zip(&dense) {
                    assert!((a - dot(row, &v)).abs() < 1e-11);
                }
                assert!(dot(&v, &actual) > 0.);
                let inverse = memory.inverse_hessian_product(&actual);
                for (a, b) in inverse.iter().zip(&v) {
                    assert!((a - b).abs() < 1e-10);
                }
            }
            let bs = memory.hessian_product(&pairs.last().unwrap().0);
            for (a, b) in bs.iter().zip(y) {
                assert!((a - b).abs() < 1e-11);
            }
        }
    }
}

#[test]
fn curvature_updates_are_scale_relative() {
    let mut memory = LbfgsMemory::new(3);
    assert!(memory.push(vec![1e-12_f64], vec![2e-12]));
    assert!((memory.theta() - 2.).abs() < 1e-15);
    assert!(!memory.push(vec![1.], vec![-1.]));
    assert!(!memory.push(vec![1.], vec![0.]));
    assert!(!memory.push(vec![1.], vec![f64::NAN]));
}

#[test]
fn cauchy_point_handles_tied_breakpoints_and_small_gradients() {
    let m = LbfgsMemory::new(3);
    let b = [
        Bound::both(0_f64, 1.),
        Bound::both(0., 1.),
        Bound::unbounded(),
    ];
    assert_eq!(
        generalized_cauchy_point(&[0.5, 0.5, 0.], &[-2., -2., -1e-10], &b, &m),
        vec![1., 1., 1e-10]
    );
}

#[test]
fn scipy_floatround_fixture_keeps_cauchy_subspace_and_trials_feasible() {
    // SciPy test_setulb_floatround has a callback defined at just four Fortran iterates.
    // Our line search follows a different trajectory: replay its exact (x,g) history
    // through our direction construction instead of inventing values off that history.
    let points: [[f64; 5]; 4] = [
        [
            0.8750000000000278,
            0.7500000000000153,
            0.9499999999999722,
            0.8214285714285992,
            0.6363636363636085,
        ],
        [1., 0., 1., 0., 0.],
        [1., 0., 0.9889733043149325, 0., 0.026353554421041155],
        [1., 0., 0.9889917442915558, 0., 0.020341986743231205],
    ];
    let gradients = [
        [
            -0.5934820547965749,
            1.6251549718258351,
            -71.99168459202559,
            5.346636965797545,
            37.10732723092604,
        ],
        [
            -0.43295349282641515,
            1.008607936794592,
            18.223666726602975,
            31.927010036981997,
            -19.667512518739386,
        ],
        [
            -0.4699874455100256,
            0.9466285353668347,
            -0.016874360242016825,
            48.44999161133457,
            5.819631620590712,
        ],
        [
            -0.46970678696829116,
            0.9612719312174818,
            0.006129809488833699,
            48.43557729419473,
            6.005481418498221,
        ],
    ];
    let bounds = [Bound::both(0., 1.); 5];
    let mut memory = LbfgsMemory::new(10);
    for i in 0..points.len() {
        if i > 0 {
            memory.push(
                sub(&points[i], &points[i - 1]),
                sub(&gradients[i], &gradients[i - 1]),
            );
        }
        let x = &points[i];
        let g = &gradients[i];
        let cauchy = generalized_cauchy_point(x, g, &bounds, &memory);
        let target = subspace_minimize(x, g, &cauchy, &bounds, &memory);
        for p in [&cauchy, &target] {
            assert!(p.iter().all(|&v| (0. ..=1.).contains(&v)));
        }
        let d = sub(&target, x);
        let max = feasible_step_max(x, &d, &bounds).min(1.);
        for step in [0., max / 3., max * 0.7, max] {
            let mut trial: Vec<_> = x.iter().zip(&d).map(|(&v, &di)| v + step * di).collect();
            project(&mut trial, &bounds);
            assert!(trial.iter().all(|&v| (0. ..=1.).contains(&v)));
        }
    }
}
