pub mod v2;
pub mod v3;

/// The nonnegative Taylor remainder of -2 sqrt(v0 v1), evaluated using
/// differences of prices rather than differences of square roots. On a fixed
/// active region a support function is linear minus C * 2 sqrt(v0 v1).
fn support_remainder(v: [f64; 2], reference: [f64; 2]) -> f64 {
    let roots = [v[0].sqrt(), v[1].sqrt()];
    let old = [reference[0].sqrt(), reference[1].sqrt()];
    let delta = [
        (v[0] - reference[0]) / (roots[0] + old[0]),
        (v[1] - reference[1]) / (roots[1] + old[1]),
    ];
    let ratio = (old[1] / old[0]).sqrt();
    (ratio * delta[0] - delta[1] / ratio).powi(2)
}
