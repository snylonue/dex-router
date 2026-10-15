//! Full-market native replay of offline precision probes; no solver acceptance changes.
use super::*;
use serde_json::{Value, json};
fn prices_from_bits(value: &Value) -> ndarray::Array1<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| f64::from_bits(u64::from_str_radix(v.as_str().unwrap(), 16).unwrap()))
        .collect()
}
#[test]
#[ignore = "replay high-precision/grid candidates using all original native pool quotes"]
fn inspect_price_resolution_probes() {
    let input: Value = serde_json::from_str(
        &std::fs::read_to_string("../target/newton_resolution_probes.json").unwrap(),
    )
    .unwrap();
    let pools: Vec<RawUniswapV3Pool> =
        serde_json::from_str(include_str!("../fixtures/uniswap_v3_pools.json")).unwrap();
    let mut records = Vec::new();
    for c in input.as_array().unwrap() {
        let full =
            build_full_market_route_with_tvl(pools.clone(), c["cutoff"].as_f64().unwrap()).unwrap();
        let o = SolveOptions {
            absolute_residual_value: c["tolerance"].as_f64().unwrap(),
            ..Default::default()
        };
        let baseline = prices_from_bits(&c["baseline_bits"]);
        for trial in c["points"].as_array().unwrap() {
            let p = prices_from_bits(&trial["price_bits"]);
            let decimal: Vec<f64> = serde_json::from_value(trial["price"].clone()).unwrap();
            let json_price_bit_mismatches: Vec<_> = decimal
                .iter()
                .zip(p.iter())
                .enumerate()
                .filter_map(|(i, (a, b))| (a.to_bits() != b.to_bits()).then_some(i))
                .collect();
            let json_price_bit_max_distance = decimal
                .iter()
                .zip(p.iter())
                .map(|(a, b)| a.to_bits().abs_diff(b.to_bits()))
                .max()
                .unwrap();
            assert_eq!(p.len(), full.route.tokens);
            let e = full.route.evaluate(&p);
            let check = super::laplacian_analysis::check(&full.route, &p, &o);
            if let Some(expected) = trial["mp_kkt"].as_f64() {
                let actual = check["kkt"].as_f64().unwrap();
                assert!(
                    (actual - expected).abs() < 1e-6 * (1.0 + expected),
                    "oracle/replay mismatch"
                );
            }
            let mut delta = (0.0, 0.0);
            let mut changed = 0;
            for (m, (a, b)) in &full.route.markets {
                let v = [p[*a], p[*b]];
                let reference = [baseline[*a], baseline[*b]];
                add(
                    &mut delta,
                    m.value_change(v, reference, m.evaluate(v), m.evaluate(reference)),
                );
                changed += usize::from(
                    (m.relative_curvature(v).unwrap() > 0.0)
                        != (m.relative_curvature(reference).unwrap() > 0.0),
                );
            }
            let record = json!({"snapshot":c["snapshot"],"label":trial["label"],"economic":check,
                "native_cost_change":delta.0+delta.1,"json_price_bit_mismatches":json_price_bit_mismatches,"json_price_bit_max_distance":json_price_bit_max_distance,"curvature_activity_changes":changed,"gradient":e.gradient.to_vec()});
            println!("RESOLUTION_RESULT {record}");
            records.push(record);
        }
    }
    std::fs::write(
        "../target/newton_resolution_native.json",
        serde_json::to_string_pretty(&records).unwrap(),
    )
    .unwrap();
}

#[test]
fn diagnostic_price_transport_preserves_all_binary64_bits() {
    let words = [
        1_u64,
        1.0_f64.to_bits(),
        1.0_f64.next_up().to_bits(),
        f64::MAX.to_bits(),
    ];
    let value = serde_json::to_value(words.map(|w| format!("{w:016x}"))).unwrap();
    let p = prices_from_bits(&value);
    for (actual, expected) in p.iter().zip(words) {
        assert_eq!(actual.to_bits(), expected);
    }
}
