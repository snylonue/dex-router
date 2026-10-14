#!/usr/bin/env python3
"""Audit the observed terminal raw searches without evaluating the market callback."""
import json
import math
from pathlib import Path

from analyze_hermes_convergence import check_case, decimal, messages


def dot(a, b):
    # Match the Rust sequential binary64 fold, not Python's compensated sum.
    value = 0.0
    for x, y in zip(a, b):
        value = value + x * y
    return value


def main():
    root = Path(__file__).resolve().parents[2]
    runs = root / "target/hermes-repro/runs"
    label = "user-kernel-terminal-observer-raw-0"
    baseline = check_case(runs, "user-kernel-narrowed-raw-0")
    observed = check_case(runs, label)
    assert baseline["warm_price_hash"] == observed["warm_price_hash"]
    assert baseline["audit_sha256"] == observed["audit_sha256"]
    for key in ("iterations", "evaluations", "status", "objective", "projected_gradient_inf_norm"):
        assert baseline["warm"][key] == observed["warm"][key]
    lines = (runs / (label + "-process.log")).read_text().splitlines()
    searches = []
    for line in lines:
        if line.startswith("USER_KERNEL_SEARCH "):
            searches.append({**json.loads(line.split(" ", 1)[1]), "trials": []})
        elif line.startswith("USER_KERNEL_TRIAL "):
            searches[-1]["trials"].append(json.loads(line.split(" ", 1)[1]))
    terminal = messages(lines, "USER_KERNEL_TERMINAL")[-1]
    assert len(searches) == 2 and terminal["reason"] == "Some(TrialLimit)"
    summaries = []
    for search in searches:
        f0, d0 = search["f0"], search["derivative0"]
        x0, g0, direction = search["x0"], search["g0"], search["direction"]
        assert len(x0) == len(g0) == len(direction) == len(search["bounds"]) == 607
        assert dot(g0, direction) == d0 and d0 < 0
        assert len(search["trials"]) == 20
        trials = []
        for trial in search["trials"]:
            alpha, f, slope = trial["step"], trial["f"], trial["derivative"]
            assert alpha > 0 and dot(trial["g"], direction) == slope
            changed = 0
            for x, initial, ray, (lo, hi) in zip(trial["x"], x0, direction, search["bounds"]):
                expected = initial + alpha * ray
                if lo is not None:
                    expected = max(lo, expected)
                    assert x >= lo
                if hi is not None:
                    expected = min(hi, expected)
                    assert x <= hi
                assert math.isfinite(x) and x == expected
                changed += x != initial
            armijo = f <= f0 + 0.001 * alpha * d0
            curvature = abs(slope) <= -0.9 * d0
            endpoint = search["step_max"] is not None and alpha == search["step_max"] and slope < 0
            assert armijo == trial["armijo"]
            assert not (armijo and (curvature or endpoint))
            trials.append({
                "step": alpha, "objective_change_ulps": float((decimal(f) - decimal(f0)) / decimal(math.ulp(f0))),
                "predicted_change_ulps": float(decimal(alpha) * decimal(d0) / decimal(math.ulp(f0))),
                "derivative_ratio": slope / -d0, "armijo": armijo, "curvature": curvature,
                "changed_coordinates": changed, "bracket": trial["bracket"],
            })
        summaries.append({"f0": f0, "derivative0": d0, "direction_inf_norm": max(map(abs, direction)),
                          "initial_step": search["initial_step"], "trials": trials})
    print(json.dumps({
        "method": "Sequential binary64 recomputation of slopes, original Wolfe tests and rounded feasible points; Decimal objective changes in ulps",
        "scope": "Observation only, no extra callback calls, no accepted or rejected point replaced",
        "same_warm_counts_prices_and_final_DD_quotes_as_unobserved": True,
        "observer_binary_sha256": observed["binary_sha256"], "terminal": terminal,
        "remaining_issue": "Two 20-trial searches exhaust the unchanged budget. This alone proves neither that a better trial policy is impossible nor that a larger budget is the correct production fix.",
        "searches": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
