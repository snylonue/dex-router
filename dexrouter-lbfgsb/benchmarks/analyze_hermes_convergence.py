#!/usr/bin/env python3
"""Recheck recorded Hermes trajectory experiments; never quote markets or rerun a solve.

Requires the extracted offline reproduction's runs and frozen manifest under target/.
Prints JSON to stdout; redirect to a NEW file rather than overwrite an earlier audit.
"""
import argparse
import ast
from collections import Counter
from decimal import Decimal, getcontext
import hashlib
import json
import math
from pathlib import Path
import statistics

getcontext().prec = 100


def load(path):
    return json.loads(path.read_text())


def decimal(value):
    assert math.isfinite(value)
    return Decimal.from_float(value)


def pair(parts):
    assert len(parts) == 2
    return decimal(parts[0]) + decimal(parts[1])


def messages(lines, tag):
    return [json.loads(line[len(tag) + 1:]) for line in lines if line.startswith(tag + " ")]


def check_case(runs, label):
    receipt = load(runs / (label + "-process.json"))
    assert receipt["input_hashes_unchanged"] and receipt["environment_cleared"]
    assert not receipt["timed_out"] and receipt["invocations"] == 1
    assert receipt["rpc_requests"] == receipt["evm_simulations"] == 0
    assert not receipt["dd_recovery"] and not receipt["dd_additive"]
    audit = load(runs / (label + "-dd-audit.json"))
    n = len(audit["prices"])
    assert n == 607 and len(audit["samples"]) == 994
    incoming, outgoing = [Decimal(0)] * n, [Decimal(0)] * n
    for sample in audit["samples"]:
        for side, token in enumerate(sample["tokens"]):
            i, o = pair(sample["input"][side]), pair(sample["output"][side])
            assert i >= 0 and o >= 0
            incoming[token] += o
            outgoing[token] += i
    relative = [abs(i - o) / max(i, o) if max(i, o) else Decimal(0)
                for i, o in zip(incoming, outgoing)]
    bad = [i for i in range(1, n) if relative[i] > decimal(1e-9)]
    scaled = max(abs(incoming[i] - outgoing[i]) * decimal(audit["scales"][i])
                 for i in range(1, n))
    floors = sum(pair(p) < pair(f) for p, f in zip(audit["prices"], audit["floors"]))
    valid = (not bad and scaled <= decimal(1e-9) and not floors
             and all(pair(p) > 0 for p in audit["prices"])
             and pair(audit["prices"][0]) == Decimal(1))
    assert bad == audit["metrics"]["strict_bad_tokens"]
    assert valid == (audit["status"] == "strict_and_gradient_satisfied")
    lines = (runs / (label + "-process.log")).read_text().splitlines()
    warm = (messages(lines, "HERMES_RAW") or messages(lines, "HERMES_EQUIVALENT"))[0]
    fingerprint = messages(lines, "HERMES_WARM_PRICE_HASH")[0]["hash"]
    timing = load(runs / (label + "-timing.json"))
    counters = Counter()
    for line in lines:
        if line.startswith("HERMES_KERNEL_AUDIT "):
            counters.update(ast.literal_eval(line.split(" counters=", 1)[1].split(" nanoseconds=", 1)[0]))
    events = [json.loads(line.split("diagnostics=", 1)[1]) for line in lines if "diagnostics=" in line]
    production_status = None
    if receipt["returncode"] == 0:
        production_status = load(runs / label / "admission.json")["status"]
    return {"case": label, "backend": receipt["backend"], "variant": receipt["variant"],
            "binary_sha256": receipt["binary_sha256"], "callback_trace": receipt["trace"],
            "warm_iterations_cap": receipt.get("warm_iterations_cap"),
            "audit_sha256": hashlib.sha256((runs / (label + "-dd-audit.json")).read_bytes()).hexdigest(),
            "returncode": receipt["returncode"], "warm": warm, "warm_price_hash": fingerprint,
            "process_wall_ms": receipt["process_wall_ms"], "solve_wall_ms": timing["solve_wall_ms"],
            "graph_stage_ms": timing["graph_stage_ms"],
            "dd_updates": audit["iteration"], "dd_status": audit["status"],
            "strict_acceptance_passed": valid, "bad_tokens": bad,
            "max_relative_flow_residual": str(max(relative[1:])),
            "scaled_gradient_max": str(scaled), "floor_violations": floors,
            "production_status": production_status, "kernel_counters": dict(counters),
            "milestones": messages(lines, "HERMES_THRESHOLD"),
            "restarts": messages(lines, "HERMES_RESTART"),
            "terminal_search": messages(lines, "HERMES_TERMINAL_SEARCH"),
            "dd_line_search_trials": sum(len(e.get("step", {}).get("trials", [])) for e in events),
            "dd_budget_progress": [{"iteration": e["iteration"],
                "bad_token_count": len(e["metrics"]["strict_bad_tokens"]),
                "scaled_gradient": e["metrics"]["original_scaled_gradient_max"],
                "direction_max": e.get("step", {}).get("relative_direction_max"),
                "dominant_direction_tokens": e.get("step", {}).get("factor", {}).get("dominant_direction_tokens", [])[:2]}
                for e in events if receipt.get("warm_iterations_cap") == 1000
                and e["iteration"] in (0, 6, 15, 31)],
            "receipt": str(runs / (label + "-process.json")),
            "audit": str(runs / (label + "-dd-audit.json"))}


def trace_summary(runs, labels):
    traces = {label: [json.loads(line) for line in (runs / (label + "-fg-trace.jsonl")).read_text().splitlines()]
              for label in labels}
    assert all(a[0] == next(iter(traces.values()))[0] for a in traces.values())
    rows = []
    for label, a in traces.items():
        d = [v - u for u, v in zip(a[0]["x"], a[1]["x"])]
        rows.append({"case": label, "f0": a[0]["f"], "first_f": a[1]["f"], "second_trial_f": a[2]["f"],
                     "first_displacement_inf": max(map(abs, d)), "first_displacement_l2": math.hypot(*d)})
    full, reduced = traces[labels[0]], traces[labels[1]]
    s = [v - u for u, v in zip(full[0]["x"], full[2]["x"])]
    y = [v - u for u, v in zip(full[0]["g"], full[2]["g"])]
    slope0 = sum(g * d for g, d in zip(full[0]["g"], s))
    slope2 = sum(g * d for g, d in zip(full[2]["g"], s))
    assert full[2]["f"] <= full[0]["f"] + 0.001 * slope0
    assert abs(slope2) <= -0.9 * slope0
    ys = sum(v * w for v, w in zip(y, s))
    yy = sum(v * v for v in y)
    yy_free = sum(v * v for v in y[1:])
    assert s[0] == 0
    first_difference = next(i for i, (a, b) in enumerate(zip(full, reduced)) if a != b)
    return {"all_initial_callbacks_bitwise_equal": True, "cases": rows,
            "first_accepted_native_update": {"original_wolfe_verified": True,
                "fixed_coordinate_step": s[0], "fixed_coordinate_gradient_change": y[0],
                "ys": ys, "yy_full": yy, "yy_free": yy_free,
                "theta_full_reconstructed": yy / ys, "theta_free_reconstructed": yy_free / ys,
                "theta_ratio": yy / yy_free},
            "native_full_reduced_first_different_evaluation": first_difference,
            "native_full_reduced_x_difference_inf": max(abs(a - b) for a, b in
                zip(full[first_difference]["x"], reduced[first_difference]["x"]))}


def boundary_summary(runs, label):
    lines = (runs / (label + "-process.log")).read_text().splitlines()
    event = next(json.loads(line.split("diagnostics=", 1)[1]) for line in lines if "diagnostics=" in line)
    step = event["step"]
    inc = step["trials"][0]["dominant_objective_increases"][0]
    assert inc["market_index"] == 406 and inc["tokens"] == [0, 264]
    p = pair(inc["original_prices"][1])
    upper = pair(inc["original_no_arb_ratio"][1])
    direction = pair(inc["relative_direction"][1])
    band_floor = Decimal(1) / upper
    activation_alpha = (band_floor / p - Decimal(1)) / direction
    last = step["trials"][-1]
    last_alpha = pair(last["alpha"])
    return {"case": label, "initial_bad_tokens": len(event["metrics"]["strict_bad_tokens"]),
            "token": 264, "activating_market": 406,
            "original_curvature_diagonal": step["factor"]["dominant_direction_tokens"][0]["original_curvature_diagonal"],
            "newton_relative_direction": str(direction),
            "relative_distance_to_activation": str(Decimal(1) - band_floor / p),
            "alpha_to_activation": str(activation_alpha),
            "trial_count": len(step["trials"]),
            "feasible_trials": sum(t["feasible"] for t in step["trials"]),
            "armijo_trials": sum(t.get("armijo", False) for t in step["trials"]),
            "last_alpha": str(last_alpha), "last_alpha_over_activation": str(last_alpha / activation_alpha),
            "first_halving_index_inside_original_region": next(k for k in range(100) if Decimal(2) ** (-k) < activation_alpha),
            "first_trial_increase_from_market_406": str(pair(inc["objective_increase"])),
            "first_trial_new_market_weight": inc["trial_weight"],
            "last_trial_actual_objective_change": str(pair(last["objective"]) - pair(event["metrics"]["objective"])),
            "last_trial_predicted_change": str(last_alpha * pair(step["factor"]["directional_derivative"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("target/hermes-repro/runs"))
    args = parser.parse_args()
    runs = args.runs
    repro = runs.parent / "hermes-minimize-integration-20261013"
    manifest = load(repro / "input/pools-1000/manifest.json")
    for record in manifest["inputs"].values():
        assert hashlib.sha256((repro / record["path"]).read_bytes()).hexdigest() == record["sha256"]
    matrix = [("rust-raw", "baseline"), ("rust-raw", "rust-l2"), ("rust-raw", "rust-c-search"),
              ("rust-raw", "rust-l2-c-search"), ("rust-raw", "rust-l2-c-search-cauchy"),
              ("c-raw", "baseline"), ("c-raw", "c-reduced"),
              ("rust-preprocessed", "baseline"), ("rust-preprocessed", "rust-c-search")]
    cases = [check_case(runs, "trajectory-" + b + "-" + v) for b, v in matrix]
    groups = []
    group_specs = [(b, v, None, "milestone-" + b + "-" + v) for b, v in matrix
                   if v not in ("rust-l2", "rust-l2-c-search-cauchy", "c-reduced")]
    group_specs.extend(("rust-raw", "baseline", cap, "budget-rust-raw-" + str(cap))
                       for cap in (250, 500, 1000, 2000))
    for backend, variant, cap, prefix in group_specs:
        a = [check_case(runs, prefix + "-" + str(rep)) for rep in range(3)]
        assert all(x["warm_iterations_cap"] == cap for x in a)
        assert all(not x["callback_trace"] for x in a)
        assert len({x["warm_price_hash"] for x in a}) == 1
        assert len({(x["warm"]["iterations"], x["warm"]["evaluations"], x["strict_acceptance_passed"], x["dd_updates"])
                    for x in a}) == 1
        assert len({hashlib.sha256((runs / (x["case"] + "-dd-audit.json")).read_bytes()).hexdigest() for x in a}) == 1
        row = {"backend": backend, "variant": variant, "warm_iterations_cap": cap, "repetitions": 3,
               "warm_price_hash": a[0]["warm_price_hash"], "iterations": a[0]["warm"]["iterations"],
               "evaluations": a[0]["warm"]["evaluations"], "status": a[0]["warm"]["status"],
               "objective": a[0]["warm"]["objective"], "projected_gradient_inf_norm": a[0]["warm"].get("projected_gradient_inf_norm"),
               "strict_acceptance_passed": a[0]["strict_acceptance_passed"], "dd_updates": a[0]["dd_updates"],
               "max_relative_flow_residual": a[0]["max_relative_flow_residual"], "production_status": a[0]["production_status"],
               "terminal_audit_bytes_equal_across_repeats": True, "milestones": []}
        for field in ["warm_ms", "solve_wall_ms", "process_wall_ms"]:
            values = [x["warm"]["elapsed_ms"] if field == "warm_ms" else x[field] for x in a]
            row[field] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
        for point in a[0]["milestones"]:
            times = [next(p["elapsed_ms"] for p in x["milestones"] if p["threshold"] == point["threshold"]) for x in a]
            row["milestones"].append({**{k: v for k, v in point.items() if k != "elapsed_ms"},
                                      "elapsed_ms_median": statistics.median(times)})
        groups.append(row)
        cases.extend(a)
    print(json.dumps({"scope": "Offline diagnosis only; production Rust algorithms unchanged; native predictor is not a production dependency.",
                      "source_commit": "ea10f08ef7398758a515569d97f6d77cf2c39668", "strict_tolerance": 1e-9, "absolute_flow_slack": 0,
                      "production_regression_tests": {"command": "RUSTFLAGS='-C link-arg=-fuse-ld=bfd' cargo test --release --offline -p dexrouter-lbfgsb -p dexrouter-optim", "passed": 132, "failed": 0, "ignored": 11},
                      "method": "Decimal precision 100; reconstruct recorded DD hi+lo quotes, no extra market evaluations.",
                      "timing_warning": "Three serial repeats, diagnostic counters/logs enabled; objective milestones are not KKT or convergence certificates.",
                      "inputs": manifest["inputs"], "serial_repeat_summary": groups,
                      "early_trajectory": trace_summary(runs, ["trajectory-c-raw-baseline", "trajectory-c-raw-c-reduced",
                          "trajectory-rust-raw-baseline", "trajectory-rust-raw-rust-l2", "trajectory-rust-raw-rust-l2-c-search"]),
                      "dd_initial_region_boundaries": [boundary_summary(runs, label) for label in ["c-preprocessed-audit", "712-audit"]],
                      "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
