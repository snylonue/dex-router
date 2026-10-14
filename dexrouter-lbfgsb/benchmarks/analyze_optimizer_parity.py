#!/usr/bin/env python3
"""Audit frozen-input C parity: clean repeats, strict quotes, and accepted milestones.

No market calls or optimizer invocations. Requires preserved target/ artifacts.
Use --output NEW_PATH; --check-case LABEL --require-strict also rejects negatives.
"""
import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import statistics

from analyze_hermes_convergence import load, messages, pair
from analyze_user_kernel_followup import check_run

BACKENDS = ("rust-raw", "c-raw", "rust-preprocessed", "c-preprocessed")
GOALS = {"f100": (100., None), "f10": (10., None), "f4": (4., None),
         "f373": (3.73, None), "f362": (3.62, None), "common": (3.62, .1),
         "c-terminal": (3.6159961104074276, .03944065965352126),
         "f350": (3.5, None), "pgtol": (None, 1e-9)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def state(row):
    return {k: row[k] for k in ("iteration", "evaluations", "f", "pg")}


def hit(row, goal):
    f, g = goal
    return (f is None or row["f"] <= f) and (g is None or row["pg"] <= g)


def warm_without_time(row):
    return {k: v for k, v in row["warm"].items() if k != "elapsed_ms"}


def quote_values(path):
    audit = load(path)
    net = [Decimal(0) for _ in audit["prices"]]
    for sample in audit["samples"]:
        for side, token in enumerate(sample["tokens"]):
            net[token] += pair(sample["output"][side]) - pair(sample["input"][side])
    return {"base_net": str(net[0]), "quote_dual": str(sum(
        (pair(price)*value for price, value in zip(audit["prices"], net)), Decimal(0)))}


def checked(runs, label, expected_binary=None):
    row = check_run(runs, label)
    receipt = load(runs / (label + "-process.json"))
    assert receipt["environment_cleared"]
    assert sha(Path(receipt["argv"][0])) == row["binary_sha256"]
    assert not row["callback_trace"] and row["warm_iterations_cap"] is None
    if expected_binary:
        assert row["binary_sha256"] == expected_binary
        assert row["variant"] == "baseline"
    if receipt["returncode"] == 0:
        admission = load(runs / label / "admission.json")
        baseline = admission.get("baseline_estimate") or {}
        row["execution_admission"] = {"status": admission["status"],
            "profit_basis": admission.get("profit_basis"),
            "net_profit": (admission.get("profit_breakdown") or {}).get("net_profit"),
            "baseline_diagnostic_gross": (baseline.get("profit_breakdown") or {}).get("gross_profit"),
            "baseline_rejection": baseline.get("rejection")}
    if row["independent_quote_check_available"]:
        row["quote_values"] = quote_values(runs / (label + "-dd-audit.json"))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-strict", action="store_true")
    parser.add_argument("--check-case")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    runs = root / "target/hermes-repro/runs"
    artifacts = runs.parent / "parity-followup"
    repro = runs.parent / "hermes-minimize-integration-20261013"
    build = load(artifacts / "final-build.json")
    for name, expected in build["source_sha256"].items():
        assert sha(root / "dexrouter-lbfgsb/src" / name) == expected, name
        assert sha(repro / "vendor/dexrouter-lbfgsb/src" / name) == expected, name
    observer_build = load(artifacts / "observer-build.json")
    sparse_build = load(artifacts / "sparse-observer-build.json")
    for manifest, prefix in [(observer_build, "observer"), (sparse_build, "sparse-observer")]:
        assert sha(root / manifest["binary"]) == manifest["binary_sha256"]
        solver_hash = manifest.get("solver_sha256") or manifest["source_sha256"]["solver.rs"]
        assert sha(artifacts / (prefix + "-solver.rs")) == solver_hash
        assert sha(artifacts / (prefix + "-repro_native.rs")) == manifest["bridge_sha256"]
    assert sha(repro / "vendor/minimize-only-routing/src/repro_native.rs") == observer_build["original_bridge_sha256"]
    assert len(build["source_sha256"]) == 13
    assert sha(root / build["binary"]) == build["binary_sha256"]
    assert sha(root / "target/release/hermes_minimize_snapshot") == build["binary_sha256"]
    inputs = load(repro / "input/pools-1000/manifest.json")["inputs"]
    for record in inputs.values():
        assert sha(repro / record["path"]) == record["sha256"]
    if args.check_case:
        row = checked(runs, args.check_case)
        encoded = json.dumps(row, indent=2) + "\n"
        if args.output:
            with args.output.open("x") as stream: stream.write(encoded)
        else:
            print(encoded, end="")
        return int(args.require_strict and not row["strict_acceptance_passed"])

    clean = {b: [checked(runs, f"parity-final-{b}-{i}", build["binary_sha256"])
                 for i in range(5)] for b in BACKENDS}
    summaries = {}
    for backend, rows in clean.items():
        assert all(warm_without_time(r) == warm_without_time(rows[0]) for r in rows)
        assert len({r["warm_price_hash"] for r in rows}) == 1
        assert len({r["audit_sha256"] for r in rows}) == 1
        summaries[backend] = {
            "repeats": len(rows), "strict_passes": sum(r["strict_acceptance_passed"] is True for r in rows),
            "warm_ms": stats([r["warm"]["elapsed_ms"] for r in rows]),
            "solve_ms": stats([r["solve_wall_ms"] for r in rows]),
            "warm": warm_without_time(rows[0]), "dd_updates": rows[0]["dd_updates"],
            "max_relative_flow_residual": rows[0]["max_relative_flow_residual"],
            "quote_values": rows[0]["quote_values"],
            "production_admission_status": rows[0].get("production_status"),
            "execution_admission": rows[0].get("execution_admission"),
            "warm_price_hash": rows[0]["warm_price_hash"], "audit_sha256": rows[0]["audit_sha256"]}

    full = {}
    observers = []
    milestones = {}
    sparse_path = artifacts / "sparse-observer-binary"
    for backend in BACKENDS[:2]:
        control = clean[backend][0]
        observed = checked(runs, f"parity-observer-{backend}-0")
        assert warm_without_time(observed) == warm_without_time(control)
        assert observed["warm_price_hash"] == control["warm_price_hash"]
        assert observed["audit_sha256"] == control["audit_sha256"]
        lines = (runs / (observed["case"] + "-process.log")).read_text().splitlines()
        accepted = messages(lines, "HERMES_ACCEPT")
        assert len(accepted) == control["warm"]["iterations"]
        assert [r["iteration"] for r in accepted] == list(range(1, len(accepted)+1))
        assert all(r["dimensions"] == 607 and math.isfinite(r["f"]) and math.isfinite(r["pg"]) for r in accepted)
        full[backend] = accepted
        observers.append(observed)
        milestones[backend] = {goal: [] for goal in GOALS}
        for i in range(5):
            row = checked(runs, f"parity-sparse-{backend}-{i}", sha(sparse_path))
            assert warm_without_time(row) == warm_without_time(control)
            assert row["warm_price_hash"] == control["warm_price_hash"]
            assert row["audit_sha256"] == control["audit_sha256"]
            records = messages((runs / (row["case"] + "-process.log")).read_text().splitlines(), "HERMES_MILESTONE")
            assert len({r["goal"] for r in records}) == len(records)
            actual = {r["goal"]: r for r in records}
            for goal, threshold in GOALS.items():
                first = next((r for r in accepted if hit(r, threshold)), None)
                assert (goal in actual) == (first is not None)
                if first is not None:
                    assert state(actual[goal]) == state(first), (backend, goal)
                    milestones[backend][goal].append(actual[goal])
            observers.append(row)
    assert GOALS["c-terminal"] == (full["c-raw"][-1]["f"], full["c-raw"][-1]["pg"])
    comparisons = {}
    for goal, threshold in GOALS.items():
        item = {"objective_at_most": threshold[0], "projected_gradient_at_most": threshold[1]}
        for backend in BACKENDS[:2]:
            records = milestones[backend][goal]
            item[backend] = ({"first_accepted_state": state(records[0]),
                              "sparse_observer_ms": stats([r["elapsed_ms"] for r in records])}
                             if records else None)
        if all(item[b] is not None for b in BACKENDS[:2]):
            item["paired_c_over_rust_time_ratio_median"] = statistics.median(
                c["elapsed_ms"] / r["elapsed_ms"] for c, r in
                zip(milestones["c-raw"][goal], milestones["rust-raw"][goal]))
        comparisons[goal] = item

    old = [checked(runs, "user-kernel-narrowed-raw-0"),
           checked(runs, "user-kernel-narrowed-preprocessed-0")]
    ablations = []
    for family in ("mt", "cubic-zoom", "cubic-clipped", "local-quadratic", "stalled-local", "corroborated"):
        for backend in ("rust-raw", "rust-preprocessed"):
            ablations.append(checked(runs, f"parity-{family}-{backend}-0"))
    net_gap = abs(Decimal(summaries["rust-raw"]["quote_values"]["base_net"])
                  - Decimal(summaries["c-raw"]["quote_values"]["base_net"]))
    positives = [r for b in BACKENDS[:3] for r in clean[b]]
    negatives = clean["c-preprocessed"]
    result = {
        "scope": "One frozen 1000-pool input; five sequential rotated-order repeats per backend, not independent inputs",
        "method": "Decimal100 recorded hi+lo quote sums; accepted-state provenance and sparse fixed-goal clocks; no extra FG calls",
        "parameters_unchanged": {"memory":17,"factr":1e3,"pgtol":1e-9,"c1":.001,"c2":.9,
                                 "max_line_search_steps":20,"coordinate_passes":20,"dd_budget":32,
                                 "relative_flow_tolerance":1e-9,"absolute_flow_slack":0},
        "native_predictor":False,"dd_recovery":False,"dd_additive":False,
        "strict_certification_scope":"DD numerical gate only, not transaction admission. All valid numerical controls still reject execution admission; no transaction selected and net profit unavailable.",
        "inputs":inputs,"final_build":build,"mirror_matches_production":True,
        "observer_artifacts_sha256": {name:sha(artifacts/name) for name in
            ("observer-binary","observer-solver.rs","observer-repro_native.rs",
             "sparse-observer-binary","sparse-observer-solver.rs","sparse-observer-repro_native.rs")},
        "full_trace_wall_times_not_used": True,
        "observer_control_warm_counts_values_hash_and_dd_audit_identical": True,
        "summary":summaries,"matched_accepted_milestones":comparisons,
        "final_raw_base_net_absolute_difference":str(net_gap),
        "positive_strict_passes":sum(r["strict_acceptance_passed"] is True for r in positives),
        "positive_strict_total":len(positives),
        "negative_control_rejections":sum(r["strict_acceptance_passed"] is False for r in negatives),
        "negative_control_total":len(negatives),
        "warning":"Both raw terminal states fail pgtol. Different terminal PG norms preclude terminal-time-only speed claims. Compare matched accepted states and strict completed solves; do not use full-trace clocks or C-preprocessed rejection time as a successful solve.",
        "clean_cases":[r for rows in clean.values() for r in rows],
        "observer_cases":observers,"previous_scoped_cases":old,"ablations":ablations}
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream: stream.write(encoded)
        print("Wrote", args.output)
    else:
        print(encoded, end="")
    return int(args.require_strict and any(r["strict_acceptance_passed"] is not True for r in positives))


if __name__ == "__main__":
    raise SystemExit(main())
