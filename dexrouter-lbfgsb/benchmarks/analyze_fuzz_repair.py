#!/usr/bin/env python3
"""Bind new fuzz-repair runs to their sources and reuse the unchanged DD audit.

This does not rerun quotes or replace the historical parity analyzer's bindings.
Use --runs PRESERVED_RUN_DIRECTORY --output NEW_JSON_PATH from the workspace.
"""
import argparse
import json
from pathlib import Path

from analyze_optimizer_parity import checked, sha, stats, warm_without_time
from analyze_hermes_convergence import load


def analyze(runs, root):
    manifest = load(runs / "manifest.json")
    expected = manifest["binary_sha256"]
    assert sha(runs / "binary") == expected
    fixture = root / "target/hermes-repro/hermes-minimize-integration-20261013"
    modules = manifest["module_hashes"]
    assert len(modules) == 13
    for name, digest in modules.items():
        assert sha(root / "dexrouter-lbfgsb/src" / name) == digest, name
        assert sha(fixture / "vendor/dexrouter-lbfgsb/src" / name) == digest, name
    historical = load(root / "dexrouter-lbfgsb/benchmarks/optimizer_parity.json")
    for record in historical["inputs"].values():
        assert sha(fixture / record["path"]) == record["sha256"]
    observer = load(root / "target/hermes-repro/parity-followup/observer-build.json")
    assert sha(fixture / "vendor/minimize-only-routing/src/repro_native.rs") == observer["original_bridge_sha256"]
    records = load(runs / "runs.json")
    assert len(records) == 6
    assert {(r["backend"], r["repeat"]) for r in records} == {
        (b, i) for b in ("rust-raw", "rust-preprocessed") for i in range(3)
    }
    rows = []
    for record in records:
        assert record["exit_code"] == 0
        label = record["backend"] + "-" + str(record["repeat"])
        row = checked(runs, label, expected)
        assert row["strict_acceptance_passed"]
        assert row["execution_admission"]["status"] == "no_admissible_candidate"
        assert row["execution_admission"]["net_profit"] is None
        rows.append(row)
    summaries = {}
    for backend in ("rust-raw", "rust-preprocessed"):
        group = [r for r in rows if r["backend"] == backend]
        baseline = next(r for r in historical["clean_cases"] if r["backend"] == backend)
        assert all(warm_without_time(r) == warm_without_time(group[0]) for r in group)
        assert len({r["warm_price_hash"] for r in group}) == 1
        assert len({r["audit_sha256"] for r in group}) == 1
        summaries[backend] = {
            "repeats": len(group), "warm": warm_without_time(group[0]),
            "warm_ms": stats([r["warm"]["elapsed_ms"] for r in group]),
            "solve_ms": stats([r["solve_wall_ms"] for r in group]),
            "warm_numerical_state_matches_historical": warm_without_time(group[0]) == warm_without_time(baseline),
            "warm_price_hash_matches_historical": group[0]["warm_price_hash"] == baseline["warm_price_hash"],
            "dd_audit_sha_matches_historical": group[0]["audit_sha256"] == baseline["audit_sha256"],
            "dd_updates": group[0]["dd_updates"],
            "max_relative_flow_residual": group[0]["max_relative_flow_residual"],
            "execution_admission": group[0]["execution_admission"],
        }
    return {"scope": "Three repeats of each original Rust path on one frozen input; no new C speedup claim",
            "method": "Current 13-module and binary SHA binding; unchanged Decimal100 DD quote audit, no extra market calls",
            "strict_relative_tolerance": 1e-9, "absolute_flow_slack": 0,
            "manifest": manifest, "inputs": historical["inputs"],
            "summaries": summaries, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.runs.resolve(), Path(__file__).resolve().parents[2])
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result["summaries"], indent=2))


if __name__ == "__main__":
    main()
