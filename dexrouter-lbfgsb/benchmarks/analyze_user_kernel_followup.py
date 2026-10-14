#!/usr/bin/env python3
"""Independently audit named frozen user-problem runs; never rerun market quotes.

Run from the workspace root. Requires the preserved extracted fixture under target/.
Prints JSON to stdout. --require-strict makes a failed original DD gate exit nonzero.
"""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_hermes_convergence import check_case, load, messages


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_run(runs, label):
    if (runs / (label + "-dd-audit.json")).exists():
        row = check_case(runs, label)
        row["independent_quote_check_available"] = True
        row["rejection"] = None
        return row
    # Some original DD error exits occur before the final quote dump. Do not
    # fabricate an independent residual certificate from the solver log.
    receipt = load(runs / (label + "-process.json"))
    assert receipt["returncode"] != 0 and not receipt["timed_out"]
    assert receipt["input_hashes_unchanged"] and receipt["environment_cleared"]
    assert receipt["rpc_requests"] == receipt["evm_simulations"] == 0
    assert not receipt["dd_recovery"] and not receipt["dd_additive"]
    lines = (runs / (label + "-process.log")).read_text().splitlines()
    rejection = "\n".join(lines[next(i for i, line in enumerate(lines) if line.startswith("Error:")):])
    assert "final precision refinement failed; intermediate flow rejected" in rejection
    warm = (messages(lines, "HERMES_RAW") or messages(lines, "HERMES_EQUIVALENT"))[0]
    return {
        "case": label, "backend": receipt["backend"], "variant": receipt["variant"],
        "binary_sha256": receipt["binary_sha256"], "warm": warm,
        "warm_price_hash": messages(lines, "HERMES_WARM_PRICE_HASH")[0]["hash"],
        "returncode": receipt["returncode"], "callback_trace": receipt["trace"],
        "warm_iterations_cap": receipt["warm_iterations_cap"],
        "independent_quote_check_available": False, "rejection": rejection,
        "strict_acceptance_passed": None, "dd_status": "original_process_rejected_before_quote_dump",
        "audit_sha256": None, "dd_updates": None, "bad_tokens": None,
        "max_relative_flow_residual": None, "scaled_gradient_max": None,
        "floor_violations": None, "production_status": None,
        "receipt": str(runs / (label + "-process.json")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--require-strict", action="store_true")
    parser.add_argument("--output", type=Path, help="Create a NEW JSON artifact")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    runs = root / "target/hermes-repro/runs"
    repro = runs.parent / "hermes-minimize-integration-20261013"
    manifest = load(repro / "input/pools-1000/manifest.json")
    for record in manifest["inputs"].values():
        assert sha256(repro / record["path"]) == record["sha256"]
    before = load(runs.parent / "kernel-fuzz-followup/kernel-manifest.json")
    current_sources = {}
    for module in before["modules"]:
        name = module["name"]
        source = root / "dexrouter-lbfgsb/src" / name
        mirror = repro / "vendor/dexrouter-lbfgsb/src" / name
        assert source.read_bytes() == mirror.read_bytes(), "Reproducer kernel differs: " + name
        current_sources[name] = sha256(source)
    final = load(runs.parent / "kernel-fuzz-followup/final-build.json")
    assert current_sources == final["source_sha256"]
    assert sha256(root / final["binary"]) == final["binary_sha256"]
    rows = []
    for label in args.labels:
        row = check_run(runs, label)
        receipt = load(runs / (label + "-process.json"))
        assert sha256(Path(receipt["argv"][0])) == row["binary_sha256"]
        assert row["variant"] == "baseline" and row["warm_iterations_cap"] is None
        assert not row["callback_trace"]
        rows.append({key: row[key] for key in (
            "case", "backend", "binary_sha256", "warm", "warm_price_hash",
            "independent_quote_check_available", "rejection",
            "audit_sha256", "returncode", "dd_updates", "dd_status",
            "strict_acceptance_passed", "bad_tokens", "max_relative_flow_residual",
            "scaled_gradient_max", "floor_violations", "production_status", "receipt",
        )})
    result = {
        "scope": "Frozen 1000-pool user problem; optimizer-only safeguards narrowed",
        "method": "Decimal 100-digit sums of recorded DD hi+lo quotes; no extra market calls",
        "strict_relative_tolerance": 1e-9, "absolute_flow_slack": 0,
        "coordinate_passes": 20, "dd_update_budget": 32,
        "native_predictor": False, "dd_recovery": False, "dd_additive": False,
        "warning": "Warm status is not an original-coordinate KKT certificate; DD acceptance is independently checked. These are repeated runs of one input, not multiple inputs.",
        "inputs": manifest["inputs"],
        "pre_followup_source_manifest": before,
        "current_mirror_matches_production": True, "current_source_sha256": current_sources,
        "final_build": final,
        "cases": rows,
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        with args.output.open("x") as stream:
            stream.write(encoded)
        print("Wrote", args.output)
    return int(args.require_strict and any(not row["strict_acceptance_passed"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
