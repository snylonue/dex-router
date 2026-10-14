#!/usr/bin/env python3
"""Audit the preserved line-search ablations, from the workspace root.

Requires the existing extracted offline reproduction and recorded runs; no solve,
market evaluation, RPC, or input mutation is performed. Print JSON to stdout.
"""
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

# Keep this audit read-only, including Python bytecode cache side effects.
sys.dont_write_bytecode = True
from analyze_hermes_convergence import check_case


def summarize(stage, backend, cases):
    assert len({c["warm_price_hash"] for c in cases}) == 1
    assert len({c["audit_sha256"] for c in cases}) == 1
    assert len({(c["warm"]["iterations"], c["warm"]["evaluations"]) for c in cases}) == 1
    row = {"stage": stage, "backend": backend, "repetitions": len(cases),
           "warm": {k: v for k, v in cases[0]["warm"].items() if k != "elapsed_ms"},
           "warm_price_hash": cases[0]["warm_price_hash"],
           "strict_dd_passed": cases[0]["strict_acceptance_passed"],
           "max_relative_flow_residual": cases[0]["max_relative_flow_residual"],
           "restarts": len(cases[0]["restarts"]), "milestones": []}
    for name, values in [("warm_ms", [c["warm"]["elapsed_ms"] for c in cases]),
                         ("solve_ms", [c["solve_wall_ms"] for c in cases])]:
        row[name] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
    for p in cases[0]["milestones"]:
        values = [next(q["elapsed_ms"] for q in c["milestones"] if q["threshold"] == p["threshold"]) for c in cases]
        row["milestones"].append({**{k: v for k, v in p.items() if k != "elapsed_ms"},
                                   "elapsed_ms_median": statistics.median(values)})
    return row


def main():
    runs = Path("target/hermes-repro/runs")
    specs = [("discarded-interior-secant", b, ["fix-agreed-quadratic-" + b + "-" + str(i) for i in range(3)])
             for b in ("rust-raw", "rust-preprocessed")]
    specs += [("endpoint-only", "rust-raw", ["fix-endpoint-only-raw-0"]),
              ("endpoint-only", "rust-preprocessed", ["fix-endpoint-only-preprocessed-0"]),
              ("discarded-cauchy-restart", "rust-raw", ["fix-endpoint-only-cauchy-0"])]
    specs += [("final", b, ["fix-final-search-" + b + "-" + str(i) for i in range(3)])
              for b in ("rust-raw", "rust-preprocessed")]
    cases, summaries = [], []
    for stage, backend, labels in specs:
        rows = [check_case(runs, label) for label in labels]
        for row in rows:
            row["stage"] = stage
            assert not row["callback_trace"] and row["warm_iterations_cap"] is None
        summaries.append(summarize(stage, backend, rows))
        cases.extend(rows)
    final = [c for c in cases if c["stage"] == "final"]
    assert len({c["binary_sha256"] for c in final}) == 1
    assert all(c["variant"] == "baseline" for c in final)
    baseline = json.loads(Path("dexrouter-lbfgsb/benchmarks/hermes1000_convergence_diagnosis.json").read_text())
    tests = Path("target/hermes-repro/optimizer-line-search-final-tests.log").read_text()
    test_rows = re.findall(r"test result: ok[.] (\d+) passed; (\d+) failed; (\d+) ignored", tests)
    assert test_rows
    totals = dict(zip(("passed", "failed", "ignored"), (sum(int(r[i]) for r in test_rows) for i in range(3))))
    source = Path("dexrouter-lbfgsb/src/line_search.rs")
    result = {"scope": "Pure-Rust optimizer search repair; DD acceptance is a separate integration check, not optimizer KKT convergence.",
              "base_commit": baseline["source_commit"], "final_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "tests": totals, "final_source": str(source),
              "analytic_evidence": {"objective": "h*x*x/2 +/- x", "start": 0,
                  "narrow_h_1e24_before": {"evaluations": 21, "iterations": 0, "status": "LineSearchFailed", "projected_gradient": 1},
                  "narrow_h_1e24_after": {"evaluations": 3, "iterations": 1, "status": "ConvergedProjectedGradient"},
                  "flat_h_1e12_inverse_f64_before": {"evaluations": 21, "iterations": 0, "status": "LineSearchFailed", "projected_gradient": 1},
                  "flat_h_1e12_inverse_f64_after": {"evaluations_upper_bound": 5, "status": "ConvergedProjectedGradient"},
                  "regression_file": "dexrouter-lbfgsb/tests/line_search_scaling.rs"},
              "baseline": [r for r in baseline["serial_repeat_summary"] if r["variant"] == "baseline"
                           and r["warm_iterations_cap"] is None and r["backend"] in ("rust-raw", "rust-preprocessed")],
              "summary": summaries, "cases": cases,
              "warning": "Do not compare total time to unlike terminal objectives as convergence speed. Rejected candidates are preserved but are not the final implementation."}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
