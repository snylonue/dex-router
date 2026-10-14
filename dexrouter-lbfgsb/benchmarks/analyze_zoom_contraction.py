#!/usr/bin/env python3
"""Audit original failed searches, isolated callback probes, and final full replays.

Run at the workspace root with the preserved extracted reproduction under target/.
Only reads recorded data and checks assertions; emits JSON on stdout.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from decimal import Decimal

sys.dont_write_bytecode = True
from analyze_hermes_convergence import check_case
from analyze_line_search_fix import summarize


def load(path):
    return json.loads(Path(path).read_text())


def tagged(path, tag):
    return [json.loads(l[len(tag)+1:]) for l in Path(path).read_text().splitlines() if l.startswith(tag+" ")]


def main():
    runs = Path("target/hermes-repro/runs")
    repro = runs.parent / "hermes-minimize-integration-20261013"
    manifest = load(repro / "input/pools-1000/manifest.json")
    for record in manifest["inputs"].values():
        assert hashlib.sha256((repro / record["path"]).read_bytes()).hexdigest() == record["sha256"]
    previous = load("dexrouter-lbfgsb/benchmarks/line_search_scaling_fix.json")
    baseline = [r for r in previous["summary"] if r["stage"] == "final"]
    before = next(c for c in previous["cases"] if c["case"] == "fix-final-search-rust-raw-0")
    original_label = "terminal-geometry-raw-0"
    probe_label = "terminal-shadow-clipped-raw-0"
    original = check_case(runs, original_label)
    probe = check_case(runs, probe_label)
    assert original["warm_price_hash"] == probe["warm_price_hash"] == before["warm_price_hash"]
    assert original["audit_sha256"] == probe["audit_sha256"] == before["audit_sha256"]
    original_log = runs / (original_label + "-process.log")
    probe_log = runs / (probe_label + "-process.log")
    states = tagged(original_log, "HERMES_FAILED_SEARCH")[-2:]
    fixed = tagged(probe_log, "HERMES_SHADOW_SEARCH")
    assert len(states) == len(fixed) == 2
    assert states[0]["x0"] == states[1]["x0"] and states[0]["g0"] == states[1]["g0"]
    terminal = []
    d = Decimal.from_float
    for state, recovered in zip(states, fixed):
        assert state["derivative0"] == recovered["derivative0"]
        assert len(state["trials"]) == 20 and recovered["accepted"] and recovered["evaluations"] <= 20
        assert d(recovered["f"]) <= d(state["f0"]) + d(1e-3)*d(recovered["step"])*d(state["derivative0"])
        assert abs(d(recovered["derivative"])) <= d(.9)*-d(state["derivative0"])
        assert all(t["changed"] > 0 and abs(t["actual_initial_slope"]-state["derivative0"]) <= 1e-7*abs(state["derivative0"]) for t in state["trials"])
        widths = [abs(t["bracket"][3]-t["bracket"][0]) for t in state["trials"][-5:]]
        assert all(abs(b/a-.9) < 1e-12 for a,b in zip(widths,widths[1:]))
        terminal.append({"f0": state["f0"], "derivative0": state["derivative0"],
            "dimension": len(state["x0"]), "direction_inf_norm": max(map(abs,state["direction"])),
            "vector_sha256": {k: hashlib.sha256(json.dumps(state[k]).encode()).hexdigest() for k in ("x0","g0","direction")},
            "original_trials": state["trials"], "repaired_search": recovered,
            "last_original_decrease_in_ulps": (state["f0"]-state["trials"][-1]["f"])/math.ulp(state["f0"])})
    discarded = []
    for backend in ("rust-raw", "rust-preprocessed"):
        rows = [check_case(runs,"fix-contraction-"+backend+"-"+str(i)) for i in range(3)]
        discarded.append(summarize("discarded-unrestricted-contraction",backend,rows))
    old_dd = load(runs / "fix-final-search-rust-preprocessed-0-dd-audit.json")
    weak_price = old_dd["prices"][264][0]
    precision = {"case":"fix-final-search-rust-preprocessed-0", "token":264,
        "price_hi":weak_price, "f64_ulp":math.ulp(weak_price),
        "approximate_physical_DD_correction_from_original_diagnosis":3.01545e-39,
        "warning":"The observed DD correction is below f64 resolution; this does not imply every f64 warm start must fail downstream DD."}
    source = Path("dexrouter-lbfgsb/src/line_search.rs").read_text()
    mirror = (repro / "vendor/dexrouter-lbfgsb/src/line_search.rs").read_text()
    mirror = mirror[:mirror.rfind("\nfn hybrid_wolfe")]
    mirror = mirror.replace('    if std::env::var("HERMES_REPRO_VARIANT").is_ok_and(|v|v.contains("c-search")) {return hybrid_wolfe(x0,f0,g0,direction,bounds,options,evaluation_budget,initial_step,fg);}' + "\n", "")
    mirror = mirror.replace('    let _time = crate::audit::timer("callback");' + "\n", "")
    assert mirror == source, "replay kernel differs from production source"
    cases, summaries = [], []
    for backend in ("rust-raw", "rust-preprocessed"):
        rows = [check_case(runs, "fix-contraction-clipped-"+backend+"-"+str(i)) for i in range(3)]
        for row in rows:
            assert row["variant"] == "baseline" and row["warm_iterations_cap"] is None and not row["callback_trace"]
            log = Path(row["receipt"].replace("-process.json", "-process.log")).read_text()
            assert "HERMES_SHADOW_SEARCH" not in log and "HERMES_FAILED_SEARCH" not in log
        summaries.append(summarize("two-update-clipped-contraction", backend, rows))
        cases.extend(rows)
    assert len({c["binary_sha256"] for c in cases}) == 1
    assert all(c["strict_acceptance_passed"] for c in cases), "frozen user precision regression failed"
    test_text = Path("target/hermes-repro/optimizer-clipped-contraction-tests.log").read_text()
    test_rows = re.findall(r"test result: ok[.] (\d+) passed; (\d+) failed; (\d+) ignored",test_text)
    assert test_rows and "test result: FAILED" not in test_text
    tests = dict(zip(("passed","failed","ignored"),(sum(int(r[i]) for r in test_rows) for i in range(3))))
    print(json.dumps({"scope":"Pure-Rust optimizer repair: prevent endpoint crawl without accepting unchecked steps or changing budgets.",
        "base_commit":"d382177", "source_sha256":hashlib.sha256(source.encode()).hexdigest(),
        "method":"Frozen original callbacks, then independent Decimal checks of recorded Wolfe and strict DD acceptance; no native predictor or DD recovery.",
        "tests":tests, "inputs":manifest["inputs"], "terminal_searches":terminal,
        "diagnostic_receipts":[original["receipt"],probe["receipt"]],
        "diagnostic_warning":"Shadow probes call fg in addition to the original solver, discard their results, and are NOT performance or total-budget measurements. Final runs contain no probes.",
        "baseline":baseline,"summary":summaries,"cases":cases, "discarded_candidate_summary":discarded,"precision_boundary":precision,
        "warning":"A valid Wolfe trial is not optimizer KKT convergence; DD success is not trade admission."},indent=2))


if __name__ == "__main__":
    main()
