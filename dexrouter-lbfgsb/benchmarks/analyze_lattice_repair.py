#!/usr/bin/env python3
"""Audit the recovered original preprocessed CLI, without extra market quotes.

Use --fixture EXTRACTED_ORIGINAL --runs IMMUTABLE_RUNS --output NEW_JSON.
The older source-bound analyzers and historical reports are not modified.
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from analyze_hermes_convergence import decimal, load, messages, pair
from decimal import Decimal


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root, fixture, runs):
    manifest = load(runs / "manifest.json")
    assert sha(runs / "binary") == manifest["binary_sha256"]
    modules = manifest["module_hashes"]
    assert set(modules) == {p.name for p in (root / "dexrouter-lbfgsb/src").glob("*.rs")}
    assert len(modules) == 13
    for name, digest in modules.items():
        assert sha(root / "dexrouter-lbfgsb/src" / name) == digest, name
        assert sha(fixture / "vendor/dexrouter-lbfgsb/src" / name) == digest, name
    assert manifest["archive_non_kernel_changes"] == ["Cargo.toml", "src/strategy/precision_newton.rs"]
    assert manifest["no_default_features"]
    assert manifest["observer_after_original_terminal_checks"]
    archive = Path(manifest["archive"])
    assert sha(archive) == manifest["archive_sha256"]
    changed = []
    prefix = "hermes-minimize-integration-20261013/"
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.is_dir():
                continue
            assert item.filename.startswith(prefix)
            name = item.filename[len(prefix):]
            actual = (fixture / name).read_bytes()
            before = z.read(item)
            if actual != before and not name.startswith("vendor/dexrouter-lbfgsb/src/"):
                changed.append(name)
            if name == "src/strategy/precision_newton.rs":
                lines = actual.decode().splitlines(keepends=True)
                removed = [l for l in lines if "// Post-decision observation only;" in l or 'eprintln!("HERMES_DD_AUDIT {}"' in l]
                assert len(removed) == 2
                assert "".join(l for l in lines if l not in removed).encode() == before
                assert actual.index(b'HERMES_DD_AUDIT') > actual.index(b'"Hermes 4.1 precision terminal constraints failed"')
            if name == "Cargo.toml":
                assert actual.decode().split("\n[workspace]\n")[0].rstrip() == before.decode().rstrip()
    assert sorted(changed) == manifest["archive_non_kernel_changes"]
    original = load(root / "dexrouter-lbfgsb/benchmarks/optimizer_parity.json")
    for item in original["inputs"].values():
        assert sha(fixture / item["path"]) == item["sha256"]
    baseline = next(r for r in original["clean_cases"] if r["backend"] == "rust-preprocessed")["warm"]
    baseline = {k:v for k,v in baseline.items() if k != "elapsed_ms"}
    cases = []
    for repeat in range(3):
        label = "preprocessed-" + str(repeat)
        receipt = load(runs / (label + "-process.json"))
        assert receipt["binary_sha256"] == manifest["binary_sha256"]
        assert receipt["invocations"] == 1 and not receipt["timed_out"]
        assert receipt["input_hashes_unchanged"] and receipt["environment_cleared"]
        assert receipt["rpc_requests"] == receipt["evm_simulations"] == 0
        assert receipt["returncode"] == 0
        lines = (runs / (label + "-process.log")).read_text().splitlines()
        warm, = messages(lines, "HERMES_EQUIVALENT")
        audit, = messages(lines, "HERMES_DD_AUDIT")
        n = len(audit["prices"])
        assert n == 607 and len(audit["samples"]) == 994
        assert len(audit["floors"]) == len(audit["scales"]) == n
        incoming, outgoing = [Decimal(0)] * n, [Decimal(0)] * n
        for sample in audit["samples"]:
            for side, token in enumerate(sample["tokens"]):
                assert 0 <= token < n
                i, o = pair(sample["input"][side]), pair(sample["output"][side])
                assert i >= 0 and o >= 0
                incoming[token] += o
                outgoing[token] += i
        relative = [abs(i-o)/max(i,o) if max(i,o) else Decimal(0)
                    for i,o in zip(incoming,outgoing)]
        bad = [i for i in range(1,n) if relative[i] > decimal(1e-9)]
        assert all(s > 0 for s in audit["scales"])
        scaled = max(abs(incoming[i]-outgoing[i])*decimal(audit["scales"][i]) for i in range(1,n))
        assert not bad and scaled <= decimal(1e-9)
        assert all(pair(p) >= pair(f) and pair(p) > 0 for p,f in zip(audit["prices"],audit["floors"]))
        assert pair(audit["prices"][0]) == Decimal(1)
        assert audit["status"] == "strict_and_gradient_satisfied"
        assert audit["iteration"] <= 32
        assert audit["metrics"]["strict_bad_tokens"] == []
        admission = load(runs / label / "admission.json")
        assert admission["status"] == "no_admissible_candidate"
        state = {k:v for k,v in warm.items() if k != "elapsed_ms"}
        cases.append({"repeat":repeat,"strict_dd_passed":True,"warm":warm,
                      "warm_numerical_state_matches_historical":state == baseline,
                      "dd_updates":audit["iteration"],"max_relative_flow_residual":str(max(relative[1:])),
                      "scaled_gradient_max":str(scaled),"execution_admission":admission["status"],
                      "audit_content_sha256":hashlib.sha256(json.dumps(audit,sort_keys=True).encode()).hexdigest()})
    assert len({c["audit_content_sha256"] for c in cases}) == 1
    assert all({k:v for k,v in c["warm"].items() if k != "elapsed_ms"} ==
               {k:v for k,v in cases[0]["warm"].items() if k != "elapsed_ms"} for c in cases)
    return {"scope":"Recovered ORIGINAL preprocessed path only, 3 repeats of one input; no raw/C timing claim",
            "method":"Post-decision observation of existing DD samples; Decimal100 independent sums, zero absolute slack",
            "manifest":manifest,"cases":cases,"strict_dd_passed":True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    result = analyze(Path(__file__).resolve().parents[2],a.fixture.resolve(),a.runs.resolve())
    with a.output.open("x") as f:
        json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps(result["cases"],indent=2))


if __name__ == "__main__":
    main()
