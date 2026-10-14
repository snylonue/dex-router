#!/usr/bin/env python3
"""Bounded single-worker live fuzzing; logs and checkpoints survive cancellation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import time


def source_fingerprint(root):
    digest = hashlib.sha256()
    names = {str(p.relative_to(root)) for p in root.glob("dexrouter-lbfgsb/src/**/*.rs")}
    names.update(["dexrouter-lbfgsb/tests/convex_fuzz.rs", "dexrouter-lbfgsb/Cargo.toml",
                  "Cargo.toml", "Cargo.lock"])
    for name in sorted(names):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def interrupt(_signum, _frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--cases", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    if args.seconds <= 0 or args.cases <= 0 or args.seed < 0:
        parser.error("seconds/cases must be positive and seed non-negative")
    root = Path(__file__).resolve().parents[2]
    logs = root / "target/convex-fuzz-live"
    logs.mkdir(parents=True, exist_ok=True)
    if hasattr(os, "nice"):
        os.nice(15)
    if hasattr(os, "sched_getaffinity"):
        allowed = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {max(allowed)})
    signal.signal(signal.SIGTERM, interrupt)
    run_id = str(time.time_ns())
    started = time.monotonic()
    batch = 0
    while time.monotonic() - started < args.seconds:
        mode = "ordinary" if batch % 2 == 0 else "extreme"
        seed = args.seed + batch
        output = logs / f"{run_id}-{batch:05d}-{mode}-{seed}.log"
        command = ["cargo", "test", "--release", "--offline", "-p", "dexrouter-lbfgsb", "--test", "convex_fuzz"]
        if mode == "extreme":
            command += ["extreme_convex_campaign", "--", "--ignored", "--exact", "--nocapture", "--test-threads=1"]
        else:
            command += ["--", "--nocapture", "--test-threads=1"]
        env = os.environ.copy()
        env.update(PROPTEST_CASES=str(args.cases), PROPTEST_RNG_SEED=str(seed))
        env.pop("CONVEX_FUZZ_REQUIRE_KKT", None)
        env.pop("CONVEX_FUZZ_HUNT", None)
        source_start = source_fingerprint(root)
        begin = time.monotonic()
        with output.open("w") as stream:
            try:
                with subprocess.Popen(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                      start_new_session=True) as process:
                    try:
                        code = process.wait(timeout=min(300, max(1, args.seconds - (begin - started))))
                    except BaseException:
                        # Cargo may have a test/rustc child; kill the entire batch,
                        # not just its parent, on timeout or managed cancellation.
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                        raise
            except subprocess.TimeoutExpired:
                code = 124
        source_end = source_fingerprint(root)
        state = dict(session=run_id, source_schema="all-rust-src+manifests-v2",
                     source_hash=source_end, changed_during_batch=source_start != source_end,
                     batch=batch, mode=mode, seed=seed, cases=args.cases, exit_code=code,
                     batch_seconds=round(time.monotonic()-begin, 3), elapsed_seconds=round(time.monotonic()-started, 3),
                     log=str(output.relative_to(root)))
        (logs / "latest.json").write_text(json.dumps(state, indent=2) + "\n")
        with (logs / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(state) + "\n")
        print(json.dumps(state), flush=True)
        if code != 0:
            print("Stopped on a failure; preserve the log and proptest seed before restarting.", flush=True)
            return code
        batch += 1
    print("Completed bounded fuzz session.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
