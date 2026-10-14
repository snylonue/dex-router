"""Supervisor lifecycle checks; no optimizer subprocess or real signal is used."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_convex_fuzz as worker


class FingerprintTests(unittest.TestCase):
    def test_edits_and_new_interpolation_modules_change_the_fingerprint(self):
        with tempfile.TemporaryDirectory(prefix="convex-fuzz-fingerprint-") as temporary:
            root = Path(temporary).resolve()
            self.assertTrue(root.is_dir())
            for name in ["Cargo.lock", "Cargo.toml", "dexrouter-lbfgsb/Cargo.toml",
                         "dexrouter-lbfgsb/tests/convex_fuzz.rs", "dexrouter-lbfgsb/src/cubic.rs"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture")
            before = worker.source_fingerprint(root)
            (root / "dexrouter-lbfgsb/src/cubic.rs").write_text("changed interpolation")
            after = worker.source_fingerprint(root)
            self.assertNotEqual(before, after)
            (root / "dexrouter-lbfgsb/src/new_model.rs").write_text("new module")
            self.assertNotEqual(after, worker.source_fingerprint(root))


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="convex-fuzz-worker-test-")
        self.root = Path(temporary.name).resolve()
        self.assertTrue(self.root.is_dir())
        self.addCleanup(temporary.cleanup)
        self.stack = self.enterContext(ExitStack())
        for target, value in [
            ("__file__", str(self.root / "crate/benchmarks/run_convex_fuzz.py")),
        ]:
            self.stack.enter_context(patch.object(worker, target, value))
        self.stack.enter_context(patch.object(worker, "source_fingerprint", return_value="fixture-hash"))
        self.stack.enter_context(patch.object(worker.os, "nice"))
        self.stack.enter_context(patch.object(worker.os, "sched_getaffinity", return_value={0}))
        self.stack.enter_context(patch.object(worker.os, "sched_setaffinity"))
        self.stack.enter_context(patch.object(worker.signal, "signal"))
        self.kill = self.stack.enter_context(patch.object(worker.os, "killpg"))
        self.spawn = self.stack.enter_context(patch.object(worker.subprocess, "Popen"))
        self.process = self.spawn.return_value.__enter__.return_value
        self.process.pid = 12345
        self.stack.enter_context(patch("sys.argv", ["worker", "--seconds", "1", "--cases", "4"]))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def once(self):
        with patch.object(worker.time, "monotonic", side_effect=[0, 0, 0, 0, 0, 2]):
            return worker.main()

    def checkpoint(self):
        return json.loads((self.root / "target/convex-fuzz-live/latest.json").read_text())

    def test_timeout_kills_cargo_and_its_children_and_preserves_checkpoint(self):
        self.process.wait.side_effect = [subprocess.TimeoutExpired("cargo", 1), -9]
        self.assertEqual(self.once(), 124)
        self.kill.assert_called_once_with(12345, signal.SIGKILL)
        self.assertEqual(self.checkpoint()["exit_code"], 124)
        self.assertTrue((self.root / self.checkpoint()["log"]).is_file())

    def test_cancellation_also_kills_the_batch(self):
        self.process.wait.side_effect = [KeyboardInterrupt(), -9]
        with self.assertRaises(KeyboardInterrupt):
            self.once()
        self.kill.assert_called_once_with(12345, signal.SIGKILL)

    def test_restart_preserves_logs_and_clears_discovery_flags(self):
        self.process.wait.return_value = 0
        with patch.dict(worker.os.environ, {"CONVEX_FUZZ_HUNT": "scale-f32", "CONVEX_FUZZ_REQUIRE_KKT": "1"}):
            self.assertEqual(self.once(), 0)
            first = self.checkpoint()
            self.assertEqual(self.once(), 0)
        second = self.checkpoint()
        self.assertNotEqual(first["log"], second["log"])
        self.assertTrue((self.root / first["log"]).is_file())
        self.assertTrue((self.root / second["log"]).is_file())
        self.assertFalse(second["changed_during_batch"])
        self.assertEqual(second["source_hash"], "fixture-hash")
        for call in self.spawn.call_args_list:
            self.assertTrue(call.kwargs["start_new_session"])
            self.assertNotIn("CONVEX_FUZZ_HUNT", call.kwargs["env"])
            self.assertNotIn("CONVEX_FUZZ_REQUIRE_KKT", call.kwargs["env"])
        self.kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
