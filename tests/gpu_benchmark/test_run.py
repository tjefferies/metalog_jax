# Copyright: Travis Jefferies 2026
"""Tests for the benchmark that runs in each pod: settings, scheduling and data."""

import json
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from benchmarks.gpu import environment, run
from benchmarks.gpu.common import knee, sha256_file
from benchmarks.gpu.data import quantiles
from benchmarks.gpu.serve import ResultsServer

PLAN = json.loads(run.PACKAGE_DIR.joinpath("plan.json").read_text())


class ConfigTest(absltest.TestCase):
    """Settings come from the environment, and a served run needs its token."""

    def test_served_runs_need_a_token(self):
        """Without BENCH_LOCAL, results are served, so the path secret is required."""
        with self.assertRaisesRegex(ValueError, "BENCH_TOKEN"):
            run.Config.from_environment({})
        config = run.Config.from_environment(
            {"BENCH_TOKEN": "t", "BENCH_DEADLINE": "100"}
        )
        self.assertEqual(
            (config.token, config.deadline, config.port), ("t", 100.0, 8000)
        )

    def test_local_runs(self):
        """A local run needs no token and defaults to a time budget."""
        config = run.Config.from_environment(
            {"BENCH_LOCAL": "1", "BENCH_TIME_BUDGET_S": "60"}
        )
        self.assertTrue(config.local)
        self.assertAlmostEqual(config.deadline, time.time() + 60, delta=5)


class SecretsTest(absltest.TestCase):
    """The manifest never records a credential."""

    def test_secrets_are_dropped(self):
        """Keys, tokens and passwords go; the setup's variables stay."""
        kept = environment.safe_environment(
            {
                "RUNPOD_API_KEY": "secret",
                "BENCH_TOKEN": "secret",
                "PUBLIC_KEY": "ssh-ed25519 ...",
                "JUPYTER_PASSWORD": "secret",
                "RUNPOD_PUBLIC_IP": "1.2.3.4",
                "RUNPOD_POD_ID": "pod1",
                "RUNPOD_DC_ID": "US-TX-3",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "JAX_PLATFORMS": "cuda,cpu",
                "HOME": "/root",
            }
        )
        self.assertEqual(
            sorted(kept),
            [
                "JAX_PLATFORMS",
                "RUNPOD_DC_ID",
                "RUNPOD_POD_ID",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
            ],
        )


class ResultsServerTest(absltest.TestCase):
    """Results are served only under the token, only by name; ack is recorded."""

    def setUp(self):
        """Serve a results directory with one file."""
        super().setUp()
        self.dir = Path(self.enter_context(tempfile.TemporaryDirectory()))
        (self.dir / "results.jsonl").write_text("{}\n")
        self.server = ResultsServer(self.dir, "tok", 0).start()
        self.addCleanup(self.server.stop)
        self.base = f"http://127.0.0.1:{self.server.httpd.server_port}"

    def get(self, path):
        """GET a path; return the status and body."""
        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, b""

    def test_files_need_the_token(self):
        """The file is served under the token, and nowhere else."""
        self.assertEqual(self.get("/tok/results.jsonl"), (200, b"{}\n"))
        self.assertEqual(self.get("/results.jsonl")[0], 404)
        self.assertEqual(self.get("/other/results.jsonl")[0], 404)
        self.assertEqual(self.get("/tok/../run.py")[0], 404)
        self.assertEqual(self.get("/tok/missing.json")[0], 404)

    def test_ack(self):
        """The launcher's ack tells the pod it may leave."""
        self.assertFalse(self.server.acknowledged.is_set())
        self.assertEqual(self.get("/tok/ack")[0], 200)
        self.assertTrue(self.server.acknowledged.is_set())


class SchedulingTest(absltest.TestCase):
    """Phases share the time, doubling stops in time, and chunks use the knee."""

    def runner(self, seconds=1000.0):
        """A runner whose measurements are recorded, not run."""
        directory = Path(self.enter_context(tempfile.TemporaryDirectory()))
        config = run.Config.from_environment(
            {
                "BENCH_LOCAL": "1",
                "BENCH_RESULTS": str(directory),
                "BENCH_TIME_BUDGET_S": str(seconds),
            }
        )
        runner = run.Runner(config, PLAN)
        runner.specs = []
        return runner

    def test_phase_shares_add_up(self):
        """The phases' shares divide the time until the deadline, less the reserve."""
        self.assertAlmostEqual(sum(p["share"] for p in PLAN["phases"]), 1.0)
        runner = self.runner()
        self.assertAlmostEqual(runner.phase_end["sampling"], runner.end, delta=1e-6)

    def test_min_s(self):
        """A phase gets at least its min_s, within the overall end."""
        runner = self.runner(seconds=200)
        check = PLAN["phases"][0]
        self.assertGreaterEqual(runner.until(check), time.time() + check["min_s"] - 1)

    def test_doubling_stops_before_running_out_of_time(self):
        """A batch whose predicted time doesn't fit is recorded as skipped."""
        runner = self.runner()
        now = [1000.0]
        self.enter_context(mock.patch.object(run.time, "time", lambda: now[0]))

        def measure(spec, until):
            runner.specs.append(spec)
            wall = 30.0 * spec["N"] / 1024
            now[0] += wall
            return runner.record(
                {**spec, "status": "ok", "wall_s": wall, "mode": "feasible"}
            )

        runner.measure = measure
        phase = {
            "name": "batches",
            "methods": {"feasible": 1.0},
            "first": 1024,
            "last": 2**24,
        }
        runner.phase_end["batches"] = now[0] + 600
        runner.run_batches(phase)
        # 16,384 datasets would take about 2.2 * 240 s, and only 150 s are left
        self.assertEqual([s["N"] for s in runner.specs], [1024, 2048, 4096, 8192])
        self.assertEqual(
            (runner.records[-1]["N"], runner.records[-1]["status"]), (16384, "skipped")
        )

    def test_knee(self):
        """The knee is the smallest batch within the tolerance of the best time."""
        records = [
            {"N": 1024, "per_dataset_ms": 2.0},
            {"N": 2048, "per_dataset_ms": 1.04},
        ]
        records += [
            {"N": 4096, "per_dataset_ms": 1.0},
            {"N": 8192, "per_dataset_ms": 1.01},
        ]
        self.assertEqual(knee(records, 0.05)["N"], 2048)
        self.assertEqual(knee(records, 0.01)["N"], 4096)
        self.assertIsNone(knee([], 0.05))

    def test_chunks_use_the_knee(self):
        """The chunked run fits chunks of the knee's size for about phase['seconds']."""
        runner = self.runner()
        for n, ms in ((1024, 2.0), (2048, 1.0), (4096, 1.0)):
            steady = n * ms / 1e3
            runner.record(
                {
                    "phase": "batches",
                    "mode": "feasible",
                    "status": "ok",
                    "N": n,
                    "per_dataset_ms": ms,
                    "steady_s": steady,
                    "repetitions": 3,
                    "wall_s": 10 + 4 * steady,
                }
            )
        specs = []
        runner.measure = lambda spec, until: specs.append(spec) or {"status": "ok"}
        phase = {"name": "chunked", "methods": {"feasible": 1.0}, "seconds": 20}
        phase |= {"min_chunks": 2, "knee_tolerance": 0.05}
        runner.phase_end["chunked"] = time.time() + 600
        runner.run_chunked(phase)
        self.assertEqual((specs[0]["chunk"], specs[0]["N"]), (2048, 2048 * 9))
        self.assertAlmostEqual(specs[0]["time_s"], 20)


class DataTest(absltest.TestCase):
    """Datasets don't depend on the chunk they're drawn in, and stay above 0."""

    def test_chunks_draw_the_same_datasets(self):
        """Datasets 4 to 7 are the same drawn alone or with 0 to 3."""
        whole = quantiles(59, 0, 8)
        np.testing.assert_array_equal(quantiles(59, 4, 4), whole[4:])

    def test_quantiles_are_positive_and_increasing(self):
        """Every family is positive, and quantiles increase with the level."""
        x = np.asarray(quantiles(59, 0, 20))
        self.assertEqual(x.shape, (20, 105))
        self.assertTrue((x > 0).all())
        self.assertTrue((np.diff(x, axis=1) >= 0).all())

    def test_seed_matters(self):
        """A different seed draws different datasets."""
        self.assertFalse(jnp.allclose(quantiles(59, 0, 2), quantiles(60, 0, 2)))


class FinishTest(absltest.TestCase):
    """The checksums cover every result file."""

    def test_checksums(self):
        """checksums.sha256 lists each file's digest, and status gets them all."""
        directory = Path(self.enter_context(tempfile.TemporaryDirectory()))
        (directory / "results.jsonl").write_text("{}\n")
        config = run.Config.from_environment(
            {"BENCH_LOCAL": "1", "BENCH_RESULTS": str(directory)}
        )
        files = run.finish(config, {"plan": {}}, "done")
        self.assertEqual(
            sorted(files), ["checksums.sha256", "manifest.json", "results.jsonl"]
        )
        for name, digest in files.items():
            self.assertEqual(sha256_file(directory / name), digest)
        listed = (directory / "checksums.sha256").read_text().splitlines()
        self.assertEqual(len(listed), 2)

    def test_self_delete_outside_a_pod(self):
        """Outside a pod there's nothing to delete."""
        self.assertIsNone(run.self_delete({}))


if __name__ == "__main__":
    absltest.main()
