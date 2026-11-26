"""Run the benchmark plan in this container: the image's entrypoint.

Usage: python -m benchmarks.gpu.run

Each phase of ``plan.json`` gets its share of the time until the deadline, and each
measurement runs in its own process (``benchmarks.gpu.measure``). This process never
imports JAX, so it holds no GPU memory. Results go to ``BENCH_RESULTS``, and are
served to the launcher until it acknowledges them or the download window closes.
Then the process exits, which stops Runpod's compute billing, and the launcher
terminates the pod.

Environment:
    BENCH_RESULTS: Results directory. Default ``/results``.
    BENCH_PLAN: The plan. Default ``plan.json`` beside this module.
    BENCH_PLAN_JSON: The plan itself, as the launcher passes it, in place of
        BENCH_PLAN. Either plan is saved with the results as ``plan.json``.
    BENCH_DEADLINE: Unix time by which the results must be ready. Default: now
        plus ``BENCH_TIME_BUDGET_S``, which defaults to 3600.
    BENCH_BACKEND: The backend to require, ``gpu`` (the default) or ``cpu``.
    BENCH_GPU: The GPU type (``a100-sxm``, ``h100-sxm`` or ``h200-sxm``), checked
        against the device. Unset for a local run.
    BENCH_LOCAL: ``1`` to write the results and exit, without serving them.
    BENCH_TOKEN: The secret path prefix for serving results. Required unless local.
    BENCH_PORT: Port to serve on. Default 8000.
    BENCH_DOWNLOAD_S: Seconds to keep serving after the deadline. Default 180.
"""

import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.gpu import environment
from benchmarks.gpu.common import (
    GPU_TYPES,
    RESULT_PREFIX,
    home_as_tilde,
    knee,
    sha256_file,
    to_json,
    utc_now,
    write_json,
)
from benchmarks.gpu.serve import ResultsServer

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent.parent

# A measurement needs at least this long to start JAX, draw data and compile
MIN_MEASUREMENT_S = 20
# The next batch size is skipped unless its predicted wall time, times this margin,
# remains in the phase: compilation varies from size to size
GROWTH_MARGIN = 1.5
# Errors that mean the device ran out of memory
_OOM_MARKERS = ("RESOURCE_EXHAUSTED", "out of memory", "Out of memory", "OOM")
_TELEMETRY_FIELDS = (
    "timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,"
    "power.draw,clocks.sm,clocks.mem,temperature.gpu"
)


@dataclasses.dataclass
class Config:
    """The run's settings, from the environment."""

    results: Path
    plan: Path
    plan_json: str | None
    deadline: float
    backend: str
    gpu: str | None
    local: bool
    token: str | None
    port: int
    download_s: float

    @classmethod
    def from_environment(cls, environ: dict[str, str] | None = None) -> "Config":
        """Read the settings, as the module docstring describes them.

        Args:
            environ: The environment; ``os.environ`` if None.

        Returns:
            Config: The settings.

        Raises:
            ValueError: Without ``BENCH_TOKEN`` when the run isn't local.
        """
        env = os.environ if environ is None else environ
        budget = float(env.get("BENCH_TIME_BUDGET_S", 3600))
        config = cls(
            results=Path(env.get("BENCH_RESULTS", "/results")),
            plan=Path(env.get("BENCH_PLAN", PACKAGE_DIR / "plan.json")),
            plan_json=env.get("BENCH_PLAN_JSON") or None,
            deadline=float(env.get("BENCH_DEADLINE", time.time() + budget)),
            backend=env.get("BENCH_BACKEND", "gpu"),
            gpu=env.get("BENCH_GPU") or None,
            local=env.get("BENCH_LOCAL") == "1",
            token=env.get("BENCH_TOKEN") or None,
            port=int(env.get("BENCH_PORT", 8000)),
            download_s=float(env.get("BENCH_DOWNLOAD_S", 180)),
        )
        if not config.local and not config.token:
            raise ValueError("BENCH_TOKEN is required unless BENCH_LOCAL=1")
        return config


def predicted_wall(
    record: dict, growth: float, repetitions: int, budget_s: float
) -> float:
    """The wall time of a batch ``growth`` times the size of ``record``'s.

    The work grows at most in proportion to the batch. The rest of the record's wall
    time, starting the process, drawing the data and compiling, stays about the same.
    The steady-state calls are counted as benchmarks.gpu.measure.timed makes them.

    Args:
        record: A successful single-batch record.
        growth: The next batch's size over this one's.
        repetitions: The plan's most steady-state calls.
        budget_s: The plan's seconds for the steady-state calls.

    Returns:
        float: The predicted seconds, without a margin.
    """
    steady = record["steady_s"] * growth
    overhead = record["wall_s"] - record["steady_s"] * (record["repetitions"] + 1)
    calls = 1 + max(1, min(repetitions, math.floor(budget_s / steady)))
    return max(overhead, 0.0) + calls * steady


def describe(record: dict) -> str:
    """A record in a few words, for the launcher's progress lines."""
    words = [record.get("kind"), record.get("method") or record.get("mode")]
    words += [f"{key}={record[key]}" for key in ("N", "K") if record.get(key)]
    wall = f" in {record['wall_s']:.0f} s" if record.get("wall_s") else ""
    return f"{' '.join(w for w in words if w)}: {record['status']}{wall}"


class Runner:
    """Run the plan's phases, each measurement in its own process."""

    def __init__(self, config: Config, plan: dict):
        """Divide the time until the deadline among the phases by their shares."""
        self.config, self.plan = config, plan
        self.records: list[dict] = []
        self.phase: str | None = None
        self.results = config.results / "results.jsonl"
        self.log = config.results / "measurements.log"
        self.start = time.time()
        self.end = config.deadline - plan["reserve_s"]
        total = max(self.end - self.start, 0.0)
        self.phase_end, cumulative = {}, 0.0
        for phase in plan["phases"]:
            cumulative += phase["share"]
            self.phase_end[phase["name"]] = self.start + cumulative * total

    def until(self, phase: dict) -> float:
        """When a phase starting now must end: its share, and at least its min_s."""
        planned = self.phase_end[phase["name"]]
        return min(max(planned, time.time() + phase.get("min_s", 0)), self.end)

    def status(self, state: str, **extra) -> None:
        """Write status.json, which the launcher polls and prints the progress of."""
        last = describe(self.records[-1]) if self.records else None
        write_json(
            self.config.results / "status.json",
            {
                "state": state,
                "phase": self.phase,
                "measurements": len(self.records),
                "last": last,
                "updated": utc_now(),
                **extra,
            },
        )

    def record(self, record: dict) -> dict:
        """Keep a record: append it to results.jsonl, print it and update the status."""
        self.records.append(record)
        with self.results.open("a") as f:
            f.write(to_json(record) + "\n")
        print(RESULT_PREFIX + to_json(record), flush=True)
        self.status("running")
        return record

    def measure(self, spec: dict, until: float) -> dict:
        """Run one measurement before ``until``, and keep its record.

        Args:
            spec: The measurement, as benchmarks.gpu.measure takes it.
            until: When it must have finished.

        Returns:
            dict: Its record, with ``status`` ``ok``, ``oom``, ``timeout``,
            ``error``, ``failed`` or ``skipped``.
        """
        spec = {"plan": str(self.config.plan), **spec}
        base = {key: spec.get(key) for key in ("phase", "kind", "method", "N", "K")}
        timeout = until - time.time()
        if timeout < MIN_MEASUREMENT_S:
            return self.record({**base, "status": "skipped", "reason": "time budget"})
        command = [sys.executable, "-m", "benchmarks.gpu.measure", json.dumps(spec)]
        began = time.time()
        try:
            proc = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout, cwd=REPO_DIR
            )
        except subprocess.TimeoutExpired:
            return self.record(
                {**base, "status": "timeout", "wall_s": time.time() - began}
            )
        with self.log.open("a") as f:
            entry = f"--- {utc_now()} {json.dumps(spec)}\n{proc.stderr[-20000:]}\n"
            f.write(home_as_tilde(entry))
        lines = [l for l in proc.stdout.splitlines() if l.startswith(RESULT_PREFIX)]
        if proc.returncode == 0 and lines:
            return self.record({**base, **json.loads(lines[-1][len(RESULT_PREFIX) :])})
        oom = any(marker in proc.stderr for marker in _OOM_MARKERS)
        tail = proc.stderr.strip().splitlines()[-3:]
        return self.record(
            {
                **base,
                "status": "oom" if oom else "error",
                "reason": " | ".join(tail)[-2000:],
                "wall_s": time.time() - began,
            }
        )

    def _method_shares(self, phase: dict) -> list[tuple[str, float]]:
        """Each method's share of the phase, as it's reached."""
        shares = phase["methods"]
        return [
            (method, share / sum(shares.values())) for method, share in shares.items()
        ]

    def run_check(self, phase: dict) -> bool:
        """Check the device and float64. Returns whether the plan can go on."""
        spec = {"phase": phase["name"], "kind": "check", "backend": self.config.backend}
        spec["device_kind"] = GPU_TYPES.get(self.config.gpu or "")
        return self.measure(spec, self.until(phase))["status"] == "ok"

    def run_correctness(self, phase: dict) -> None:
        """Compare the device's fits with the CPU's, one method per measurement."""
        end = self.until(phase)
        for method in phase["methods"]:
            spec = {"phase": phase["name"], "kind": "correctness", "method": method}
            spec |= {"N": phase["datasets"], "K": self.plan["num_terms"]}
            self.measure(spec | {"methods": [method]}, end)

    def run_batches(self, phase: dict) -> None:
        """Fit each method's batch sizes in turn, until memory or time runs out.

        A size is skipped, with the rest, when its predicted wall time doesn't fit in
        what's left of the method's share.
        """
        end = self.until(phase)
        methods = self._method_shares(phase)
        for i, (method, _) in enumerate(methods):
            left = sum(share for _, share in methods[i:])
            until = time.time() + (end - time.time()) * methods[i][1] / left
            last = None
            for n in phase["sizes"]:
                spec = {"phase": phase["name"], "kind": "fit", "method": method}
                spec |= {"N": n, "K": self.plan["num_terms"]}
                if last:
                    need = GROWTH_MARGIN * predicted_wall(
                        last,
                        n / last["N"],
                        self.plan["repetitions"],
                        self.plan["repetition_budget_s"],
                    )
                    if need > until - time.time():
                        self.record(
                            {**spec, "status": "skipped", "reason": "time budget"}
                        )
                        break
                last = self.measure(spec, until)
                if last["status"] != "ok":
                    break

    def batches(self, mode: str) -> list[dict]:
        """The successful single-batch records of one mode."""
        return [
            r
            for r in self.records
            if r.get("phase") == "batches"
            and r.get("mode") == mode
            and r["status"] == "ok"
        ]

    def run_chunked(self, phase: dict) -> None:
        """Fit chunks of each method's throughput knee for about phase['seconds']."""
        end = self.until(phase)
        methods = self._method_shares(phase)
        for i, (method, _) in enumerate(methods):
            left = sum(share for _, share in methods[i:])
            until = time.time() + (end - time.time()) * methods[i][1] / left
            point = knee(
                self.batches("feasible" if method == "feasible" else "ols_grid"),
                phase["knee_tolerance"],
            )
            spec = {"phase": phase["name"], "kind": "chunked", "method": method}
            spec["K"] = self.plan["num_terms"]
            if point is None:
                self.record(
                    {**spec, "status": "skipped", "reason": "no single-batch result"}
                )
                continue
            # Time for the chunks: what's left after starting and compiling, as the knee's
            # measurement took
            overhead = point["wall_s"] - point["steady_s"] * (point["repetitions"] + 1)
            seconds = min(
                phase["seconds"],
                until - time.time() - max(overhead, 0) - MIN_MEASUREMENT_S,
            )
            chunks = max(phase["min_chunks"], math.floor(seconds / point["steady_s"]))
            spec |= {"N": chunks * point["N"], "chunk": point["N"], "time_s": seconds}
            self.measure(spec, until)

    def run_terms(self, phase: dict) -> None:
        """Feasible at each term count, then all of them in one grid."""
        end = self.until(phase)
        terms = phase["num_terms"]
        for k in terms:
            spec = {"phase": phase["name"], "kind": "fit", "method": "feasible"}
            spec |= {"N": phase["datasets"], "K": k}
            self.measure(spec, end)
        spec = {"phase": phase["name"], "kind": "fit", "method": "feasible"}
        spec |= {"N": phase["datasets"], "K": max(terms), "num_terms": terms}
        self.measure(spec, end)

    def run_sampling(self, phase: dict) -> None:
        """Each sampling kind at growing sizes, until one fails."""
        end = self.until(phase)
        for kind, sizes in phase["kinds"].items():
            for n in sizes:
                spec = {
                    "phase": phase["name"],
                    "kind": "sampling",
                    "sampling": kind,
                    "N": n,
                }
                if self.measure(spec, end)["status"] != "ok":
                    break

    def run(self) -> str:
        """Run every phase; return the final state, ``done`` or ``failed``."""
        phases = {
            "check": self.run_check,
            "correctness": self.run_correctness,
            "batches": self.run_batches,
            "chunked": self.run_chunked,
            "terms": self.run_terms,
            "sampling": self.run_sampling,
        }
        for phase in self.plan["phases"]:
            self.phase = phase["name"]
            self.status("running")
            if phases[phase["name"]](phase) is False:
                return "failed"
        return "done"


class Telemetry:
    """Sample GPU utilization, memory, power and clocks with nvidia-smi."""

    def __init__(self, path: Path):
        """Start sampling every half second, if nvidia-smi is installed."""
        self.proc = None
        if shutil.which("nvidia-smi"):
            self.file = path.open("w")
            self.proc = subprocess.Popen(
                [
                    "nvidia-smi",
                    f"--query-gpu={_TELEMETRY_FIELDS}",
                    "--format=csv",
                    "-lms",
                    "500",
                ],
                stdout=self.file,
                stderr=subprocess.DEVNULL,
            )

    def stop(self) -> None:
        """Stop sampling."""
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=10)
            self.file.close()


def finish(config: Config, manifest: dict, state: str) -> dict:
    """Write the manifest's end and the checksums; return the files and digests."""
    manifest["ended"] = utc_now()
    manifest["state"] = state
    write_json(config.results / "manifest.json", manifest)
    names = sorted(
        p.name
        for p in config.results.iterdir()
        if p.is_file()
        and p.name not in ("status.json", "checksums.sha256")
        and not p.name.endswith(".tmp")
    )
    digests = {name: sha256_file(config.results / name) for name in names}
    lines = "".join(f"{digest}  {name}\n" for name, digest in digests.items())
    (config.results / "checksums.sha256").write_text(lines)
    digests["checksums.sha256"] = sha256_file(config.results / "checksums.sha256")
    return digests


def load_plan(config: Config) -> tuple[dict, str]:
    """The plan and its SHA-256, saved with the results, where measurements read it."""
    text = config.plan_json or config.plan.read_text()
    config.plan = config.results / "plan.json"
    config.plan.write_text(text)
    return json.loads(text), sha256_file(config.plan)


def main() -> int:
    """Run the plan, serve the results, then leave."""
    config = Config.from_environment()
    config.results.mkdir(parents=True, exist_ok=True)
    plan, plan_sha256 = load_plan(config)
    server = None
    if not config.local:
        server = ResultsServer(config.results, config.token, config.port).start()
    runner = Runner(config, plan)
    runner.status("starting", deadline=config.deadline)
    manifest = environment.capture(config.results, plan, plan_sha256)
    manifest |= {
        "started": utc_now(),
        "deadline": config.deadline,
        "gpu_type": config.gpu,
    }
    write_json(config.results / "manifest.json", manifest)
    telemetry = Telemetry(config.results / "telemetry.csv")
    try:
        state = runner.run()
    except Exception as error:
        print(f"benchmark failed: {error!r}", flush=True)
        state = "failed"
    finally:
        telemetry.stop()
    files = finish(config, manifest, state)
    runner.status(state, files=files)
    print(f"benchmark {state}: {len(runner.records)} measurements", flush=True)
    if server is None:
        return 0
    server.acknowledged.wait(max(config.deadline + config.download_s - time.time(), 0))
    server.stop()
    if not server.acknowledged.is_set():
        print("results weren't acknowledged; leaving", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
