"""Run the GPU benchmark on Runpod: one pod per GPU type, within a budget.

Usage:
    export RUNPOD_API_KEY=...
    uv run python benchmarks/gpu/launch.py run --image ghcr.io/...@sha256:...
        [--gpus a100-sxm,h100-sxm,h200-sxm] [--budget 5.00] [--cloud SECURE] [--dry-run]
        [--plan benchmarks/gpu/plan.json]
    uv run python benchmarks/gpu/launch.py cleanup

``run`` reads Runpod's live prices and gives all the pods one time cap, so that the
sum of each GPU's price times the cap stays within BUDGET_SHARE of the budget. It
creates the pods, passing each the plan, prints their progress, downloads and checks
their results, and terminates each pod when its results are in or its cap is
reached, and every pod it created when it exits. Then it reads the charges and fails
if they exceed the budget. ``cleanup`` terminates any pod left over from a run.

Uses only the standard library and Runpod's REST API v2
(https://docs.runpod.io/api-reference-v2/overview).
"""

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import secrets
import signal
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

if __package__ in (None, ""):  # run as a script: make the repository importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.gpu.common import GPU_TYPES, open_url, utc_now, write_json

API = "https://api.runpod.io/v2"
REPO_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_DIR / "benchmarks" / "gpu" / "results"
DEFAULT_PLAN = REPO_DIR / "benchmarks" / "gpu" / "plan.json"
POD_PREFIX = "metalog-bench-"
PORT = 8000

# Of the budget, the share the pods' compute may use; the rest covers provisioning,
# the container disk and rounding
BUDGET_SHARE = 0.92
# A cap shorter than this leaves too little time to measure anything
MIN_CAP_MINUTES = 15
# Seconds of a pod's cap kept for downloading its results after its deadline: a
# download takes seconds, and this leaves room for a poll and a few retries
DOWNLOAD_RESERVE_S = 120
CONTAINER_DISK_GB = 20
POLL_S = 15
# How long past the last cap to keep trying to terminate pods
TERMINATE_GRACE_S = 600
# How long to wait for Runpod's billing records of a finished run. They lag, so
# this tries a few times, then the run reports its estimate
BILLING_WAIT_S = 30
_RETRIES = 5
# Statuses that mean the proxy refuses to serve a pod's results at all, rather than
# that the pod isn't serving them yet
_REFUSED = frozenset({401, 403})
# This many refusals in a row end a pod early: it can't be collected
REFUSALS_TO_STOP = 4
# The tag GHCR gives an image's attestation: sha256-<the image's digest>
_ATTESTATION_TAG = re.compile(r":sha256-([0-9a-f]{64})$")
# Pod environment variables left out of launcher.json: the results' secret path, and
# the plan, which launcher.json records by its SHA-256
_UNRECORDED_ENV = ("BENCH_TOKEN", "BENCH_PLAN_JSON")


class RunpodError(RuntimeError):
    """A Runpod API request failed."""

    def __init__(self, status: int, detail: str):
        """Keep the HTTP status and the response's detail."""
        super().__init__(f"Runpod API returned {status}: {detail}")
        self.status = status


class RunpodClient:
    """A minimal client for Runpod's REST API v2."""

    def __init__(self, api_key: str, base: str = API, timeout: float = 30.0):
        """Authenticate every request with the API key."""
        self.api_key, self.base, self.timeout = api_key, base.rstrip("/"), timeout

    def request(self, method: str, path: str, body=None, query: dict | None = None):
        """Send a request, retrying rate limits and server errors.

        Args:
            method: The HTTP method.
            path: The path under the API's base, such as ``/pods``.
            body: A JSON body, or None.
            query: Query parameters, or None.

        Returns:
            The decoded JSON response, or None for an empty one.

        Raises:
            RunpodError: For a client error, or a server error after the retries.
        """
        url = self.base + path
        if query:
            url += "?" + urllib.parse.urlencode(query, doseq=True)
        data = None if body is None else json.dumps(body).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        for attempt in range(_RETRIES):
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with open_url(req, timeout=self.timeout) as response:
                    text = response.read().decode()
                    return json.loads(text) if text.strip() else None
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:500]
                if error.code == 429 or error.code >= 500:
                    if attempt + 1 < _RETRIES:
                        wait = float(error.headers.get("Retry-After") or 2**attempt)
                        time.sleep(min(wait, 60))
                        continue
                raise RunpodError(error.code, detail) from None
        raise RunpodError(0, "no response")  # pragma: no cover

    def gpu_catalog(self, cloud: str, min_cuda: str | None) -> list[dict]:
        """GPU types with prices and availability for pods on a cloud."""
        query = {"include": "AVAILABILITY", "product": "POD", "cloud": cloud}
        if min_cuda:
            query["minCudaVersion"] = min_cuda
        return self.request("GET", "/catalog/gpus", query=query)["gpus"]

    def create_pod(self, body: dict) -> dict:
        """Create a pod."""
        return self.request("POST", "/pods", body=body)

    def get_pod(self, pod_id: str) -> dict:
        """A pod's current record."""
        return self.request("GET", f"/pods/{pod_id}")

    def delete_pod(self, pod_id: str) -> bool:
        """Terminate a pod; True if it was there."""
        try:
            self.request("DELETE", f"/pods/{pod_id}")
        except RunpodError as error:
            if error.status == 404:
                return False
            raise
        return True

    def list_pods(self) -> list[dict]:
        """Every pod on the account."""
        response = self.request("GET", "/pods")
        if isinstance(response, dict):
            return response.get("pods") or response.get("data") or []
        return response or []

    def pod_billing(self, pod_id: str, start: str, end: str) -> dict:
        """A pod's charges by the hour from ``start`` to ``end`` (RFC 3339)."""
        query = {
            "podId": pod_id,
            "startTime": start,
            "endTime": end,
            "bucketSize": "hour",
        }
        return self.request("GET", "/billing/pods", query=query)


@dataclasses.dataclass
class GpuPlan:
    """One GPU type's part of a run."""

    slug: str
    gpu_id: str
    price: float
    availability: str | None


@dataclasses.dataclass
class RunPlan:
    """The pods to create, their shared time cap and the estimated cost."""

    gpus: list[GpuPlan]
    cloud: str
    budget: float
    cap_minutes: int
    unavailable: list[str]

    @property
    def estimated_cost(self) -> float:
        """The pods' compute at list price, for the whole cap."""
        return sum(g.price for g in self.gpus) * self.cap_minutes / 60


def plan_run(
    catalog: list[dict], slugs: list[str], budget: float, cloud: str
) -> RunPlan:
    """Choose the time cap for the GPU types the catalog has available.

    Args:
        catalog: Runpod's GPU catalog, with prices and availability.
        slugs: The GPU types to run, such as ``h100-sxm``.
        budget: The most the run may cost, in USD.
        cloud: ``SECURE`` or ``COMMUNITY``.

    Returns:
        RunPlan: The available GPUs and their cap.

    Raises:
        ValueError: For an unknown GPU type, or a cap under MIN_CAP_MINUTES.
    """
    by_id = {gpu["id"]: gpu for gpu in catalog}
    price_key = "secure" if cloud == "SECURE" else "community"
    gpus, unavailable = [], []
    for slug in slugs:
        if slug not in GPU_TYPES:
            raise ValueError(
                f"unknown GPU type {slug!r}; choose from {sorted(GPU_TYPES)}"
            )
        entry = by_id.get(GPU_TYPES[slug])
        price = (entry or {}).get("price", {}).get(price_key)
        availability = (entry or {}).get("availability")
        if not price or availability == "NONE":
            unavailable.append(slug)
            continue
        gpus.append(GpuPlan(slug, GPU_TYPES[slug], float(price), availability))
    hourly = sum(g.price for g in gpus)
    cap = math.floor(60 * BUDGET_SHARE * budget / hourly) if hourly else 0
    if gpus and cap < MIN_CAP_MINUTES:
        raise ValueError(
            f"${budget:.2f} buys {cap} minutes of {', '.join(g.slug for g in gpus)} at "
            f"${hourly:.2f}/hour; at least {MIN_CAP_MINUTES} are needed"
        )
    return RunPlan(gpus, cloud, budget, cap, unavailable)


def cuda_runtime(lock: Path = REPO_DIR / "uv.lock") -> str:
    """The CUDA runtime that uv.lock pins, as ``major.minor``: the hosts' minimum."""
    packages = tomllib.loads(lock.read_text())["package"]
    version = next(
        p["version"] for p in packages if p["name"] == "nvidia-cuda-runtime-cu12"
    )
    return ".".join(version.split(".")[:2])


def plan_json(path: Path = DEFAULT_PLAN) -> str:
    """The benchmark plan as one line of JSON, for the pods' BENCH_PLAN_JSON.

    Raises:
        ValueError: If the file isn't valid JSON.
    """  # noqa: DOC502
    return json.dumps(json.loads(path.read_text()), separators=(",", ":"))


def pod_body(
    run_id: str,
    gpu: GpuPlan,
    image: str,
    cloud: str,
    min_cuda: str,
    deadline: float,
    token: str,
    plan: str,
) -> dict:
    """The POST /v2/pods body for one GPU type, with the plan as one line of JSON."""
    return {
        "name": f"{POD_PREFIX}{run_id}-{gpu.slug}",
        "image": image,
        "gpu": {"id": gpu.gpu_id, "count": 1, "minCudaVersion": min_cuda},
        "cloud": cloud,
        "disk": CONTAINER_DISK_GB,
        "ports": [f"{PORT}/http"],
        "env": {
            "BENCH_RUN_ID": run_id,
            "BENCH_GPU": gpu.slug,
            "BENCH_DEADLINE": str(int(deadline)),
            "BENCH_TOKEN": token,
            "BENCH_IMAGE": image,
            "BENCH_PLAN_JSON": plan,
        },
    }


def proxy_url(pod_id: str) -> str:
    """The Runpod proxy's URL for a pod's results port."""
    return f"https://{pod_id}-{PORT}.proxy.runpod.net"


def fetch(url: str, timeout: float = 90.0) -> bytes:
    """GET a URL without credentials (the proxy is public; the token is the path)."""
    with open_url(url, timeout=timeout) as response:
        return response.read()


@dataclasses.dataclass
class PodRun:
    """One pod's progress through a run."""

    gpu: GpuPlan
    out: Path
    token: str
    deadline: float  # when the pod's cap ends: terminate by then
    created_at: str | None = None
    pod_id: str | None = None
    record: dict | None = None
    state: str = "creating"
    error: str | None = None
    refusals: int = 0
    progress: str | None = None
    terminated_at: str | None = None


class Launcher:
    """Create, watch and terminate one run's pods."""

    def __init__(
        self,
        client: RunpodClient,
        run_plan: RunPlan,
        image: str,
        out_dir: Path,
        url_for: Callable[[str], str] = proxy_url,
        poll_s: float = POLL_S,
        bench_plan: str | None = None,
    ):
        """Prepare a run; nothing is created until ``run``.

        ``bench_plan`` is the plan the pods run, as one line of JSON; by default
        DEFAULT_PLAN's.
        """
        self.client, self.plan, self.image = client, run_plan, image
        self.out_dir, self.url_for, self.poll_s = out_dir, url_for, poll_s
        self.bench_plan = plan_json() if bench_plan is None else bench_plan
        self.run_id = (
            dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + secrets.token_hex(2)
        )
        self.pods: list[PodRun] = []
        self.billing_errors: dict[str, str] = {}

    def create(self, min_cuda: str) -> None:
        """Create one pod per GPU type; a type that can't be placed is recorded."""
        cap_s = self.plan.cap_minutes * 60
        for gpu in self.plan.gpus:
            created = time.time()
            pod = PodRun(
                gpu, self.out_dir / gpu.slug, secrets.token_urlsafe(24), created + cap_s
            )
            pod.created_at = utc_now()
            pod.out.mkdir(parents=True, exist_ok=True)
            body = pod_body(
                self.run_id,
                gpu,
                self.image,
                self.plan.cloud,
                min_cuda,
                created + cap_s - DOWNLOAD_RESERVE_S,
                pod.token,
                self.bench_plan,
            )
            try:
                pod.record = self.client.create_pod(body)
                pod.pod_id, pod.state = pod.record["id"], "provisioning"
                print(f"{gpu.slug}: created pod {pod.pod_id}", flush=True)
            except RunpodError as error:
                pod.state, pod.error = "not placed", str(error)
                print(f"{gpu.slug}: not placed: {error}", flush=True)
            self.pods.append(pod)

    def status(self, pod: PodRun) -> dict | None:
        """A pod's status.json, or None while it isn't served.

        A refusal (401 or 403) means the proxy won't serve the results at all, not
        that the pod isn't ready, so it's printed the first time and counted.
        """
        try:
            status = json.loads(
                fetch(f"{self.url_for(pod.pod_id)}/{pod.token}/status.json", 30)
            )
        except urllib.error.HTTPError as error:
            if error.code in _REFUSED:
                detail = error.read().decode(errors="replace").strip()[:200]
                message = f"the proxy refused the request: HTTP {error.code} {detail}"
                if pod.error != message:
                    print(f"{pod.gpu.slug}: {message}", flush=True)
                pod.error, pod.refusals = message, pod.refusals + 1
            return None
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return None
        pod.refusals = 0
        return status

    def report(self, pod: PodRun, status: dict) -> None:
        """Print a pod's progress from its status.json, when it has changed."""
        state = status.get("state")
        stage = status.get("phase") if state == "running" else state
        progress = f"{stage}, {status.get('measurements', 0)} measurements"
        if status.get("last"):
            progress += f"; last: {status['last']}"
        if progress != pod.progress:
            print(f"{pod.gpu.slug}: {progress}", flush=True)
            pod.progress = progress

    def download(self, pod: PodRun, files: dict[str, str]) -> None:
        """Download a pod's results and check each one's SHA-256.

        Raises:
            ValueError: If a file doesn't match its checksum.
        """
        base = f"{self.url_for(pod.pod_id)}/{pod.token}"
        for name, digest in files.items():
            body = fetch(f"{base}/{urllib.parse.quote(name)}")
            if hashlib.sha256(body).hexdigest() != digest:
                raise ValueError(f"{pod.gpu.slug}: {name} doesn't match its checksum")
            (pod.out / name).write_bytes(body)

    def terminate(self, pod: PodRun) -> None:
        """Terminate a pod, once."""
        if pod.pod_id and pod.terminated_at is None:
            try:
                self.client.delete_pod(pod.pod_id)
            except RunpodError as error:
                print(f"{pod.gpu.slug}: terminating failed: {error}", flush=True)
                return
            pod.terminated_at = utc_now()
            print(f"{pod.gpu.slug}: terminated pod {pod.pod_id}", flush=True)

    def step(self, pod: PodRun) -> None:
        """Advance one pod: check its status, collect its results, or end it."""
        if time.time() >= pod.deadline:
            pod.state = "cap reached"
            self.terminate(pod)
            return
        status = self.status(pod)
        if pod.refusals >= REFUSALS_TO_STOP:
            pod.state = "proxy refused"
            self.terminate(pod)
            return
        if status:
            self.report(pod, status)
        if status is None or status.get("state") not in ("done", "failed"):
            try:  # keep the latest record: the data center and host CUDA come later
                pod.record = {**(pod.record or {}), **self.client.get_pod(pod.pod_id)}
            except RunpodError:
                pass
            lifecycle = (pod.record or {}).get("status")
            if status is None and lifecycle and pod.progress != lifecycle:
                print(f"{pod.gpu.slug}: pod {lifecycle.lower()}", flush=True)
                pod.progress = lifecycle
            if lifecycle in ("EXITED", "ERROR", "TERMINATED"):
                pod.state = f"pod {lifecycle.lower()} without results"
                self.terminate(pod)
            elif status:
                pod.state = "running"
            return
        try:
            self.download(pod, status["files"])
            fetch(f"{self.url_for(pod.pod_id)}/{pod.token}/ack", 30)
        except (ValueError, urllib.error.URLError, TimeoutError, OSError) as error:
            # Retried at the next poll, until the pod's cap
            pod.state, pod.error = "download failed", str(error)
            return
        pod.state = "collected" if status["state"] == "done" else "benchmark failed"
        pod.error = None
        self.terminate(pod)

    def watch(self) -> None:
        """Step every pod until all are terminated or weren't placed.

        Gives up TERMINATE_GRACE_S after the last cap, if a pod can't be terminated;
        ``cleanup`` terminates it later.
        """
        while True:
            live = [p for p in self.pods if p.pod_id and p.terminated_at is None]
            for pod in live:
                self.step(pod)
            live = [p for p in live if p.terminated_at is None]
            if not live:
                return
            if time.time() > max(p.deadline for p in live) + TERMINATE_GRACE_S:
                print("some pods couldn't be terminated; run cleanup", flush=True)
                return
            time.sleep(self.poll_s)

    def terminate_all(self) -> None:
        """Terminate every pod this run created."""
        for pod in self.pods:
            self.terminate(pod)

    def charges(self, started: str, wait_s: float = BILLING_WAIT_S) -> dict:
        """Each pod's billed cost, or None if Runpod doesn't report it in time.

        Runpod's billing records lag, so this waits up to ``wait_s``. A request that
        fails is recorded in ``billing_errors``.
        """
        costs, deadline = {}, time.time() + wait_s
        self.billing_errors = {}
        for pod in self.pods:
            if not pod.pod_id:
                continue
            while True:
                amount = None
                try:
                    metadata = (
                        self.client.pod_billing(pod.pod_id, started, utc_now()) or {}
                    )
                    totals = (metadata.get("metadata") or {}).get("totals") or {}
                    amount = totals.get("totalAmount")
                except RunpodError as error:
                    self.billing_errors[pod.gpu.slug] = str(error)
                if amount or time.time() >= deadline:
                    costs[pod.gpu.slug] = amount or None
                    break
                time.sleep(self.poll_s)
        return costs

    def record(self, started: str, costs: dict) -> dict:
        """The launcher's record of the run, written beside the results."""
        estimated = {}
        for pod in self.pods:
            if pod.record and pod.terminated_at:
                created = dt.datetime.fromisoformat(pod.created_at or started)
                ended = dt.datetime.fromisoformat(pod.terminated_at)
                hours = (ended - created).total_seconds() / 3600
                estimated[pod.gpu.slug] = (
                    float(pod.record.get("cost") or pod.gpu.price) * hours
                )
        record = {
            "run_id": self.run_id,
            "image": self.image,
            "plan_sha256": hashlib.sha256(self.bench_plan.encode()).hexdigest(),
            "cloud": self.plan.cloud,
            "budget": self.plan.budget,
            "cap_minutes": self.plan.cap_minutes,
            "started": started,
            "ended": utc_now(),
            "unavailable": self.plan.unavailable,
            "pods": [
                {
                    "gpu": pod.gpu.slug,
                    "gpu_id": pod.gpu.gpu_id,
                    "list_price": pod.gpu.price,
                    "availability": pod.gpu.availability,
                    "pod_id": pod.pod_id,
                    "created_at": pod.created_at,
                    "state": pod.state,
                    "error": pod.error,
                    "terminated_at": pod.terminated_at,
                    "pod": _recorded(pod.record),
                }
                for pod in self.pods
            ],
            "billed": costs,
            "billing_errors": self.billing_errors,
            "estimated": estimated,
        }
        # A total only when Runpod reported every pod's charges
        billed = list(costs.values())
        record["total_billed"] = sum(billed) if billed and all(billed) else None
        record["total_estimated"] = sum(estimated.values())
        write_json(self.out_dir / "launcher.json", record)
        return record


def _recorded(pod_record: dict | None) -> dict | None:
    """A pod's Runpod record without the environment variables _UNRECORDED_ENV names."""
    if not pod_record or not isinstance(pod_record.get("env"), dict):
        return pod_record
    env = {k: v for k, v in pod_record["env"].items() if k not in _UNRECORDED_ENV}
    return {**pod_record, "env": env}


def check_image(image: str) -> None:
    """Runs use an image digest, never a tag, so that every run is reproducible.

    GHCR also lists a ``sha256-<digest>`` tag beside the image: the image's
    provenance attestation, named after the image's digest. The error for it gives
    the image's reference.

    Raises:
        ValueError: If the image isn't pinned by digest.
    """
    if "@sha256:" in image:
        return
    attestation = _ATTESTATION_TAG.search(image)
    if attestation:
        fixed = f"{image[: attestation.start()]}@sha256:{attestation.group(1)}"
        raise ValueError(
            f"{image!r} is the image's provenance attestation, not the image; "
            f"pass {fixed}"
        )
    raise ValueError(f"pass the image by digest (…@sha256:…), not {image!r}")


def _client() -> RunpodClient:
    """A client with the key from RUNPOD_API_KEY.

    Raises:
        SystemExit: If RUNPOD_API_KEY isn't set.
    """
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("set RUNPOD_API_KEY to a Runpod API key")
    return RunpodClient(key)


def run(args: argparse.Namespace, client: RunpodClient | None = None) -> int:
    """Plan the run, then create, watch and terminate the pods.

    Returns:
        int: 0 if the run stayed within budget, 2 if it didn't, 1 for an error.
    """
    check_image(args.image)
    bench_plan = plan_json(Path(args.plan))
    slugs = [s.strip() for s in args.gpus.split(",") if s.strip()]
    if args.prices:
        catalog = json.loads(Path(args.prices).read_text())
    else:
        client = client or _client()
        catalog = client.gpu_catalog(args.cloud, cuda_runtime())
    plan = plan_run(catalog, slugs, args.budget, args.cloud)
    for gpu in plan.gpus:
        print(
            f"{gpu.slug}: {gpu.gpu_id}, ${gpu.price:.2f}/hour, availability {gpu.availability}"
        )
    for slug in plan.unavailable:
        print(f"{slug}: unavailable, skipped")
    print(
        f"cap {plan.cap_minutes} minutes per pod, estimated ${plan.estimated_cost:.2f} "
        f"of a ${plan.budget:.2f} budget ({plan.cloud})"
    )
    digest = hashlib.sha256(bench_plan.encode()).hexdigest()
    print(
        f"plan {args.plan} (version {json.loads(bench_plan).get('version')}, {digest})"
    )
    if args.dry_run:
        return 0
    if not plan.gpus:
        print("no GPU type is available; nothing to run")
        return 1
    client = client or _client()
    launcher = Launcher(
        client,
        plan,
        args.image,
        Path(args.out) if args.out else RESULTS_DIR,
        bench_plan=bench_plan,
    )
    launcher.out_dir = launcher.out_dir / launcher.run_id
    launcher.out_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        launcher.create(cuda_runtime())
        launcher.watch()
    finally:
        launcher.terminate_all()
        costs = launcher.charges(started, args.billing_wait)
        record = launcher.record(started, costs)
    total = record["total_billed"]
    if total is None:
        total = record["total_estimated"]
        print(
            "Runpod didn't report every pod's charges in time; the cost below is "
            "estimated from the pods' hourly rates (see the Runpod console)"
        )
    print(f"results in {launcher.out_dir}; cost ${total:.2f} of ${plan.budget:.2f}")
    for pod in record["pods"]:
        print(
            f"{pod['gpu']}: {pod['state']}"
            + (f" ({pod['error']})" if pod["error"] else "")
        )
    return 2 if total > plan.budget else 0


def cleanup(args: argparse.Namespace, client: RunpodClient | None = None) -> int:
    """Terminate every pod whose name marks it as a benchmark pod."""
    del args
    client = client or _client()
    leftover = [
        p for p in client.list_pods() if str(p.get("name", "")).startswith(POD_PREFIX)
    ]
    for pod in leftover:
        client.delete_pod(pod["id"])
        print(f"terminated {pod['name']} ({pod['id']})")
    if not leftover:
        print("no benchmark pods")
    return 0


def parser() -> argparse.ArgumentParser:
    """The command line."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the benchmark on Runpod")
    r.add_argument("--image", required=True, help="the image, by digest")
    r.add_argument(
        "--gpus", default=",".join(GPU_TYPES), help="comma-separated GPU types"
    )
    r.add_argument(
        "--budget", type=float, default=5.0, help="the most the run may cost, in USD"
    )
    r.add_argument("--cloud", choices=("SECURE", "COMMUNITY"), default="SECURE")
    r.add_argument("--out", help="results directory (default benchmarks/gpu/results)")
    r.add_argument(
        "--dry-run", action="store_true", help="print the plan and cost, create nothing"
    )
    r.add_argument(
        "--prices", help="a saved GPU catalog (JSON), instead of Runpod's live one"
    )
    r.add_argument(
        "--plan",
        default=str(DEFAULT_PLAN),
        help="the benchmark plan the pods run (default benchmarks/gpu/plan.json)",
    )
    r.add_argument(
        "--billing-wait",
        type=float,
        default=BILLING_WAIT_S,
        help="seconds to wait for charges",
    )
    sub.add_parser("cleanup", help="terminate leftover benchmark pods")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run a command."""
    args = parser().parse_args(argv)
    return {"run": run, "cleanup": cleanup}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
