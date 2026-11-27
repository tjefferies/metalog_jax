# Copyright: Travis Jefferies 2026
"""Tests for the Runpod launcher, against a fake Runpod API and pod.

Nothing here reaches Runpod: the API is a local HTTP server that records requests,
and each pod is a ResultsServer serving a results directory.
"""

import argparse
import contextlib
import hashlib
import io
import json
import tempfile
import threading
import time
import urllib.error
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from absl.testing import absltest

from benchmarks.gpu import common, launch
from benchmarks.gpu.common import GPU_TYPES
from benchmarks.gpu.serve import ResultsServer

PRICES = {"a100-sxm": (1.79, 1.39), "h100-sxm": (3.99, 2.69), "h200-sxm": (5.29, 3.59)}
CATALOG = [
    {
        "id": GPU_TYPES[slug],
        "price": {"secure": secure, "community": community},
        "availability": "HIGH",
    }
    for slug, (secure, community) in PRICES.items()
]


class FakeRunpod:
    """A local stand-in for Runpod's REST API v2, recording every request."""

    def __init__(self, catalog=CATALOG, billing=0.5, throttle_once=False):
        """Serve on a free port."""
        self.catalog, self.billing = catalog, billing
        self.throttle = throttle_once
        self.requests: list[tuple[str, str, dict | None]] = []
        self.pods: dict[str, dict] = {}
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, status, body=None, headers=None):
                data = b"" if body is None else json.dumps(body).encode()
                self.send_response(status)
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle(self, method):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                fake.requests.append((method, self.path, body))
                status, reply, headers = fake.route(method, self.path, body)
                self._reply(status, reply, headers)

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

            def do_DELETE(self):
                self._handle("DELETE")

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_port}/v2"

    def route(self, method, path, body):
        """The fake API's response to a request."""
        route = path.split("?")[0]
        if self.throttle:
            self.throttle = False
            return 429, {"detail": "slow down"}, {"Retry-After": "0"}
        if route == "/v2/catalog/gpus":
            return 200, {"gpus": self.catalog}, None
        if route == "/v2/pods" and method == "POST":
            pod_id = f"pod{len(self.pods)}"
            self.pods[pod_id] = {
                "id": pod_id,
                "name": body["name"],
                "cost": 2.0,
                "env": body["env"],
            }
            return 201, self.pods[pod_id], None
        if route == "/v2/pods" and method == "GET":
            return 200, list(self.pods.values()), None
        if route.startswith("/v2/pods/") and method == "GET":
            pod = self.pods.get(route.rsplit("/", 1)[1])
            return (200, {"status": "RUNNING", **pod}, None) if pod else (404, {}, None)
        if route.startswith("/v2/pods/") and method == "DELETE":
            pod_id = route.rsplit("/", 1)[1]
            return (204, None, None) if self.pods.pop(pod_id, None) else (404, {}, None)
        if route == "/v2/billing/pods":
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)
            if ("startTime" in query) != ("endTime" in query):
                detail = "startTime and endTime must be provided together"
                return 400, {"detail": detail}, None
            totals = {"totalAmount": self.billing}
            return 200, {"records": [], "metadata": {"totals": totals}}, None
        return 404, {"detail": route}, None

    def deleted(self) -> list[str]:
        """The pods the launcher deleted, in order."""
        return [p.rsplit("/", 1)[1] for m, p, _ in self.requests if m == "DELETE"]

    def close(self):
        """Stop serving."""
        self.httpd.shutdown()


class RefusingProxy:
    """A local stand-in for a proxy that refuses every request, as Cloudflare did."""

    def __init__(self):
        """Serve on a free port, recording each request's user agent."""
        self.user_agents: list[str] = []
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                proxy.user_agents.append(self.headers.get("User-Agent"))
                body = b"error code: 1010"
                self.send_response(403)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"

    def close(self):
        """Stop serving."""
        self.httpd.shutdown()


def make_results(directory: Path, state="done", corrupt=None) -> dict:
    """Write a finished pod's results and status.json; return the file digests."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {"results.jsonl": b'{"status": "ok"}\n', "manifest.json": b"{}\n"}
    digests = {}
    for name, body in files.items():
        (directory / name).write_bytes(body)
        digests[name] = hashlib.sha256(body).hexdigest()
    if corrupt:
        digests[corrupt] = "0" * 64
    (directory / "status.json").write_text(
        json.dumps({"state": state, "files": digests})
    )
    return digests


class PlanRunTest(absltest.TestCase):
    """The time cap keeps the run's list-price cost within 92% of the budget."""

    def test_secure_cloud_cap(self):
        """At the 2026-10-30 list prices, $5 buys 24 minutes of all three on Secure Cloud."""
        plan = launch.plan_run(CATALOG, list(PRICES), 5.0, "SECURE")
        self.assertEqual(plan.cap_minutes, 24)
        self.assertAlmostEqual(plan.estimated_cost, 4.428, places=3)
        self.assertLessEqual(plan.estimated_cost, launch.BUDGET_SHARE * 5.0)

    def test_community_cloud_cap(self):
        """Community Cloud's lower prices buy 35 minutes."""
        plan = launch.plan_run(CATALOG, list(PRICES), 5.0, "COMMUNITY")
        self.assertEqual(plan.cap_minutes, 35)
        self.assertAlmostEqual(plan.estimated_cost, 4.474, places=3)

    def test_unavailable_gpu_is_skipped(self):
        """A GPU type without availability is left out, and the others share the cap."""
        catalog = [dict(g) for g in CATALOG]
        catalog[2]["availability"] = "NONE"
        plan = launch.plan_run(catalog, list(PRICES), 5.0, "SECURE")
        self.assertEqual([g.slug for g in plan.gpus], ["a100-sxm", "h100-sxm"])
        self.assertEqual(plan.unavailable, ["h200-sxm"])
        self.assertEqual(plan.cap_minutes, 47)

    def test_too_small_a_budget_raises(self):
        """A cap under MIN_CAP_MINUTES isn't worth running."""
        with self.assertRaisesRegex(ValueError, "at least 15 are needed"):
            launch.plan_run(CATALOG, list(PRICES), 2.0, "SECURE")

    def test_unknown_gpu_raises(self):
        """GPU types are the three SXM cards."""
        with self.assertRaisesRegex(ValueError, "unknown GPU type"):
            launch.plan_run(CATALOG, ["rtx-4090"], 5.0, "SECURE")


class PodBodyTest(absltest.TestCase):
    """Each pod gets the digest, one GPU, the CUDA floor, the port and its secrets."""

    GPU = launch.GpuPlan("h100-sxm", GPU_TYPES["h100-sxm"], 3.99, "HIGH")
    IMAGE = "ghcr.io/tjefferies/metalog_jax/gpu-bench@sha256:" + "a" * 64

    def test_body(self):
        """The body follows POST /v2/pods, and passes the plan."""
        body = launch.pod_body(
            "run1", self.GPU, self.IMAGE, "SECURE", "12.9", 1e9, "tok", '{"version":2}'
        )
        self.assertEqual(body["name"], "metalog-bench-run1-h100-sxm")
        self.assertEqual(body["image"], self.IMAGE)
        self.assertEqual(
            body["gpu"],
            {"id": "NVIDIA H100 80GB HBM3", "count": 1, "minCudaVersion": "12.9"},
        )
        self.assertEqual((body["cloud"], body["ports"]), ("SECURE", ["8000/http"]))
        self.assertEqual(body["env"]["BENCH_DEADLINE"], "1000000000")
        self.assertEqual(body["env"]["BENCH_TOKEN"], "tok")
        self.assertEqual(body["env"]["BENCH_GPU"], "h100-sxm")
        self.assertEqual(body["env"]["BENCH_PLAN_JSON"], '{"version":2}')

    def test_the_plan_is_one_line(self):
        """The plan goes to the pods as one line of JSON, with the file's content."""
        text = launch.plan_json()
        self.assertNotIn("\n", text)
        self.assertEqual(json.loads(text), json.loads(launch.DEFAULT_PLAN.read_text()))

    def test_tags_are_refused(self):
        """Runs use a digest, so that they're reproducible."""
        with self.assertRaisesRegex(ValueError, "by digest"):
            launch.check_image("ghcr.io/tjefferies/metalog_jax/gpu-bench:latest")
        launch.check_image(self.IMAGE)

    def test_the_attestation_tag_names_the_image(self):
        """The sha256-<digest> tag is the attestation; the error gives the image."""
        tag = self.IMAGE.replace("@sha256:", ":sha256-")
        with self.assertRaises(ValueError) as error:
            launch.check_image(tag)
        self.assertIn("provenance attestation", str(error.exception))
        self.assertIn(f"pass {self.IMAGE}", str(error.exception))

    def test_cuda_floor_comes_from_the_lock(self):
        """The hosts' minimum CUDA is the runtime uv.lock pins, as major.minor."""
        version = launch.cuda_runtime()
        major, minor = version.split(".")
        self.assertGreaterEqual(int(major), 12)
        self.assertTrue(minor.isdigit())


class ClientTest(absltest.TestCase):
    """The client retries rate limits and reads Runpod's responses."""

    def test_rate_limit_is_retried(self):
        """A 429 is retried after Retry-After."""
        api = FakeRunpod(throttle_once=True)
        self.addCleanup(api.close)
        client = launch.RunpodClient("key", base=api.base)
        self.assertLen(client.gpu_catalog("SECURE", "12.9"), 3)
        self.assertEqual([m for m, _, _ in api.requests], ["GET", "GET"])

    def test_deleting_a_missing_pod(self):
        """A pod that's already gone isn't an error."""
        api = FakeRunpod()
        self.addCleanup(api.close)
        self.assertFalse(launch.RunpodClient("key", base=api.base).delete_pod("nope"))


class LauncherTest(absltest.TestCase):
    """A run creates, collects and terminates its pods, and records what it cost."""

    def setUp(self):
        """A fake API, and a results server standing in for every pod's proxy."""
        super().setUp()
        self.api = FakeRunpod()
        self.addCleanup(self.api.close)
        self.out = Path(self.enter_context(tempfile.TemporaryDirectory()))
        self.pod_dir = Path(self.enter_context(tempfile.TemporaryDirectory()))
        self.client = launch.RunpodClient("key", base=self.api.base)
        self.plan = launch.plan_run(CATALOG, ["h100-sxm"], 5.0, "SECURE")

    def launcher(self, token_from_body=True):
        """A launcher whose pods all answer at the local results server."""
        launcher = launch.Launcher(
            self.client, self.plan, PodBodyTest.IMAGE, self.out, poll_s=0.01
        )
        self.server = None

        def url_for(_pod_id):
            if self.server is None:
                token = launcher.pods[0].token
                self.server = ResultsServer(self.pod_dir, token, 0).start()
                self.addCleanup(self.server.stop)
            return f"http://127.0.0.1:{self.server.httpd.server_port}"

        launcher.url_for = url_for
        return launcher

    def test_results_are_collected_and_the_pod_terminated(self):
        """Done results are downloaded, checked and acknowledged; the pod is deleted."""
        digests = make_results(self.pod_dir)
        launcher = self.launcher()
        launcher.create("12.9")
        launcher.watch()
        pod = launcher.pods[0]
        self.assertEqual(pod.state, "collected")
        for name, digest in digests.items():
            body = (pod.out / name).read_bytes()
            self.assertEqual(hashlib.sha256(body).hexdigest(), digest)
        self.assertTrue(self.server.acknowledged.is_set())
        self.assertEqual(self.api.deleted(), ["pod0"])
        record = launcher.record(
            launch.utc_now(), launcher.charges(launch.utc_now(), 0)
        )
        self.assertEqual(record["total_billed"], 0.5)
        self.assertEqual(record["billing_errors"], {})
        self.assertEqual(record["pods"][0]["state"], "collected")
        self.assertTrue((self.out / "launcher.json").exists())

    def test_billing_asks_for_a_window(self):
        """The billing request has both ends of its window, as Runpod requires."""
        launcher = self.launcher()
        launcher.create("12.9")
        launcher.terminate_all()
        self.assertEqual(launcher.charges(launch.utc_now(), 0), {"h100-sxm": 0.5})
        path = next(p for m, p, _ in self.api.requests if "/billing/pods" in p)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)
        self.assertEqual(sorted(query), ["bucketSize", "endTime", "podId", "startTime"])

    def test_the_record_leaves_out_the_token_and_the_plan(self):
        """launcher.json keeps the pod's settings, but not its token or the plan."""
        launcher = self.launcher()
        launcher.create("12.9")
        launcher.terminate_all()
        record = launcher.record(launch.utc_now(), {"h100-sxm": None})
        env = record["pods"][0]["pod"]["env"]
        self.assertNotIn("BENCH_TOKEN", env)
        self.assertNotIn("BENCH_PLAN_JSON", env)
        self.assertEqual(env["BENCH_GPU"], "h100-sxm")
        digest = hashlib.sha256(launcher.bench_plan.encode()).hexdigest()
        self.assertEqual(record["plan_sha256"], digest)
        self.assertIn("BENCH_TOKEN", self.api.requests[0][2]["env"])

    def test_a_bad_checksum_is_retried_until_the_cap(self):
        """A download that doesn't match is retried, said once, and ended at the cap."""
        make_results(self.pod_dir, corrupt="results.jsonl")
        launcher = self.launcher()
        launcher.create("12.9")
        launcher.pods[0].deadline = time.time() + 1
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            launcher.watch()
        self.assertEqual(launcher.pods[0].state, "cap reached")
        self.assertIn("checksum", launcher.pods[0].error)
        self.assertEqual(out.getvalue().count("downloading failed, retrying"), 1)
        self.assertFalse(self.server.acknowledged.is_set())
        self.assertEqual(self.api.deleted(), ["pod0"])

    def test_a_failed_ack_still_collects(self):
        """Checked results are collected, and the pod deleted, though the ack fails."""
        digests = make_results(self.pod_dir)
        respond = ResultsServer.respond

        def refuse_ack(server, path):
            if path.rstrip("/").endswith("/ack"):
                return 404, b"not found\n", "text/plain"
            return respond(server, path)

        self.enter_context(mock.patch.object(ResultsServer, "respond", refuse_ack))
        launcher = self.launcher()
        launcher.create("12.9")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            launcher.watch()
        pod = launcher.pods[0]
        self.assertEqual((pod.state, pod.error), ("collected", None))
        self.assertIn("the ack failed (HTTP Error 404", out.getvalue())
        for name, digest in digests.items():
            body = (pod.out / name).read_bytes()
            self.assertEqual(hashlib.sha256(body).hexdigest(), digest)
        self.assertEqual(self.api.deleted(), ["pod0"])

    def test_the_cap_terminates_a_pod_without_results(self):
        """A pod that's still running at its cap is deleted."""
        launcher = self.launcher()
        launcher.create("12.9")
        launcher.pods[0].deadline = 0
        launcher.watch()
        self.assertEqual(launcher.pods[0].state, "cap reached")
        self.assertEqual(self.api.deleted(), ["pod0"])

    def test_an_exited_pod_without_results_is_terminated(self):
        """A pod whose container exited before serving results isn't waited for."""
        launcher = self.launcher()
        launcher.create("12.9")
        self.api.pods["pod0"]["status"] = "EXITED"
        launcher.watch()
        self.assertEqual(launcher.pods[0].state, "pod exited without results")
        self.assertEqual(self.api.deleted(), ["pod0"])

    def test_a_refusing_proxy_ends_the_pod_early(self):
        """Refusals stop the pod after REFUSALS_TO_STOP polls, not at its cap."""
        proxy = RefusingProxy()
        self.addCleanup(proxy.close)
        launcher = launch.Launcher(
            self.client, self.plan, PodBodyTest.IMAGE, self.out, poll_s=0.01
        )
        launcher.url_for = lambda _pod_id: proxy.url
        launcher.create("12.9")
        launcher.watch()
        pod = launcher.pods[0]
        self.assertEqual(pod.state, "proxy refused")
        self.assertIn("HTTP 403 error code: 1010", pod.error)
        self.assertLen(proxy.user_agents, launch.REFUSALS_TO_STOP)
        self.assertEqual(self.api.deleted(), ["pod0"])

    def test_requests_name_the_tool(self):
        """Every request sends USER_AGENT, not urllib's, which Cloudflare refuses."""
        proxy = RefusingProxy()
        self.addCleanup(proxy.close)
        with self.assertRaises(urllib.error.HTTPError):
            launch.fetch(proxy.url + "/status.json")
        self.assertEqual(proxy.user_agents, [common.USER_AGENT])

    def test_progress_is_printed_when_it_changes(self):
        """Each new measurement in status.json is printed once."""
        self.pod_dir.mkdir(parents=True, exist_ok=True)
        status = {"state": "running", "phase": "batches", "measurements": 3}
        status["last"] = "fit feasible N=1024 K=10: ok in 29 s"
        (self.pod_dir / "status.json").write_text(json.dumps(status))
        launcher = self.launcher()
        launcher.create("12.9")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            launcher.step(launcher.pods[0])
            launcher.step(launcher.pods[0])
        launcher.terminate_all()
        line = "h100-sxm: batches, 3 measurements; last: fit feasible N=1024 K=10: ok in 29 s"
        self.assertEqual(out.getvalue().splitlines(), [line])
        self.assertEqual(launcher.pods[0].state, "running")

    def test_unreported_charges_have_no_billed_total(self):
        """Without every pod's charges, the total is None, not $0.00."""
        launcher = self.launcher()
        launcher.create("12.9")
        launcher.terminate_all()
        started = launch.utc_now()
        record = launcher.record(started, {"h100-sxm": None})
        self.assertIsNone(record["total_billed"])
        self.assertGreaterEqual(record["total_estimated"], 0)
        self.assertEqual(
            launcher.record(started, {"h100-sxm": 0.5})["total_billed"], 0.5
        )

    def test_terminate_all_deletes_each_pod_once(self):
        """On exit every created pod is deleted, and none twice."""
        launcher = self.launcher()
        launcher.create("12.9")
        launcher.terminate_all()
        launcher.terminate_all()
        self.assertEqual(self.api.deleted(), ["pod0"])

    def test_dry_run_creates_nothing(self):
        """--dry-run prints the plan from a saved catalog and creates no pod."""
        prices = self.out / "prices.json"
        prices.write_text(json.dumps(CATALOG))
        args = launch.parser().parse_args(
            ["run", "--image", PodBodyTest.IMAGE, "--prices", str(prices), "--dry-run"]
        )
        self.assertEqual(launch.run(args, self.client), 0)
        self.assertEqual(self.api.requests, [])


class CleanupTest(absltest.TestCase):
    """cleanup deletes only the benchmark's pods."""

    def test_only_benchmark_pods(self):
        """Pods named metalog-bench-* are deleted; others are left alone."""
        api = FakeRunpod()
        self.addCleanup(api.close)
        api.pods = {
            "a": {"id": "a", "name": "metalog-bench-run-h100-sxm"},
            "b": {"id": "b", "name": "someone-elses-pod"},
        }
        client = launch.RunpodClient("key", base=api.base)
        launch.cleanup(argparse.Namespace(), client)
        self.assertEqual(api.deleted(), ["a"])
        self.assertIn("b", api.pods)


if __name__ == "__main__":
    absltest.main()
