"""Serve a pod's results to the launcher, through Runpod's HTTP proxy.

Files are served only from the results directory, only by name, and only under a
path that starts with the run's random token, because the proxy makes the port
public. The launcher requests ``ack`` after it has checked every file, which tells
the pod it may exit.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ResultsServer:
    """An HTTP server for one results directory, run in a daemon thread."""

    def __init__(self, results_dir: Path, token: str, port: int):
        """Bind to every interface, as Runpod's proxy requires."""
        self.results_dir = results_dir
        self.token = token
        self.acknowledged = threading.Event()
        self.downloads: list[str] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                status, body, kind = server.respond(self.path)
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                pass  # requests are counted, not logged

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def respond(self, path: str) -> tuple[int, bytes, str]:
        """The status, body and content type for a request path.

        Args:
            path: The request path, such as ``/<token>/results.jsonl``.

        Returns:
            tuple: 200 with the file, 200 for ``ack``, or 404.
        """
        parts = path.split("?", 1)[0].strip("/").split("/")
        if len(parts) != 2 or parts[0] != self.token:
            return 404, b"not found\n", "text/plain"
        name = parts[1]
        if name == "ack":
            self.acknowledged.set()
            return 200, b"ok\n", "text/plain"
        file = self.results_dir / name
        if "/" in name or name.startswith(".") or not file.is_file():
            return 404, b"not found\n", "text/plain"
        self.downloads.append(name)
        kind = (
            "application/json" if name.endswith(".json") else "application/octet-stream"
        )
        return 200, file.read_bytes(), kind

    def start(self) -> "ResultsServer":
        """Start serving in the background."""
        self.thread.start()
        return self

    def stop(self) -> None:
        """Stop serving."""
        self.httpd.shutdown()
        self.httpd.server_close()
