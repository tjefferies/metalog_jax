"""Helpers shared by the benchmark's processes, without importing JAX.

run.py imports this module and never JAX, so that the orchestrating process holds no
GPU memory: every measurement gets the whole device.
"""

import datetime as dt
import hashlib
import json
import os
import urllib.request
from pathlib import Path

# The line a measurement process prints its record on, after any JAX warnings
RESULT_PREFIX = "RESULT "

# Runpod's GPU type IDs, which are also the device kinds JAX and nvidia-smi report
GPU_TYPES = {
    "a100-sxm": "NVIDIA A100-SXM4-80GB",
    "h100-sxm": "NVIDIA H100 80GB HBM3",
    "h200-sxm": "NVIDIA H200",
}


def utc_now() -> str:
    """The current UTC time, to the second, in ISO 8601."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def to_json(value) -> str:
    """Serialize a record, converting NumPy and JAX scalars and arrays."""

    def default(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)

    return json.dumps(value, default=default, sort_keys=True)


def write_json(path: Path, value) -> None:
    """Write JSON atomically, so a reader never sees a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json.loads(to_json(value)), indent=1) + "\n")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    """The SHA-256 of a file, in hex."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def knee(records: list[dict], tolerance: float) -> dict | None:
    """The throughput knee: the smallest batch within tolerance of the best time.

    Args:
        records: Successful single-batch records of one method, with ``N`` and
            ``per_dataset_ms``.
        tolerance: How far above the best time per dataset still counts, such as
            0.05 for 5%.

    Returns:
        dict | None: The knee's record, or None without records.
    """
    if not records:
        return None
    best = min(r["per_dataset_ms"] for r in records)
    within = [r for r in records if r["per_dataset_ms"] <= best * (1 + tolerance)]
    return min(within, key=lambda r: r["N"])


# Requests go to Runpod over HTTPS, or to a local server in tests
_ALLOWED_URLS = ("https://", "http://127.0.0.1:", "http://localhost:")


def open_url(request: urllib.request.Request | str, timeout: float):
    """Open an HTTPS request, refusing any other scheme, such as file://.

    Args:
        request: The request or URL.
        timeout: Seconds to wait.

    Returns:
        The response, as urllib.request.urlopen returns it.

    Raises:
        ValueError: For a URL that isn't HTTPS or local.
    """
    url = request.full_url if isinstance(request, urllib.request.Request) else request
    if not url.startswith(_ALLOWED_URLS):
        raise ValueError(f"refusing to open {url!r}")
    # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
    return urllib.request.urlopen(request, timeout=timeout)
