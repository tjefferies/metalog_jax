"""The experimental setup, captured for the manifest without importing JAX.

Records the image, the software, the GPU (from ``nvidia-smi -q -x``), the host and the
cloud. Secrets never reach it: environment variables are kept only by prefix, and
any whose name suggests a credential is dropped.
"""

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys

# nvidia-smi's own output, never untrusted XML
import xml.etree.ElementTree as ET  # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml
from pathlib import Path

# Variables worth recording, by prefix
_ENV_PREFIXES = ("BENCH_", "CUDA_", "JAX_", "NVIDIA_", "RUNPOD_", "TF_", "XLA_")
# Never recorded, whatever their prefix: credentials, and what identifies a person
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
_DROPPED = ("RUNPOD_PUBLIC_IP", "RUNPOD_POD_HOSTNAME")

# Written into the image by its Dockerfile
BUILD_INFO = Path("/opt/metalog_jax/build-info.json")
_KEY_PACKAGES = (
    "metalog-jax",
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "numpy",
    "flax",
    "chex",
)


def safe_environment(environ: dict[str, str] | None = None) -> dict[str, str]:
    """The environment variables worth recording, without any credential.

    Args:
        environ: The environment; ``os.environ`` if None.

    Returns:
        dict[str, str]: The kept variables, sorted by name.
    """
    environ = os.environ if environ is None else environ
    return {
        name: value
        for name, value in sorted(environ.items())
        if name.startswith(_ENV_PREFIXES)
        and name not in _DROPPED
        and not any(marker in name.upper() for marker in _SECRET_MARKERS)
    }


def software() -> dict:
    """Python, the key packages' versions, and every installed distribution."""
    installed = {
        d.metadata["Name"].lower(): d.version
        for d in importlib.metadata.distributions()
    }
    return {
        "python": sys.version,
        "executable": sys.executable,
        "packages": {name: installed.get(name) for name in _KEY_PACKAGES},
        "freeze": dict(sorted(installed.items())),
    }


def _text(root: ET.Element, *paths: str) -> str | None:
    """The first non-empty text among XML paths, stripped."""
    for path in paths:
        text = root.findtext(path)
        if text and text.strip():
            return text.strip()
    return None


def gpu(results_dir: Path) -> dict:
    """The GPU from ``nvidia-smi -q -x``, saved beside the results; {} without one."""
    if not shutil.which("nvidia-smi"):
        return {}
    xml = subprocess.run(
        ["nvidia-smi", "-q", "-x"], capture_output=True, text=True, timeout=60
    ).stdout
    (results_dir / "nvidia-smi.xml").write_text(xml)
    root = ET.fromstring(xml)
    gpus = []
    for g in root.findall("gpu"):
        gpus.append(
            {
                "name": _text(g, "product_name"),
                "uuid": _text(g, "uuid"),
                "vbios": _text(g, "vbios_version"),
                "memory_total": _text(g, "fb_memory_usage/total"),
                "ecc": _text(g, "ecc_mode/current_ecc"),
                "mig": _text(g, "mig_mode/current_mig"),
                "persistence": _text(g, "persistence_mode"),
                "power_limit": _text(
                    g,
                    "gpu_power_readings/current_power_limit",
                    "power_readings/power_limit",
                ),
                "application_clock_sm": _text(g, "applications_clocks/graphics_clock"),
                "max_clock_sm": _text(g, "max_clocks/graphics_clock"),
                "max_clock_memory": _text(g, "max_clocks/mem_clock"),
                "pcie": _text(g, "pci/pci_gpu_link_info/pcie_gen/current_link_gen"),
                "pcie_width": _text(
                    g, "pci/pci_gpu_link_info/link_widths/current_link_width"
                ),
            }
        )
    return {
        "driver": _text(root, "driver_version"),
        "cuda": _text(root, "cuda_version"),
        "gpus": gpus,
    }


def _read(path: str) -> str | None:
    """A file's stripped text, or None if it can't be read."""
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _sysctl(name: str) -> str | None:
    """A macOS sysctl value, or None elsewhere."""
    if sys.platform != "darwin":
        return None
    result = subprocess.run(
        ["sysctl", "-n", name], capture_output=True, text=True, timeout=10
    )
    return result.stdout.strip() or None


def host() -> dict:
    """The host's CPU, memory, kernel and container limits."""
    cpuinfo = _read("/proc/cpuinfo") or ""
    model = next(
        (
            line.split(":", 1)[1].strip()
            for line in cpuinfo.splitlines()
            if line.startswith("model name")
        ),
        _sysctl("machdep.cpu.brand_string") or platform.processor() or None,
    )
    meminfo = _read("/proc/meminfo") or ""
    mem_kb = next(
        (
            int(line.split()[1])
            for line in meminfo.splitlines()
            if line.startswith("MemTotal")
        ),
        None,
    )
    mem_bytes = mem_kb * 1024 if mem_kb else int(_sysctl("hw.memsize") or 0) or None
    affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
    return {
        "cpu_model": model,
        "cpu_count": os.cpu_count(),
        "cpus_available": len(affinity) if affinity is not None else None,
        "machine_model": _sysctl("hw.model"),
        "mem_total_bytes": mem_bytes,
        "platform": platform.platform(),
        "kernel": platform.release(),
        "cgroup_cpu_max": _read("/sys/fs/cgroup/cpu.max"),
        "cgroup_memory_max": _read("/sys/fs/cgroup/memory.max"),
    }


def build() -> dict:
    """The image's build record, or the working tree's commit when run outside it."""
    if BUILD_INFO.exists():
        return json.loads(BUILD_INFO.read_text())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except OSError:
        commit = None
    return {"git_sha": commit or None, "image": None}


def capture(results_dir: Path, plan: dict, plan_sha256: str) -> dict:
    """The whole manifest, apart from the run's timestamps and results.

    Args:
        results_dir: Where nvidia-smi's XML is saved.
        plan: The benchmark plan.
        plan_sha256: The plan file's SHA-256.

    Returns:
        dict: The manifest.
    """
    return {
        "build": build(),
        "image": os.environ.get("BENCH_IMAGE"),
        "software": software(),
        "gpu": gpu(results_dir),
        "host": host(),
        "environment": safe_environment(),
        "plan": plan,
        "plan_sha256": plan_sha256,
    }
