"""One benchmark measurement, run in its own process by run.py.

Usage: python -m benchmarks.gpu.measure '<spec as JSON>'

Prints the measurement as one line, ``RESULT <json>``. A fresh process per
measurement means its peak memory starts from zero, and running out of memory ends
only that measurement. Data are drawn on the device and excluded from the timings;
compilation is timed separately from the steady state.
"""

import contextlib
import io
import json
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import checkify

from benchmarks.gpu.common import RESULT_PREFIX, to_json, utc_now
from benchmarks.gpu.data import quantiles
from metalog_jax.base import (
    MetalogBoundedness,
    MetalogFitMethod,
    MetalogInputData,
    MetalogParameters,
    MetalogRandomVariableParameters,
)
from metalog_jax.grid_search import fit_grid
from metalog_jax.metalog import fit
from metalog_jax.utils import (
    DEFAULT_Y,
    HDRPRNGParameters,
    JaxUniformDistributionParameters,
)

# Uniforms for sampling stay this far inside (0, 1), where ppf is finite
_UNIFORM_MARGIN = 1e-12


class CompileClock:
    """Seconds JAX spends tracing and compiling, summed from jax.monitoring events."""

    def __init__(self):
        """Start listening for compilation events."""
        self.seconds = 0.0
        jax.monitoring.register_event_duration_secs_listener(self._on_event)

    def _on_event(self, event: str, duration: float, **_kwargs) -> None:
        """Add a compilation event's duration."""
        if event.startswith("/jax/core/compile/"):
            self.seconds += duration


def host_peak_bytes() -> int:
    """This process's peak resident memory, in bytes on every platform."""
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # bytes on macOS


def device_memory() -> dict:
    """Peak and limit of the default device's memory; None on the CPU."""
    stats = jax.devices()[0].memory_stats() or {}
    return {
        "peak_gpu_bytes": stats.get("peak_bytes_in_use"),
        "bytes_limit": stats.get("bytes_limit"),
    }


def parameters(plan: dict, method: str, num_terms: int) -> MetalogParameters:
    """The benchmark's MetalogParameters for a fit method and term count."""
    return MetalogParameters(
        boundedness=MetalogBoundedness[plan["data"]["boundedness"]],
        lower_bound=plan["data"]["lower_bound"],
        upper_bound=0.0,
        method=MetalogFitMethod.Feasible
        if method == "feasible"
        else MetalogFitMethod.OLS,
        num_terms=num_terms,
    )


def timed(run, repetitions: int, budget_s: float, clock: CompileClock) -> tuple:
    """Time a first call, then up to ``repetitions`` steady-state calls.

    Steady-state calls stop early once another would pass ``budget_s``, but at least
    one runs. The steady time is the fastest, with the compilation it included.

    Args:
        run: A function that runs the work and blocks until it's done.
        repetitions: The most steady-state calls.
        budget_s: Seconds for the steady-state calls.
        clock: The process's compilation clock.

    Returns:
        tuple: The first call's result, and the timings.
    """
    compiled, start = clock.seconds, time.perf_counter()
    result = run()
    first_s, first_compile_s = time.perf_counter() - start, clock.seconds - compiled
    steady, steady_compile = [], []
    started = time.perf_counter()
    while len(steady) < repetitions:
        if steady and time.perf_counter() - started + min(steady) > budget_s:
            break
        compiled, start = clock.seconds, time.perf_counter()
        run()
        steady.append(time.perf_counter() - start)
        steady_compile.append(clock.seconds - compiled)
    best = int(np.argmin(steady))
    return result, {
        "first_s": first_s,
        "first_compile_s": first_compile_s,
        "steady_s": steady[best],
        "steady_compile_s": steady_compile[best],
        "repetitions": len(steady),
    }


def iteration_stats(iterations) -> dict:
    """Mean, maximum and histogram of a Feasible grid's exchange iterations."""
    if iterations is None:
        return {}
    it = np.asarray(iterations).reshape(-1)
    return {
        "iterations_mean": float(it.mean()),
        "iterations_max": int(it.max()),
        "iterations_histogram": np.bincount(it).tolist(),
    }


def draw(plan: dict, start: int, count: int) -> tuple[jax.Array, jax.Array, float]:
    """Draw datasets on the device; return them, their levels and the seconds taken."""
    began = time.perf_counter()
    x = quantiles(plan["seed"], start, count, plan["data"]["samples_per_dataset"])
    y = jnp.broadcast_to(DEFAULT_Y, x.shape)
    jax.block_until_ready((x, y))
    return x, y, time.perf_counter() - began


def measure_check(spec: dict, plan: dict, clock: CompileClock) -> dict:
    """Check the backend, the device kind and float64; record JAX's environment."""
    del plan, clock
    info = io.StringIO()
    with contextlib.redirect_stdout(info):
        jax.print_environment_info()
    device = jax.devices()[0]
    problems = []
    if jax.default_backend() != spec["backend"]:
        problems.append(
            f"the backend is {jax.default_backend()}, not {spec['backend']}"
        )
    if spec.get("device_kind") and device.device_kind != spec["device_kind"]:
        problems.append(
            f"the device is {device.device_kind}, not {spec['device_kind']}"
        )
    if not jax.config.jax_enable_x64:
        problems.append("float64 is off")
    return {
        "status": "failed" if problems else "ok",
        "reason": "; ".join(problems) or None,
        "backend": jax.default_backend(),
        "devices": [f"{d.platform}:{d.id} {d.device_kind}" for d in jax.devices()],
        "device_kind": device.device_kind,
        "x64": bool(jax.config.jax_enable_x64),
        "jax_environment_info": info.getvalue(),
    }


def measure_correctness(spec: dict, plan: dict, clock: CompileClock) -> dict:
    """Fit the same datasets on the default device and on the CPU, and compare."""
    del clock
    x, y, _ = draw(plan, 0, spec["N"])
    cpu = jax.devices("cpu")[0]
    methods = {}
    for method in spec["methods"]:
        params = parameters(plan, method, spec["K"])
        device_fit = fit_grid(x, y, params)
        with jax.default_device(cpu):
            cpu_fit = fit_grid(jax.device_put(x, cpu), jax.device_put(y, cpu), params)
        a, b = np.asarray(device_fit.metalog.a), np.asarray(cpu_fit.metalog.a)
        scale = np.max(np.abs(b), axis=1)
        methods[method] = {
            "max_relative_coefficient_difference": float(
                np.max(np.max(np.abs(a - b), axis=1) / scale)
            ),
            "max_ks_difference": float(
                np.max(
                    np.abs(np.asarray(device_fit.ks_dist) - np.asarray(cpu_fit.ks_dist))
                )
            ),
            "feasible_masks_agree": bool(
                np.array_equal(device_fit.feasible, cpu_fit.feasible)
            ),
        }
    return {"N": spec["N"], "K": spec["K"], "methods": methods}


def measure_fit(spec: dict, plan: dict, clock: CompileClock) -> dict:
    """Time one fit_grid call on a single batch, optionally over several term counts."""
    n, k, num_terms = spec["N"], spec["K"], spec.get("num_terms")
    x, y, generate_s = draw(plan, 0, n)
    params = parameters(plan, spec["method"], k)

    def run():
        return jax.block_until_ready(fit_grid(x, y, params, num_terms=num_terms))

    result, timing = timed(run, plan["repetitions"], plan["repetition_budget_s"], clock)
    if num_terms:
        mode = f"{spec['method']}_terms"
    else:
        mode = "feasible" if spec["method"] == "feasible" else "ols_grid"
    return {
        "mode": mode,
        "N": n,
        "K": k,
        "num_terms": num_terms,
        **timing,
        "per_dataset_ms": 1e3 * timing["steady_s"] / n,
        "generate_s": generate_s,
        "feasible_fraction": float(np.mean(np.asarray(result.feasible))),
        **iteration_stats(result.iterations),
    }


def measure_chunked(spec: dict, plan: dict, clock: CompileClock) -> dict:
    """Fit up to N datasets in chunks of one size, as bench_chunked.py does.

    Stops starting chunks once another wouldn't finish within ``spec["time_s"]``, if
    given, and reports the chunks it fitted, so that running out of time doesn't lose
    the measurement.
    """
    total, chunk, k = spec["N"], spec["chunk"], spec["K"]
    params = parameters(plan, spec["method"], k)
    budget = spec.get("time_s", float("inf"))
    fit_s = compile_s = generate_s = slowest = 0.0
    feasible, iterations = [], []
    began = time.perf_counter()
    for i in range(total // chunk):
        if feasible and time.perf_counter() - began + slowest > budget:
            break
        x, y, seconds = draw(plan, i * chunk, chunk)
        generate_s += seconds
        compiled, start = clock.seconds, time.perf_counter()
        result = jax.block_until_ready(fit_grid(x, y, params))
        elapsed = time.perf_counter() - start
        fit_s, slowest = fit_s + elapsed, max(slowest, elapsed)
        compile_s += clock.seconds - compiled
        feasible.append(np.asarray(result.feasible))
        if result.iterations is not None:
            iterations.append(np.asarray(result.iterations))
    fitted = len(feasible) * chunk
    excluding = fit_s - compile_s
    return {
        "mode": f"{'feasible' if spec['method'] == 'feasible' else 'ols'}_chunked",
        "N": fitted,
        "planned_N": total,
        "chunk": chunk,
        "K": k,
        "fit_s": fit_s,
        "compile_s": compile_s,
        "fit_s_excl_compile": excluding,
        "per_dataset_ms": 1e3 * excluding / fitted,
        "generate_s": generate_s,
        "feasible_fraction": float(np.mean(np.concatenate(feasible))),
        **iteration_stats(np.concatenate(iterations) if iterations else None),
    }


def _model(plan: dict):
    """A 10-term Feasible fit of the first dataset, to sample from."""
    x, _, _ = draw(plan, 0, 1)
    data = MetalogInputData.from_values(x[0], DEFAULT_Y, precomputed_quantiles=True)
    return fit(data, parameters(plan, "feasible", plan["num_terms"]))


def measure_sampling(spec: dict, plan: dict, clock: CompileClock) -> dict:
    """Time drawing or evaluating N points of one fitted metalog."""
    kind, n = spec["sampling"], spec["N"]
    model = _model(plan)
    u = jax.random.uniform(
        jax.random.key(plan["seed"]),
        (n,),
        dtype=jnp.float64,
        minval=_UNIFORM_MARGIN,
        maxval=1 - _UNIFORM_MARGIN,
    )
    if kind in ("rvs_jax", "rvs_hdr"):
        prng = (
            JaxUniformDistributionParameters(seed=plan["seed"])
            if kind == "rvs_jax"
            else HDRPRNGParameters()
        )
        rv = MetalogRandomVariableParameters(prng_params=prng, size=n, max_draws=n)

        def run():
            return jax.block_until_ready(model.rvs(rv))

    else:
        functions = {
            "ppf_jit": lambda m, v: m.ppf(v),
            "cdf": lambda m, v: m.cdf(v),
            "pdf_cdf": lambda m, v: m.pdf(m.cdf(v)),
        }
        jitted = jax.jit(checkify.checkify(functions[kind]))
        arg = u if kind == "ppf_jit" else jax.block_until_ready(model.ppf(u))

        def run():
            err, value = jitted(model, arg)
            err.throw()
            return jax.block_until_ready(value)

    _, timing = timed(run, spec.get("repetitions", 3), spec.get("budget_s", 30), clock)
    return {
        "mode": kind,
        "N": n,
        **timing,
        "points_per_s": n / timing["steady_s"],
        "ns_per_point": 1e9 * timing["steady_s"] / n,
    }


MEASUREMENTS = {
    "check": measure_check,
    "correctness": measure_correctness,
    "fit": measure_fit,
    "chunked": measure_chunked,
    "sampling": measure_sampling,
}


def main(argv: list[str]) -> int:
    """Run the measurement in argv[1] and print its record."""
    spec = json.loads(argv[1])
    plan = json.loads(open(spec["plan"]).read())
    clock = CompileClock()
    started, began = utc_now(), time.perf_counter()
    record = MEASUREMENTS[spec["kind"]](spec, plan, clock)
    record = {
        "status": "ok",
        **record,
        "kind": spec["kind"],
        "phase": spec.get("phase"),
        "started": started,
        "ended": utc_now(),
        "wall_s": time.perf_counter() - began,
        "compile_s_total": clock.seconds,
        "host_peak_bytes": host_peak_bytes(),
        **device_memory(),
    }
    print(RESULT_PREFIX + to_json(record), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
