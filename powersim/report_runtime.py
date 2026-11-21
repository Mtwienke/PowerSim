"""Controlled CPU timing; no timings or solver calls in the default report."""
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
import os
import platform
import subprocess
from time import perf_counter

import numpy as np

from .data import package_versions
from .network import digest
from .report_metrics import accuracy_summary, speedup


def environment():
    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                             text=True, check=False).stdout.strip() or cpu
    versions = package_versions()
    try:
        versions["matplotlib"] = version("matplotlib")
    except PackageNotFoundError:
        versions["matplotlib"] = None
    return {
        "platform": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count(),
        "device": "CPU", "gpu_used": False, "gpu_synchronization": "not applicable (CPU)",
        "versions": versions,
        "thread_environment": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")},
    }


def elapsed_rows(function, x):
    values, times = [], []
    start = perf_counter()
    for row in x:
        t = perf_counter()
        values.append(function(row))
        times.append(perf_counter()-t)
    return perf_counter()-start, values, np.asarray(times)


def _row(common, component, n, wall, *, latencies=None, batch_size=1, **details):
    return {
        **common, "component": component, "samples": n, "total_seconds": wall, "batch_size": batch_size,
        "mean_seconds_per_sample": wall/n,
        "mean_latency_seconds": float(latencies.mean()) if latencies is not None else None,
        "median_latency_seconds": float(np.median(latencies)) if latencies is not None else None,
        "p95_latency_seconds": float(np.quantile(latencies, .95)) if latencies is not None else None,
        "samples_per_second": n/wall, **details,
    }


def benchmark(entry, dataset, labels, physics, *, repeats=3, limit=2000):
    """Time actual solver, NN, decoder and complete hybrid on identical rows."""
    import torch

    from .config import SimulationConfig
    from .hybrid import HybridPredictor
    from .ml import RECONSTRUCTION_SCORES, load_decoder, load_model, reconstruction_errors, tensor
    from .network import NodalLoadWriter
    from .solver import solve_reference

    torch.set_num_threads(1)
    x, ids = labels.x[:limit], labels.ids[:limit]
    if not len(x) or entry.decoder is None:
        raise ValueError("Controlled hybrid timing needs nonempty labels and a matching decoder")
    workload = digest([dataset.id, entry.split, ids.tolist()])
    model, _ = load_model(entry.model)
    decoder, meta = load_decoder(entry.decoder, entry.model, model)
    gate = HybridPredictor(model, decoder, meta, dataset.gen_ids)
    config = SimulationConfig(**dataset.config["simulation"])

    def solver(writer, row):
        writer.apply(row)
        return solve_reference(writer.net, config)

    def raw(row):
        return model(tensor(row[None]))[0].numpy()

    def score(row):
        return float(reconstruction_errors(model, decoder, tensor(row[None]), meta["error_units"])[0])

    result = []
    batch_size, warmup = 256, min(5, len(x))
    with torch.no_grad():
        for i in range(repeats):
            common = {
                "measurement": "controlled_fresh_cpu", "run": entry.run, "profile": entry.profile,
                "network": dataset.network, "method": entry.method, "scenario": entry.split,
                "configuration": entry.mode, "workload_id": workload, "repeat": i+1,
                "device": "CPU", "workers": 1, "torch_threads": 1, "warmup_cases": warmup,
                "threshold": meta["threshold"], "score_type": meta["error_units"],
                "score_units": RECONSTRUCTION_SCORES[meta["error_units"]][0],
                "gpu_synchronization": "not applicable (CPU)",
            }
            numerical_writer = NodalLoadWriter(deepcopy(dataset.net))
            hybrid_writer = NodalLoadWriter(deepcopy(dataset.net))

            def hybrid(row):
                return gate.predict(row, lambda: solver(hybrid_writer, row))

            model(tensor(x[:batch_size]))
            for row in x[:warmup]:
                solver(numerical_writer, row)
                # Exercise the fallback solver even when all warm-up rows are accepted.
                solver(hybrid_writer, row)
                raw(row), score(row), hybrid(row)
            started = perf_counter()
            (loaded,) = dataset.arrays(f"{entry.split}_x")
            loads = np.ascontiguousarray(loaded[:limit], dtype=np.float64)
            load_seconds = perf_counter()-started
            if not np.array_equal(loads, x):
                raise ValueError("Timed data loading did not produce the benchmarked inputs")
            numerical_seconds, numeric, opf_latencies = elapsed_rows(lambda row: solver(numerical_writer, row), loads)
            failures = sum(not (r.status == "ok" and r.converged and r.metrics.get("feasible")) for r in numeric)
            raw_seconds, _, raw_latencies = elapsed_rows(raw, loads)
            score_seconds, _, score_latencies = elapsed_rows(score, loads)
            hybrid_seconds, decisions, hybrid_latencies = elapsed_rows(hybrid, loads)
            started = perf_counter()
            for offset in range(0, len(loads), batch_size):
                model(tensor(loads[offset:offset+batch_size])).numpy()
            batch_seconds = perf_counter()-started
            fresh = np.array([r.generation_mw if r.generation_mw is not None else [np.nan]*labels.y.shape[1]
                              for r in decisions])
            fresh_accuracy = accuracy_summary(fresh, labels.y[:len(loads)], requested=len(loads))
            feasible = physics.evaluate(fresh, loads, labels.cost[:len(loads)])["feasible"].mean()
            calls = sum(not r.accepted for r in decisions)
            fallback_failures = sum(r.status == "failed" for r in decisions)
            fallback_seconds = sum(r.fallback_seconds for r in decisions)
            for name, seconds, latencies, batch, extra in (
                ("data_load_preprocess", load_seconds, None, None, {}),
                ("numerical_opf", numerical_seconds, opf_latencies, 1, {"failures": failures}),
                ("neural_single", raw_seconds, raw_latencies, 1, {}),
                ("decoder_score_with_features", score_seconds, score_latencies, 1, {}),
                ("neural_batch", batch_seconds, None, batch_size, {}),
                ("hybrid", hybrid_seconds, hybrid_latencies, 1, {}),
                ("numerical_including_load", numerical_seconds+load_seconds, None, 1, {"failures": failures}),
                ("neural_including_load", raw_seconds+load_seconds, None, 1, {}),
                ("hybrid_including_load", hybrid_seconds+load_seconds, None, 1, {}),
            ):
                if name.startswith("hybrid"):
                    extra.update(fallback_calls=calls, fallback_failures=fallback_failures,
                                 fallback_seconds=fallback_seconds,
                                 element_lt_1mw=fresh_accuracy["element_lt_1mw"],
                                 max_lt_1mw=fresh_accuracy["max_lt_1mw"], feasible_rate=float(feasible))
                if name in ("neural_single", "neural_batch", "hybrid", "neural_including_load", "hybrid_including_load"):
                    numerator = numerical_seconds + (load_seconds if name.endswith("_including_load") else 0)
                    extra["speedup_vs_numerical"] = speedup(numerator, seconds, workload, workload)
                result.append(_row(common, name, len(x), seconds, latencies=latencies, batch_size=batch, **extra))
    return result
