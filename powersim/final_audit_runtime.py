"""Matched, live DCOPF runtime benchmark for the final clean-suite audit."""
import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .config import SimulationConfig
from .data import write_json
from .hybrid import HybridPredictor
from .ml import load_decoder, load_model, reconstruction_errors, tensor
from .network import NodalLoadWriter, digest, network_fingerprint
from .report_inputs import Inventory, read_json, sha256
from .report_runtime import _row, environment
from .solver import solve_reference
from .workers import ordered_pool, validate_workers


def _init_worker(network_path, simulation, model_path, decoder_path, ids, x, model_id):
    import pandapower as pp
    import torch

    torch.set_num_threads(1)
    logging.getLogger("pandapower").setLevel(logging.ERROR)
    net = pp.from_json(network_path)
    model, checkpoint = load_model(model_path)
    if network_fingerprint(net) != model_id or checkpoint["model_id"] != model_id:
        raise ValueError("Runtime model and reference network must match")
    decoder, meta = load_decoder(decoder_path, model_path, model)
    gate = HybridPredictor(model, decoder, meta, ids)
    _TIMING.update(model=model, decoder=decoder, meta=meta, gate=gate, config=simulation,
                   m0=NodalLoadWriter(net), fallback=NodalLoadWriter(deepcopy(net)))
    for name in ("numerical", "neural", "decoder", "hybrid"):
        _case((name, x))  # once per operation in every worker, excluded from timing


_TIMING = {}


def _case(task):
    import torch

    method, x = task
    w = _TIMING

    def solve(writer):
        writer.apply(x)
        return solve_reference(writer.net, w["config"])

    t = perf_counter()
    with torch.no_grad():
        if method == "numerical":
            result = solve(w["m0"])
            success = result.status == "ok" and result.converged and result.metrics.get("feasible")
            calls, failure = 0, not success
        elif method == "neural":
            w["model"](tensor(x[None]))[0].numpy()
            calls = failure = 0
        elif method == "decoder":
            float(reconstruction_errors(w["model"], w["decoder"], tensor(x[None]), w["meta"]["error_units"])[0])
            calls = failure = 0
        else:
            decision = w["gate"].predict(x, lambda: solve(w["fallback"]))
            calls, failure = not decision.accepted, decision.status == "failed"
    return perf_counter()-t, int(calls), int(failure)


def benchmark_suite(root, output, *, workers=4, repeats=2, limit=2000):
    validate_workers(workers)
    if repeats < 2 or limit < 1:
        raise ValueError("Runtime requires at least two repeats and a nonempty workload")
    output.mkdir(parents=True, exist_ok=True)
    if (output/"runtime.csv").exists() or (output/"runtime_environment.json").exists():
        raise FileExistsError("Runtime outputs already exist")
    inventory = Inventory(root.parent, [], clean_suites=[root])
    seed = read_json(root/"suite_manifest.json")["plan"]["seeds"][0]
    rows = []
    for e in inventory.evaluations():
        if (e.method, e.mode, e.split) != ("M5", "hybrid-q95-rmse", "ood_test_2"):
            continue
        training = read_json(e.model.parent/"training.json")
        if training["configuration"]["seed"] != seed:
            continue
        d = inventory.dataset(e.dataset)
        net_path, model, decoder = map(inventory.checked, (d.path/"network.json", e.model, e.decoder))
        simulation = SimulationConfig(**d.config["simulation"])
        x, ids = d.arrays("ood_test_2_x", "ood_test_2_scenario_ids")
        x, ids = x[:limit], ids[:limit]
        workload = digest([d.id, e.split, ids.tolist()])
        common = dict(network=d.network, method=e.method, seed=seed, split=e.split, gate=e.mode,
                      workload_id=workload, workers=workers, threads_per_worker=1, device="CPU",
                      batch_size=1, samples=len(x), reference_valid_only=True)
        print(f"Timing {d.network}: {len(x)} matched cases, {repeats} repeats, {workers} workers", flush=True)
        with ordered_pool(workers, _init_worker, (net_path, simulation, model, decoder, d.gen_ids, x[0],
                                                d.manifest["model_id"])) as mapper:
            # Trigger process creation and transport warm-up outside measurement.
            list(mapper(_case, (("neural", row) for row in x[:max(16, workers*8)])))
            for repeat in range(1, repeats+1):
                start = perf_counter()
                (loaded,) = d.arrays("ood_test_2_x")
                workload_rows = np.ascontiguousarray(loaded[:limit], dtype=float)
                preprocess = perf_counter()-start
                if not np.array_equal(workload_rows, x):
                    raise ValueError("Timing repeat input order changed")
                records = []
                for method in (("numerical", "neural", "decoder", "hybrid") if repeat % 2
                               else ("hybrid", "decoder", "neural", "numerical")):
                    start = perf_counter()
                    timings = list(mapper(_case, ((method, row) for row in workload_rows)))
                    wall = perf_counter()-start
                    elapsed, calls, failures = map(np.asarray, zip(*timings))
                    records.append(_row({**common, "repeat": repeat}, method, len(x), wall, latencies=elapsed,
                                        preprocessing_seconds=preprocess, end_to_end_seconds=wall+preprocess,
                                        fallback_calls=int(calls.sum()), failed_outputs=int(failures.sum())))
                m0 = next(r for r in records if r["component"] == "numerical")["end_to_end_seconds"]
                for record in records:
                    record["speedup_end_to_end"] = m0/record["end_to_end_seconds"]
                rows.extend(records)
        pd.DataFrame(rows).to_csv(output/"runtime.partial.csv", index=False)
    if not rows:
        raise ValueError("No first-seed M5 Q95 timing workloads found")
    (output/"runtime.partial.csv").rename(output/"runtime.csv")
    write_json(output/"runtime_environment.json", {
        **environment(), "workers": workers, "repeats": repeats, "limit": limit,
        "implementation_sha256": sha256(Path(__file__)), "source_suite": str(root),
        "warmup": "One operation of each type per worker; pool transport primed before timing",
        "wall_scope": "Timed read/decompression and contiguous preparation plus ordered dispatch and result transfer. "
                      "Loaded model/network, process startup, offline scoring, and result writing excluded.",
        "latency_scope": "Within-worker timings are measured concurrently; report speedups use matched wall durations.",
        "fallback_scope": "Every rejected hybrid row invokes live network OPF; no cached labels or recorded timings.",
        "decoder_scope": "Feature extraction and reconstruction-score calculation; not a complete dispatch.",
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args(argv)
    benchmark_suite(args.run.resolve(), args.output, workers=args.workers, repeats=args.repeats, limit=args.limit)


if __name__ == "__main__":
    main()
