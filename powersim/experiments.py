"""Measured single-case latency, batch throughput and independent physical metrics."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .config import SimulationConfig
from .contracts import DispatchDecision
from .data import load_dataset, package_versions, write_json
from .hybrid import HybridPredictor
from .ml import artifact_hash, load_decoder, load_model, tensor
from .network import apply_nodal_loads, digest, make_case
from .solver import evaluate_decision, solve_reference


def accuracy(prediction, reference, *, denominator=None):
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if not len(reference):
        return {"mae_mw": None, "rmse_mw": None, "maximum_error_mw": None,
                "success_rate_1mw": 0.0 if denominator else None, "accuracy_samples": 0}
    error = prediction-reference
    valid = np.isfinite(error).all(axis=1)
    finite = error[valid]
    return {"mae_mw": float(np.abs(finite).mean()) if len(finite) else None,
            "rmse_mw": float(np.sqrt(np.mean(finite**2))) if len(finite) else None,
            "maximum_error_mw": float(np.abs(finite).max()) if len(finite) else None,
            "success_rate_1mw": float(np.sum(valid & (np.nan_to_num(np.abs(error), nan=np.inf).max(axis=1) <= 1)) /
                                     (denominator if denominator is not None else len(reference))),
            "accuracy_samples": int(valid.sum())}


def _physical_summary(evaluations, reference_objectives):
    feasible = [i for i, e in enumerate(evaluations) if e.status == "ok" and e.metrics["feasible"]]
    gaps = [(evaluations[i].objective-reference_objectives[i])/max(abs(reference_objectives[i]), 1e-9)*100 for i in feasible]
    balances = [abs(e.raw_balance_residual_mw) for e in evaluations if e.raw_balance_residual_mw is not None]
    return {
        "pf_failures": sum(e.status != "ok" for e in evaluations),
        "feasible_predictions": len(feasible),
        "feasible_rate": len(feasible)/len(evaluations) if evaluations else None,
        "balance_mean_absolute_mw": float(np.mean(balances)) if balances else None,
        "balance_maximum_absolute_mw": float(np.max(balances)) if balances else None,
        "generator_bound_violations": sum(e.metrics.get("generator_bound_violations", 0) for e in evaluations),
        "branch_violations": sum(e.metrics.get("branch_violations", 0) for e in evaluations),
        "maximum_generator_bound_violation_mw": max((e.metrics.get("maximum_generator_bound_violation_mw", 0) for e in evaluations), default=0),
        "maximum_branch_violation_mw": max((e.metrics.get("maximum_branch_violation_mw", 0) for e in evaluations), default=0),
        "voltage_violations": None, "reactive_violations": None,
        "objective_gap_mean_percent_feasible_only": float(np.mean(gaps)) if gaps else None,
        "objective_gap_maximum_percent_feasible_only": float(np.max(gaps)) if gaps else None,
        "objective_gap_samples": len(gaps),
    }


def evaluate_model(dataset, model_path, output, *, split="iid_test", decoder_path=None):
    manifest, a = load_dataset(dataset)
    if split not in manifest["splits"] or split in ("train", "validation"):
        raise ValueError("Evaluation requires a held-out IID/OOD test split")
    model, checkpoint = load_model(model_path)
    if (checkpoint["model_id"] != manifest["model_id"] or checkpoint["bus_ids"] != manifest["bus_ids"]
            or checkpoint["generator_ids"] != manifest["generator_ids"]):
        raise ValueError("Model structure/ordering mismatch")
    if set(a[f"{split}_scenario_ids"]) & set(checkpoint["training_scenario_ids"] + checkpoint["validation_scenario_ids"]):
        raise ValueError("Held-out split overlaps training/validation")
    if {digest(row.tolist()) for row in a[f"{split}_x"]} & set(checkpoint["training_input_ids"] + checkpoint["validation_input_ids"]):
        raise ValueError("Held-out nodal inputs overlap training/validation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = SimulationConfig(**manifest["configuration"]["simulation"])
    if config.physics_mode != "dc":
        raise ValueError("DC-trained models must not be evaluated as AC methods")
    ids = manifest["generator_ids"]
    x, y = a[f"{split}_x"], a[f"{split}_y"]
    if not len(x):
        raise ValueError("No valid reference cases in requested split")
    objectives = a[f"{split}_objective"]
    predictor = None
    if decoder_path:
        decoder, meta = load_decoder(decoder_path, model_path, model)
        predictor = HybridPredictor(model, decoder, meta, ids)
    # Warm-up is explicit, excluded from measured inference latency/throughput.
    with torch.no_grad():
        model(tensor(x[:1]))
        if predictor:
            from .ml import reconstruction_errors
            reconstruction_errors(model, decoder, tensor(x[:1]), meta["error_units"])
        started = time.perf_counter()
        batch_prediction = model(tensor(x)).numpy()
        batch_seconds = time.perf_counter()-started
    raw, hybrid, accepted, runtimes, fallback_times, statuses = [], [], [], [], [], []
    raw_evaluations, hybrid_evaluations, records = [], [], []
    eval_started = time.perf_counter()
    for i, loads in enumerate(x):
        def fallback():
            net = make_case(config)
            apply_nodal_loads(net, loads)
            return solve_reference(net, config)
        if predictor:
            result = predictor.predict(loads, fallback)  # NEVER log held-out production samples
            prediction = np.asarray(result.raw_prediction_mw) if result.raw_prediction_mw is not None else np.full(len(ids), np.nan)
            chosen = np.asarray(result.generation_mw) if result.generation_mw is not None else np.full(len(ids), np.nan)
            accepted.append(result.accepted); runtimes.append(result.elapsed_seconds)
            fallback_times.append(result.fallback_seconds); statuses.append(result.status)
            record = asdict(result)
        else:
            started = time.perf_counter()
            with torch.no_grad():
                prediction = model(tensor(loads[None]))[0].numpy()
            runtimes.append(time.perf_counter()-started)
            chosen = prediction
            accepted.append(True); fallback_times.append(0); statuses.append("learned")
            record = {}
        raw.append(prediction); hybrid.append(chosen)
        # Physical audit time is separate from decision latency and is measured.
        net = make_case(config); apply_nodal_loads(net, loads)
        e = evaluate_decision(net, DispatchDecision(active_power_mw=dict(zip(ids, map(float, prediction)))), config)
        raw_evaluations.append(e)
        if np.array_equal(prediction, chosen):
            he = e
        else:
            net = make_case(config); apply_nodal_loads(net, loads)
            he = evaluate_decision(net, DispatchDecision(active_power_mw=dict(zip(ids, map(float, chosen)))), config)
        hybrid_evaluations.append(he)
        records.append({"scenario_id": str(a[f"{split}_scenario_ids"][i]), **record,
                        "raw_evaluation": e.to_dict(), "hybrid_evaluation": he.to_dict()})
    raw, hybrid, accepted = np.asarray(raw), np.asarray(hybrid), np.asarray(accepted)
    total_requested = manifest["splits"][split]["requested"]
    report = {
        "split": split, "labeled_cases": len(x), "requested_cases": total_requested,
        "reference_failures_excluded_from_error_metrics": total_requested-len(x),
        "success_definition": "Fraction of reference-valid scenarios with COMPLETE output and max absolute source error <= 1 MW; failed hybrid outputs count as unsuccessful.",
        "raw_accuracy": accuracy(raw, y), "hybrid_accuracy": accuracy(hybrid, y),
        "success_rate_all_requested_1mw": accuracy(hybrid, y, denominator=total_requested)["success_rate_1mw"],
        "accepted_accuracy": accuracy(raw[accepted], y[accepted]),
        "raw_physics": _physical_summary(raw_evaluations, objectives),
        "hybrid_physics": _physical_summary(hybrid_evaluations, objectives),
        "decoder_acceptance_rate": float(accepted.mean()) if predictor else None,
        "fallback_rate": float((~accepted).mean()) if predictor else 0.0,
        "fallback_failures": statuses.count("failed"),
        "timing": {
            "device": "CPU", "warmup_cases": 1, "single_scenario_samples": len(x),
            "single_scenario_mean_seconds": float(np.mean(runtimes)),
            "single_scenario_p95_seconds": float(np.quantile(runtimes, .95)),
            "decision_seconds_total_including_decoder_and_fallback": float(sum(runtimes)),
            "fallback_seconds_total": float(sum(fallback_times)),
            "opf_model_only_batch_seconds": batch_seconds,
            "opf_model_only_batch_cases_per_second": len(x)/batch_seconds,
            "evaluation_loop_seconds_including_physical_audits": time.perf_counter()-eval_started,
            "note": "Decision latency includes normalization, decoder and fallback construction/OPF when used; excludes independent PF audit. Batch throughput excludes decoder/fallback.",
        },
        "model_hash": artifact_hash(model_path), "versions": package_versions(),
        "warning": "Decoder acceptance is empirical selection, not a feasibility or optimality certificate.",
    }
    write_json(output / "evaluation.json", report)
    # Preserve failed outputs as JSON nulls, not non-standard NaN literals.
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    with (output / "predictions.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(clean(record), allow_nan=False) + "\n")
    np.savez_compressed(output / "predictions.npz", raw_generation_mw=raw, hybrid_generation_mw=hybrid,
                        batch_generation_mw=batch_prediction, accepted=accepted)
    return report
