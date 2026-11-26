"""Measured single-case latency, batch throughput and independent physical metrics."""
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..core.config import SimulationConfig
from ..core.contracts import DispatchDecision
from ..core.power_flow import evaluate_decision
from ..core.solve import solve_reference
from ..core.workers import ordered_pool, validate_workers
from ..instance_generation.data import load_dataset, package_versions, write_json
from ..instance_generation.network import NodalLoadWriter, build_dispatch_inputs, digest, make_case
from ..output_generation.metrics import ACCURACY_DEFINITIONS, accuracy, selection_curve
from ..solvers.gao_double_neural.hybrid import HybridPredictor
from ..solvers.gao_double_neural.ml import (artifact_hash, calibrate_threshold, load_decoder, load_model,
                                           reconstruction_errors, tensor)


def _physical_summary(evaluations, reference_objectives):
    feasible = [i for i, e in enumerate(evaluations) if e.status == "ok" and e.metrics["feasible"]]
    gaps = [(evaluations[i].objective-reference_objectives[i])/max(abs(reference_objectives[i]), 1e-9)*100 for i in feasible]
    balances = [abs(e.raw_balance_residual_mw) for e in evaluations if e.raw_balance_residual_mw is not None]
    violated_branches = [b for e in evaluations for b in e.branches if b.violated]
    branch_excesses = [abs(b.p_from_mw)-b.limit_mw for b in violated_branches if b.limit_mw is not None]
    source_events = [b for e in evaluations for b in e.branches
                     if b.limit_mw is not None and abs(b.p_from_mw) > b.limit_mw+1.]
    return {
        "pf_failures": sum(e.status != "ok" for e in evaluations),
        "feasible_predictions": len(feasible),
        "feasible_rate": len(feasible)/len(evaluations) if evaluations else None,
        "balance_mean_absolute_mw": float(np.mean(balances)) if balances else None,
        "balance_maximum_absolute_mw": float(np.max(balances)) if balances else None,
        "generator_bound_violations": sum(e.metrics.get("generator_bound_violations", 0) for e in evaluations),
        "branch_violations": sum(e.metrics.get("branch_violations", 0) for e in evaluations),
        "scenarios_with_generator_bound_violations": sum(e.metrics.get("generator_bound_violations", 0) > 0 for e in evaluations),
        "scenarios_with_branch_violations": sum(e.metrics.get("branch_violations", 0) > 0 for e in evaluations),
        "distinct_violated_branch_ids": sorted({b.asset_id for b in violated_branches}),
        "mean_branch_violation_mw": float(np.mean(branch_excesses)) if branch_excesses else None,
        "maximum_generator_bound_violation_mw": max((e.metrics.get("maximum_generator_bound_violation_mw", 0) for e in evaluations), default=0),
        "maximum_branch_violation_mw": max((e.metrics.get("maximum_branch_violation_mw", 0) for e in evaluations), default=0),
        "source_like_branch_events_beyond_limit_plus_1mw": len(source_events),
        "source_like_mean_excess_beyond_limit_plus_1mw": (
            float(np.mean([abs(b.p_from_mw)-b.limit_mw-1 for b in source_events])) if source_events else None),
        "source_like_branch_metric_note": "Branch-scenario events after PF slack rebalance above configured RATE_A + 1 MW; not strict raw feasibility.",
        "voltage_violations": None, "reactive_violations": None,
        "objective_gap_mean_percent_feasible_only": float(np.mean(gaps)) if gaps else None,
        "objective_gap_maximum_percent_feasible_only": float(np.max(gaps)) if gaps else None,
        "objective_gap_samples": len(gaps),
    }


def _init_evaluation_worker(config, model_path, decoder_path, ids, warmup, model_id):
    global _evaluation_worker
    torch.set_num_threads(1)
    model, _ = load_model(model_path)
    predictor, fallback = None, None
    audit = NodalLoadWriter(make_case(config))
    if build_dispatch_inputs(audit.net, config)["context"].network_fingerprint != model_id:
        raise ValueError("Current benchmark does not match dataset network")
    with torch.no_grad():
        model(tensor(warmup[None]))
        if decoder_path:
            decoder, meta = load_decoder(decoder_path, model_path, model)
            predictor = HybridPredictor(model, decoder, meta, ids)
            reconstruction_errors(model, decoder, tensor(warmup[None]), meta["error_units"])
            fallback = NodalLoadWriter(make_case(config))
    _evaluation_worker = config, model, predictor, audit, fallback, ids


def _evaluate_case(loads):
    config, model, predictor, audit, fallback, ids = _evaluation_worker

    def reference():
        fallback.apply(loads)
        return solve_reference(fallback.net, config)

    if predictor:
        result = predictor.predict(loads, reference)  # never log held-out samples
        prediction = result.raw_prediction_mw
        chosen = result.generation_mw
        record = asdict(result)
        elapsed, raw_seconds = result.elapsed_seconds, result.raw_inference_seconds
    else:
        started = time.perf_counter()
        with torch.no_grad():
            prediction = tuple(float(p) for p in model(tensor(loads[None]))[0].numpy())
        elapsed = raw_seconds = time.perf_counter()-started
        chosen = prediction
        record = {"status": "learned", "accepted": True, "fallback_seconds": 0}

    def audit_output(power):
        audit.apply(loads)
        return evaluate_decision(audit.net, DispatchDecision(active_power_mw=dict(zip(ids, power))), config)

    unavailable = (float("nan"),) * len(ids)
    raw = prediction if prediction is not None else unavailable
    selected = chosen if chosen is not None else unavailable
    raw_evaluation = audit_output(raw)
    chosen_evaluation = raw_evaluation if selected == raw else audit_output(selected)
    return raw, selected, elapsed, raw_seconds, record, raw_evaluation, chosen_evaluation


def evaluate_model(dataset, model_path, output, *, split="iid_test", decoder_path=None, workers=1,
                   prediction_format="jsonl", decoder_norm_sensitivity=False):
    validate_workers(workers)
    if prediction_format not in ("jsonl", "jsonl.gz"):
        raise ValueError("prediction_format must be jsonl or jsonl.gz")
    manifest, a = load_dataset(dataset, sensitivities=False)
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
    decoder_metadata = None
    if decoder_path:
        _, decoder_metadata = load_decoder(decoder_path, model_path, model)
        if decoder_norm_sensitivity and ("validation_x" not in a or not len(a["validation_x"])):
            raise ValueError("Decoder norm diagnostics require source validation inputs")
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
    # Model-only batched benchmark is measured separately, without worker contention.
    with torch.no_grad():
        model(tensor(x[:1]))
        started = time.perf_counter()
        batch_prediction = model(tensor(x)).numpy()
        batch_seconds = time.perf_counter()-started
    raw, hybrid, accepted, runtimes, raw_times, fallback_times, statuses = [], [], [], [], [], [], []
    raw_evaluations, hybrid_evaluations, reconstruction_scores = [], [], []
    eval_started = time.perf_counter()
    initargs = config, model_path, decoder_path, ids, x[0], manifest["model_id"]
    prediction_path = output / f"predictions.{prediction_format}"
    prediction_stream = (gzip.open(prediction_path, "wt", encoding="utf-8", compresslevel=3)
                         if prediction_format == "jsonl.gz" else prediction_path.open("w"))
    with prediction_stream as out, ordered_pool(workers, _init_evaluation_worker, initargs) as map_cases:
        for sid, case in zip(a[f"{split}_scenario_ids"], map_cases(_evaluate_case, x)):
            prediction, chosen, elapsed, raw_seconds, record, audit, chosen_audit = case
            raw.append(prediction); hybrid.append(chosen)
            accepted.append(record["accepted"]); statuses.append(record["status"])
            runtimes.append(elapsed); fallback_times.append(record["fallback_seconds"])
            if raw_seconds is not None:
                raw_times.append(raw_seconds)
            raw_evaluations.append(audit); hybrid_evaluations.append(chosen_audit)
            score = record.get("reconstruction_error")
            reconstruction_scores.append(float(score) if score is not None else float("nan"))
            detail = {"scenario_id": str(sid), **record, "raw_evaluation": audit.to_dict(),
                      "hybrid_evaluation": chosen_audit.to_dict()}
            out.write(json.dumps(detail, allow_nan=False) + "\n")
    elapsed = time.perf_counter()-eval_started
    raw, hybrid, accepted = np.asarray(raw), np.asarray(hybrid), np.asarray(accepted)
    total_requested = manifest["splits"][split]["requested"]
    raw_all_requested = accuracy(raw, y, denominator=total_requested)
    hybrid_all_requested = accuracy(hybrid, y, denominator=total_requested)
    report = {
        "schema_version": "3.0",
        "split": split, "labeled_cases": len(x), "requested_cases": total_requested,
        "reference_failures_excluded_from_error_metrics": total_requested-len(x),
        "success_definition": "Fraction of reference-valid scenarios with COMPLETE output and max absolute source error <= 1 MW; failed hybrid outputs count as unsuccessful.",
        "accuracy_definitions": ACCURACY_DEFINITIONS,
        "raw_accuracy": accuracy(raw, y), "hybrid_accuracy": accuracy(hybrid, y),
        "raw_all_requested_accuracy": raw_all_requested,
        "hybrid_all_requested_accuracy": hybrid_all_requested,
        "success_rate_all_requested_1mw": hybrid_all_requested["success_rate_1mw"],
        "accepted_accuracy": accuracy(raw[accepted], y[accepted]),
        "fallback_accuracy": accuracy(hybrid[~accepted], y[~accepted]),
        "accepted_physics": _physical_summary([e for e, keep in zip(raw_evaluations, accepted) if keep], objectives[accepted]),
        "raw_physics": _physical_summary(raw_evaluations, objectives),
        "hybrid_physics": _physical_summary(hybrid_evaluations, objectives),
        "decoder_acceptance_rate": float(accepted.mean()) if decoder_path else None,
        "fallback_rate": float((~accepted).mean()) if decoder_path else 0.0,
        "fallback_failures": statuses.count("failed"),
        "coverage": {
            "reference_valid_scenarios": len(x), "original_requested_scenarios": total_requested,
            "reference_label_coverage": len(x)/total_requested,
            "accepted_scenarios": int(accepted.sum()), "fallback_scenarios": int((~accepted).sum()),
            "neural_coverage": float(accepted.mean()),
            "note": "Accepted accuracy is conditional on these scenarios, not completed-hybrid accuracy.",
        },
        "timing": {
            "device": "CPU", "workers": workers, "warmup_cases_per_worker": 1,
            "single_scenario_samples": len(x) if workers == 1 else None,
            "single_scenario_mean_seconds": float(np.mean(runtimes)) if workers == 1 else None,
            "single_scenario_p95_seconds": float(np.quantile(runtimes, .95)) if workers == 1 else None,
            "in_worker_mean_decision_seconds": float(np.mean(runtimes)),
            "in_worker_mean_raw_inference_seconds": float(np.mean(raw_times)) if raw_times else None,
            "decision_seconds_total_including_decoder_and_fallback": float(sum(runtimes)),
            "fallback_seconds_total": float(sum(fallback_times)),
            "opf_model_only_batch_seconds": batch_seconds,
            "opf_model_only_batch_cases_per_second": len(x)/batch_seconds,
            "evaluation_loop_seconds_including_physical_audits": elapsed,
            "evaluation_cases_per_second_including_physical_audits": len(x)/elapsed,
            "note": "Decision timing includes normalization, decoder and fallback OPF when used, with a prepared network; excludes PF audit and queue time. Concurrent worker durations are not isolated latency; wall throughput includes startup, scheduling, PF and JSONL.",
        },
        "model_hash": artifact_hash(model_path), "versions": package_versions(),
        "prediction_file": prediction_path.name,
        "warning": "Decoder acceptance is empirical selection, not a feasibility or optimality certificate.",
    }
    if decoder_metadata is not None:
        thresholds = [*decoder_metadata.get("validation_thresholds", []), decoder_metadata["threshold"]]
        report["decoder"] = {key: decoder_metadata.get(key) for key in (
            "threshold", "error_units", "score_units", "aggregation", "calibration_method", "quantile")}
        report["selection_curve"] = selection_curve(reconstruction_scores, raw, y, thresholds)
        scores = np.asarray(reconstruction_scores)
        for row in report["selection_curve"]:
            mask = (np.isfinite(scores) & (scores <= row["threshold"]) if row["threshold"] is not None
                    else np.zeros(len(scores), dtype=bool))
            row["accepted_physics"] = _physical_summary(
                [e for e, keep in zip(raw_evaluations, mask) if keep], objectives[mask])
        report["selection_curve_note"] = "Descriptive held-out results at validation-derived/explicit thresholds; no test threshold is selected."
        if decoder_norm_sensitivity:
            decoder, _ = load_decoder(decoder_path, model_path, model)
            diagnostics = {}
            for units in ("mae_mw", "rmse_mw", "l1_mw", "linf_mw"):
                with torch.no_grad():
                    scores = reconstruction_errors(model, decoder, tensor(x), units).numpy()
                    validation_scores = reconstruction_errors(
                        model, decoder, tensor(a["validation_x"]), units).numpy()
                q95 = calibrate_threshold(validation_scores, .95)
                diagnostics[units] = {
                    "thresholds": {"fixed_6mw": 6., "fixed_7mw": 7., "validation_q95": q95},
                    "selection_curve": selection_curve(scores, raw, y, [6., 7., q95]),
                }
            report["decoder_norm_sensitivity"] = {
                "note": "Predeclared frozen-decoder score hypotheses. No test-selected norm, refitting, fallback timing or feasibility guarantee.",
                "scores": diagnostics,
            }
    write_json(output / "evaluation.json", report)
    np.savez_compressed(output / "predictions.npz", raw_generation_mw=raw, hybrid_generation_mw=hybrid,
                         batch_generation_mw=batch_prediction, accepted=accepted,
                         reconstruction_error=np.asarray(reconstruction_scores))
    return report
