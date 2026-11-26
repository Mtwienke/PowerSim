"""Produce a report from frozen PowerSim evaluations (training is never run)."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from ..instance_generation.data import write_json
from ..instance_generation.network import digest
from .report_inputs import Inventory, RUNS, load_gate, read_json, sha256
from .report_metrics import (BALANCE_MW, BOUND_MW, accuracy_summary, change, errors, fraction,
                             gate_classification, gate_mask, physics_summary, ranking_metrics)
from .report_runtime import benchmark, environment


SPLITS = ("iid_test", "ood_test_1", "ood_test_2")
UNAVAILABLE = [
    {"item": "Genuine DeepOPF / M3", "reason": "Only the public Gao-example branch-penalty comparator is implemented."},
    {"item": "Other IEEE systems / AC learner", "reason": "No compatible report-scale learned artifacts; IEEE-14 AC/DC support is numerical."},
    {"item": "Demand-only shift", "reason": "Generator exposes load configuration but no saved matched held-out demand-only scenario suite."},
    {"item": "Isolated training-seed confidence interval", "reason": "The two historical seeds also vary sites, data and test stream; other profiles use one training seed."},
    {"item": "Feasibility repair / online adaptation", "reason": "No dispatch projection or test-time learning is implemented; fallback is a replacement, not repair."},
    {"item": "Exact Fig. 6, Fig. 10, Table V and FGSM reproduction", "reason": "Data, decoder norm, attack target and/or systems differ from the paper."},
    {"item": "Germany-30 v2 learner, decoder, and hybrid", "reason": "No compatible v2 checkpoints; v1 checkpoints are not transferable to the new network."},
]
DOCS = Path(__file__).resolve().parents[2] / "docs"
SAMPLE_FIELDS = (
    "experiment_id", "dataset_id", "sample_id", "source_sample_id", "attempt",
    "reference_status", "reference_converged", "reference_valid", "realized_penetration", "accuracy_target",
    "prediction_finite", "mean_absolute_error_mw", "max_absolute_error_mw", "element_hits_lt_1mw",
    "accuracy_reliable", "raw_max_absolute_error_mw", "raw_accuracy_reliable", "raw_feasible",
    "reconstruction_error", "threshold", "score_type", "score_units", "accepted", "fallback_invoked", "fallback_status",
    "balance_residual_mw", "balance_pass", "generator_pass", "line_pass", "feasible",
    "generator_violation_events", "generator_max_violation_mw", "line_violation_events", "line_max_violation_mw",
    "predicted_cost_eur_h", "reference_cost_eur_h", "cost_difference_eur_h", "relative_cost_difference",
)


def _table(output, filename, rows, *, columns=None):
    pd.DataFrame.from_records(rows, columns=columns).to_csv(output / filename, index=False)


def _metadata(entry, dataset, mode=None):
    configuration = mode or entry.mode
    stats = dataset.manifest["splits"][entry.split]
    model_path = entry.inventory.checked(entry.model)
    train = read_json(entry.inventory.checked(model_path.parent / "training.json"))
    train_seed = train["configuration"]["seed"]
    identity = [entry.run, entry.profile, entry.method, entry.checkpoint, entry.split, configuration, dataset.id]
    key = digest(identity + ([train_seed] if entry.run in entry.inventory.clean_suites else []))
    return {
        "experiment_id": key, "dataset_id": dataset.id, "run": entry.run, "profile": entry.profile,
        "network": dataset.network, "method": entry.method, "checkpoint": entry.checkpoint, "evaluation": configuration,
        "scenario": entry.split, "seed": train_seed, "data_seed": dataset.config["seed"],
        "test_seed": dataset.config.get("test_seed"), "branch_limit_scale": dataset.config["simulation"]["branch_limit_scale"],
        "nameplate_penetration": stats.get("nameplate_penetration"),
        "penetration_basis": dataset.config.get("penetration_basis", "nominal_demand"),
    }


def _coverage(labels):
    return {
        "requested": len(labels.attempts), "reference_valid": len(labels.ids),
        "reference_failures": len(labels.attempts)-len(labels.ids),
        "reference_coverage": fraction(len(labels.ids), len(labels.attempts)),
        "reference_converged_attempts": sum(row["reference_converged"] for row in labels.attempts),
        "reference_invalid_after_convergence": sum(row["reference_converged"] and not row["reference_valid"]
                                                   for row in labels.attempts),
    }


def _gate_row(accepted, raw_error, raw_physics):
    complete = np.isfinite(raw_error).all(axis=1)
    accurate = complete & (raw_error < 1).all(axis=1)
    passed = raw_physics["feasible"]
    known = raw_physics["evaluated"]
    accepted_n = int(accepted.sum())
    rejected_n = len(accepted)-accepted_n
    row = {
        "accepted": accepted_n, "rejected": rejected_n, "acceptance_rate": fraction(accepted_n, len(accepted)),
        "rejection_rate": fraction(rejected_n, len(accepted)),
        "accepted_max_lt_1mw": fraction(accurate[accepted].sum(), accepted_n),
        "rejected_max_lt_1mw": fraction(accurate[~accepted].sum(), rejected_n),
        "accepted_element_lt_1mw": fraction((raw_error[accepted] < 1).sum(), accepted_n*raw_error.shape[1]),
        "rejected_element_lt_1mw": fraction((raw_error[~accepted] < 1).sum(), rejected_n*raw_error.shape[1]),
        "accepted_feasible_rate": fraction(passed[accepted].sum(), accepted_n),
        "rejected_feasible_rate": fraction(passed[~accepted].sum(), rejected_n),
    }
    for label, reliable, eligible in (("accuracy", accurate, np.ones(len(accepted), bool)), ("feasibility", passed, known)):
        row.update({f"{label}_{k}": v for k, v in gate_classification(accepted, reliable, eligible).items()})
    return row


def _saved_times(metadata, report, stage_seconds, n):
    t = report["timing"]
    identity = {k: v for k, v in metadata.items() if k not in ("nameplate_penetration", "penetration_basis")}
    common = {**identity, "measurement": "archived", "samples": n, "device": t["device"],
              "workers": t.get("workers", 1), "gpu_synchronization": "not applicable (CPU)"}
    rows = []
    for label, elapsed in (("model_only_batch", t.get("opf_model_only_batch_seconds")),
                           ("evaluation_wall_with_audits_io", t.get("evaluation_loop_seconds_including_physical_audits")),
                           ("stage_wall_with_startup", stage_seconds),
                           ("summed_worker_decisions", t.get("decision_seconds_total_including_decoder_and_fallback"))):
        if elapsed is not None:
            rows.append({**common, "component": label, "total_seconds": elapsed,
                         "mean_seconds_per_sample": fraction(elapsed, n),
                         "samples_per_second": fraction(n, elapsed) if label != "summed_worker_decisions" else None,
                         "p95_latency_seconds": t.get("single_scenario_p95_seconds") if label == "summed_worker_decisions" else None,
                         "batch_size": n if label == "model_only_batch" else 1 if label == "summed_worker_decisions" else None})
    return rows


def _write_samples(writer, key, dataset, attempts, prediction, reference, phys=None, *,
                   raw_error=None, raw_phys=None, scores=None, accepted=None, gate=None, target="scenario_opf"):
    error = errors(prediction, reference)
    rows = error.shape[0]
    complete = np.isfinite(error).all(axis=1)
    gen = np.maximum(phys["gen_lower_violation_mw"], phys["gen_upper_violation_mw"]) if phys is not None else None
    line = phys["line_violation_mw"] if phys is not None else None
    physical_fields = ("balance_residual_mw", "predicted_cost_eur_h", "reference_cost_eur_h",
                       "cost_difference_eur_h", "relative_cost_difference")
    for i, attempt in enumerate(attempts):
        row = {k: v for k, v in attempt.items() if k in SAMPLE_FIELDS}
        row.update(experiment_id=key, dataset_id=dataset.id, accuracy_target=target, prediction_finite=bool(complete[i]),
                   mean_absolute_error_mw=float(error[i].mean()) if complete[i] else None,
                   max_absolute_error_mw=float(error[i].max()) if complete[i] else None,
                   element_hits_lt_1mw=int((error[i] < 1).sum()), accuracy_reliable=bool((error[i] < 1).all()))
        if phys is not None:
            row["feasible"] = bool(phys["feasible"][i])
            if phys["evaluated"][i]:
                row.update({k: bool(phys[k][i]) for k in ("balance_pass", "generator_pass", "line_pass")})
                row.update({k: float(phys[k][i]) if np.isfinite(phys[k][i]) else None for k in physical_fields})
                for tag, matrix in (("generator", gen), ("line", line)):
                    row[f"{tag}_violation_events"] = int((matrix[i] > BOUND_MW).sum())
                    row[f"{tag}_max_violation_mw"] = float(matrix[i].max(initial=0.))
        if gate is not None:
            keep = bool(accepted[i])
            row.update(raw_max_absolute_error_mw=float(raw_error[i].max()) if np.isfinite(raw_error[i]).all() else None,
                       raw_accuracy_reliable=bool((raw_error[i] < 1).all()),
                       raw_feasible=bool(raw_phys["feasible"][i]) if raw_phys["evaluated"][i] else None,
                       reconstruction_error=float(scores[i]) if np.isfinite(scores[i]) else None,
                       threshold=gate["threshold"], score_type=gate["error_units"], score_units=gate["score_units"], accepted=keep,
                       fallback_invoked=not keep,
                       fallback_status="not_needed" if keep else "succeeded" if complete[i] else "failed")
        writer.writerow(row)
    return error


def _gate_analysis(metadata, raw, raw_error, raw_phys, scores, accepted, gate, thresholds, source, timing):
    if not np.array_equal(accepted, gate_mask(scores, np.isfinite(raw).all(axis=1), gate["threshold"])):
        raise ValueError(f"Archived gate decisions do not match threshold: {metadata['experiment_id']}")
    main = _gate_row(accepted, raw_error, raw_phys)
    for label, unreliable, eligible in (
        ("accuracy", ~(raw_error < 1).all(axis=1), np.ones(len(scores), bool)),
        ("feasibility", ~raw_phys["feasible"], raw_phys["evaluated"]),
    ):
        main.update({f"{label}_{k}": v for k, v in ranking_metrics(scores, unreliable, eligible).items()})
    values = {float(x): "validation_quantile" for x in thresholds}
    if gate["error_units"] in ("rmse_mw", "mae_mw"):
        values.update({6.: "fixed_6_mw_diagnostic", 7.: "fixed_7_mw_diagnostic"})
    values[gate["threshold"]] = "archived_operating_threshold"
    fallback_wall = timing.get("fallback_seconds_total")
    decision_wall = timing.get("decision_seconds_total_including_decoder_and_fallback")
    rows = []
    for threshold in [None, *sorted(values)]:
        keep = gate_mask(scores, np.isfinite(raw).all(axis=1), threshold)
        stats = _gate_row(keep, raw_error, raw_phys)
        est = None
        if decision_wall is not None and fallback_wall is not None:
            if stats["rejected"] == 0:
                est = max(0., decision_wall-fallback_wall)
            elif main["rejected"]:
                est = max(0., decision_wall-fallback_wall) + stats["rejected"]*fallback_wall/main["rejected"]
        rows.append({**metadata, "threshold": threshold, "threshold_role": values.get(threshold, "reject_all"),
                     "score_type": gate["error_units"], "score_units": gate["score_units"],
                     "validation_grid_source": source, **stats,
                     "estimated_total_worker_decision_seconds": est})
    return main, rows


def _evaluation(entry, dataset, labels, physics, output, writer):
    report = entry.report
    metadata = _metadata(entry, dataset)
    key = metadata["experiment_id"]
    model_hash = entry.inventory.inputs[entry.inventory.relative(entry.model)]["sha256"]
    if report["model_hash"] != model_hash:
        raise ValueError(f"Saved prediction/model binding differs: {entry.directory}")
    raw, chosen, accepted, scores = entry.predictions(labels.ids, report)
    pred = chosen if entry.decoder else raw
    body = accuracy_summary(pred, labels.y, requested=len(labels.attempts))
    old = report["hybrid_accuracy" if entry.decoder else "raw_accuracy"]
    # Across schema versions the inclusive joint count is an independent
    # compatibility check on the frozen prediction and dataset pairing.
    if not np.isclose(body["legacy_joint_le_1mw"], old["success_rate_1mw"], rtol=0, atol=1e-10):
        raise ValueError(f"Archived score does not match ordered saved products: {entry.directory}")
    phys = physics.evaluate(pred, labels.x, labels.cost)
    matrix = {k: v for k, v in phys.items() if isinstance(v, np.ndarray) and v.ndim == 2}
    gate, curves, extras = None, [], {}
    raw_error, raw_phys = None, None
    binding = {"checkpoint": entry.inventory.relative(entry.model), "checkpoint_sha256": model_hash,
               "prediction_directory": entry.inventory.relative(entry.directory)}
    if entry.decoder:
        entry.inventory.checked(entry.decoder)
        gate, thresholds, source = load_gate(entry.model, entry.decoder, dataset)
        raw_error, raw_phys = errors(raw, labels.y), physics.evaluate(raw, labels.x, labels.cost)
        extras, curves = _gate_analysis(metadata, raw, raw_error, raw_phys, scores, accepted, gate, thresholds,
                                        source, report["timing"])
        fallback_failed = int((~accepted & ~np.isfinite(pred).all(axis=1)).sum())
        if (extras["rejected"] != round(report["fallback_rate"]*len(pred)) or fallback_failed != report["fallback_failures"]
                or not np.array_equal(chosen[accepted], raw[accepted], equal_nan=True)):
            raise ValueError(f"Archived hybrid routing or fallback counts differ: {entry.directory}")
        extras.update(threshold=gate["threshold"], score_type=gate["error_units"], score_units=gate["score_units"],
                      fallback_calls=extras["rejected"],
                      fallback_calls_per_sample=extras["rejection_rate"], fallback_failures=fallback_failed,
                      fallback_feasible_count=int((~accepted & phys["feasible"]).sum()),
                      fallback_decision_seconds_total=report["timing"].get("fallback_seconds_total"),
                      raw_element_lt_1mw=fraction((raw_error < 1).sum(), raw_error.size))
        matrix["raw_absolute_error_mw"] = raw_error
        matrix["reconstruction_error"] = scores
        matrix["accepted"] = accepted
        binding.update(decoder=entry.inventory.relative(entry.decoder),
                       decoder_sha256=entry.inventory.inputs[entry.inventory.relative(entry.decoder)]["sha256"],
                       gate={k: gate.get(k) for k in ("threshold", "calibration_method", "calibration_split", "quantile",
                                                     "error_units", "score_units", "aggregation", "feature_layer", "seed")},
                       validation_thresholds=thresholds, validation_grid_source=source)
    attempts = [labels.lookup[s] for s in labels.ids]
    if entry.mode == "raw-fgsm":
        matrix["source_sample_ids"] = [a["source_sample_id"] for a in attempts]
    error = _write_samples(writer, key, dataset, attempts, pred, labels.y, phys, raw_error=raw_error,
                           raw_phys=raw_phys, scores=scores, accepted=accepted, gate=gate)
    np.savez_compressed(output / "matrices" / f"{key}.npz", sample_ids=labels.ids,
                        generator_ids=dataset.gen_ids, branch_ids=dataset.branch_ids,
                        predicted_generation_mw=pred, reference_generation_mw=labels.y, absolute_error_mw=error,
                        branch_lower_mw=physics.branch_min, branch_upper_mw=physics.branch_max, **matrix)
    row = {**metadata, **_coverage(labels), **body, **physics_summary(phys), **extras,
           "solver_status_note": "Reference solver converged and validated; not independently certified."}
    if entry.raw_view_of_hybrid_stage:
        binding["raw_view_of_hybrid_stage"] = True
    times = [] if entry.raw_view_of_hybrid_stage else _saved_times(metadata, report, entry.stage_seconds, len(pred))
    return row, curves, times, binding


def _fixed_target(entry, dataset, labels, output, writer):
    keys = ("attack_all_generation_mw", "attack_all_original_y", "attack_all_source_ids", "attack_all_delta_mw")
    pred, reference, original_ids, delta = dataset.arrays(*keys)
    attempts = labels.attempts
    if len(attempts) != len(original_ids) or not all(
            a["attempt"] == i and a["source_sample_id"] == original_ids[i] for i, a in enumerate(attempts)):
        raise ValueError("FGSM fixed-target pairing disagrees with archived attack attempts")
    metadata = _metadata(entry, dataset, "fgsm_fixed_original_target")
    error = _write_samples(writer, metadata["experiment_id"], dataset, attempts, pred, reference,
                           target="original_clean_opf")
    np.savez_compressed(output / "matrices" / f"{metadata['experiment_id']}.npz",
                        sample_ids=[a["sample_id"] for a in attempts], source_sample_ids=original_ids,
                        generator_ids=dataset.gen_ids, absolute_error_mw=error, predicted_generation_mw=pred,
                        reference_generation_mw=reference)
    return {**metadata, **_coverage(labels), **accuracy_summary(pred, reference, requested=len(reference)),
            "accuracy_reference": "original_dispatch_all_attacks", "epsilon_linf_mw": dataset.manifest["attack"]["epsilon_mw"],
            "maximum_observed_linf_mw": float(np.abs(delta).max()),
            "median_absolute_total_net_load_change_mw": float(np.median(np.abs(delta.sum(axis=1))))}


def _reference_result(dataset, split, labels, physics):
    site = dataset.manifest["splits"][split]
    shares = [a["realized_penetration"] for a in labels.attempts
              if a["reference_valid"] and a["realized_penetration"] is not None]
    return {
        "experiment_id": digest(["M0", dataset.id, split]), "dataset_id": dataset.id, "network": dataset.network,
        "run": "reference", "profile": "reference", "method": "numerical-dc-OPF (M0 analogue)", "evaluation": "reference",
        "scenario": split, "data_seed": dataset.config["seed"], "test_seed": dataset.config.get("test_seed"),
        "branch_limit_scale": dataset.config["simulation"]["branch_limit_scale"],
        "nameplate_penetration": site.get("nameplate_penetration"),
        "penetration_basis": dataset.config.get("penetration_basis", "nominal_demand"), **_coverage(labels),
        "realized_penetration_mean_reference_valid": float(np.mean(shares)) if shares else None,
        "reference_samples_with_active_branch": int(labels.active.any(axis=1).sum()),
        **physics_summary(physics.evaluate(labels.y, labels.x, labels.cost)),
        "solver_status_note": "Reference itself; no self-comparison accuracy. Feasibility is conditional on valid labels.",
    }


def comparisons(metrics, bindings, matrix_dir):
    raw = [r for r in metrics if r["evaluation"] == "raw"]
    by_key = {(r["run"], r["network"], r["profile"], r["seed"], r["checkpoint"], r["method"], r["scenario"]): r for r in raw}
    comparisons_out = []
    for (run, network, profile, seed, checkpoint, method, scenario), row in by_key.items():
        if scenario != "iid_test":
            iid = by_key.get((run, network, profile, seed, checkpoint, method, "iid_test"))
            if iid:
                if bindings[row["experiment_id"]]["checkpoint_sha256"] != bindings[iid["experiment_id"]]["checkpoint_sha256"]:
                    raise ValueError("OOD generalization must use the identical frozen checkpoint")
                comparisons_out.append(_difference(row, iid, "shift_vs_iid"))
        baselines = {"M5": ("M1", "M1-continued", "M2", "M3-example", "M4"),
                     "physics-guided": ("baseline", "without-L5")}
        for m in baselines.get(method, ()):
            base = by_key.get((run, network, profile, seed, checkpoint, m, scenario))
            if base:
                if row["dataset_id"] != base["dataset_id"]:
                    raise ValueError("Matched-method comparison requires identical dataset")
                comparisons_out.append(_difference(row, base, "method_vs_baseline"))
    for row in metrics:
        if row["evaluation"] not in ("raw-fgsm", "fgsm_fixed_original_target"):
            continue
        source = "iid_test" if row["scenario"] == "attack1" else "ood_test_1"
        clean = by_key.get((row["run"], row["network"], row["profile"], row["seed"], row["checkpoint"], row["method"], source))
        if clean:
            if bindings[row["experiment_id"]]["checkpoint_sha256"] != bindings[clean["experiment_id"]]["checkpoint_sha256"]:
                raise ValueError("FGSM comparison requires the same prediction checkpoint")
            with (
                np.load(matrix_dir / f"{row['experiment_id']}.npz", allow_pickle=False) as a,
                np.load(matrix_dir / f"{clean['experiment_id']}.npz", allow_pickle=False) as c,
            ):
                source_order = {sid: i for i, sid in enumerate(c["sample_ids"])}
                matched = [source_order[sid] for sid in a["source_sample_ids"]]
                matching_accuracy = accuracy_summary(c["predicted_generation_mw"][matched],
                                                     c["reference_generation_mw"][matched], requested=len(matched))
            result = _difference(row, {**clean, **matching_accuracy}, "attack_vs_clean_matched")
            result["paired_cases"] = len(matched)
            comparisons_out.append(result)
    return comparisons_out


def _difference(row, base, contrast):
    result = {k: row[k] for k in ("run", "network", "profile", "method", "scenario", "evaluation", "checkpoint", "seed",
                                  "experiment_id")}
    result.update(contrast=contrast, baseline_method=base["method"], baseline_scenario=base["scenario"],
                  baseline_experiment_id=base["experiment_id"])
    for k in ("element_lt_1mw", "max_lt_1mw", "mean_lt_1mw"):
        result[k], result[k+"_baseline"] = row.get(k), base.get(k)
        result[k+"_difference_pp"], result[k+"_relative_change_pct"] = change(row.get(k), base.get(k))
    result["mae_mw"], result["baseline_mae_mw"] = row.get("mae_mw"), base.get("mae_mw")
    return result


def historical_seed_summary(metrics):
    groups = defaultdict(list)
    for row in metrics:
        if row["run"].startswith("legacy-") and row["network"] == "case30" and row["evaluation"] in ("raw", "hybrid"):
            groups[row["method"], row["scenario"], row["evaluation"]].append(row)
    summary = []
    for (method, scenario, evaluation), rows in groups.items():
        n = len({row["seed"] for row in rows})
        if n != len(rows) or n < 2:
            continue
        for metric in ("element_lt_1mw", "max_lt_1mw", "mean_lt_1mw", "feasible_rate", "mae_mw"):
            values = np.asarray([row[metric] for row in rows], float)
            mean, sd = float(values.mean()), float(values.std(ddof=1))
            radius = float(student_t.ppf(.975, n-1)*sd/np.sqrt(n))
            summary.append(dict(method=method, scenario=scenario, evaluation=evaluation, metric=metric,
                                mean=mean, sample_sd=sd, n=n, ci95_low=mean-radius, ci95_high=mean+radius,
                                scope="Joint protocol seeds; exploratory t interval, not isolated training variation"))
    return summary


def clean_seed_summary(metrics, clean_runs):
    groups = defaultdict(list)
    for row in metrics:
        if row["run"] in clean_runs and row["evaluation"] in ("raw", "hybrid-q95-rmse", "hybrid-fixed6-rmse"):
            groups[row["run"], row["network"], row["method"], row["scenario"], row["evaluation"]].append(row)
    summary = []
    for (run, network, method, scenario, evaluation), rows in groups.items():
        if len({r["dataset_id"] for r in rows}) != 1 or len({r["seed"] for r in rows}) != len(rows):
            raise ValueError("Seed comparisons require distinct trainings on the same dataset")
        for metric in ("element_lt_1mw", "max_lt_1mw", "mean_lt_1mw", "feasible_rate", "mae_mw"):
            values = [r[metric] for r in rows if r[metric] is not None]
            n = len(values)
            sd = float(np.std(values, ddof=1)) if n > 1 else None
            mean = float(np.mean(values)) if n else None
            radius = float(student_t.ppf(.975, n-1)*sd/np.sqrt(n)) if sd is not None else None
            summary.append(dict(run=run, network=network, method=method, scenario=scenario, evaluation=evaluation,
                                metric=metric, mean=mean, sample_sd=sd, n=n,
                                ci95_low=mean-radius if radius is not None else None,
                                ci95_high=mean+radius if radius is not None else None,
                                scope="Independent training seeds on identical data; no uncertainty over new datasets"))
    return summary


def _training_records(inventory, entries):
    rows = []
    for path in sorted({e.model.parent / "training.json" for e in entries} | {
            e.decoder.parent / "decoder.json" for e in entries if e.decoder and (e.decoder.parent / "decoder.json").exists()}):
        a = read_json(inventory.checked(path))
        config = a.get("configuration", {k: v for k, v in a.items() if k not in ("history", "validation_scenario_ids")})
        name = "model_training" if path.name == "training.json" else "decoder_training"
        if name == "decoder_training" and "history" not in a:
            continue  # A gate-only recalibration can inherit training duration.
        seconds = a.get("training_seconds")
        if seconds is None:
            stage = next((r["stages"][path.parent] for r in inventory.runs.values() if path.parent in r["stages"]), {})
            seconds = stage.get("seconds")
        rows.append({"measurement": "archived_training", "component": name, "device": "CPU",
                     "artifact": inventory.relative(path), "total_seconds": seconds, "configuration": config,
                     "selected_epoch": a.get("selected_epoch"), "initialization": a.get("initialization")})
    return rows


def run_report(inventory, entries, output, *, references=(), fresh_runtime=False, repeats=3, limit=2000, figures=True):
    from .report_plots import render

    if output.exists() and set(p.name for p in output.iterdir()) != {"methodology_audit.md"}:
        raise ValueError("Choose a new report output directory; existing results will not be overwritten")
    if figures:
        from importlib.util import find_spec
        if find_spec("matplotlib") is None:
            raise ValueError("Install the report extra or pass --no-figures")
    output.mkdir(parents=True, exist_ok=True)
    for name, source in (("methodology_audit.md", "report-evaluation-audit.md"),
                         ("metric_definitions.md", "report-evaluation-metrics.md")):
        if not (output / name).exists():
            shutil.copyfile(DOCS / source, output / name)
    (output / "matrices").mkdir()
    metrics, curves, runtime, bindings, seen, timing_jobs = [], [], [], {}, {}, []
    clean_plans = {name: read_json(inventory.runs[name]["source"])["plan"] for name in inventory.clean_suites}
    references = [inventory.dataset(path.resolve()) for path in references]
    with (output / "sample_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for entry in entries:
            dataset = inventory.dataset(entry.dataset)
            labels = dataset.split(entry.split)
            key = (dataset.id, entry.split)
            if key not in seen:
                seen[key] = dataset.physics(entry.split, labels)
                metrics.append(_reference_result(dataset, entry.split, labels, seen[key]))
            row, threshold_rows, timing, binding = _evaluation(entry, dataset, labels, seen[key], output, writer)
            if row["experiment_id"] in bindings:
                raise ValueError(f"Duplicate report experiment: {row['experiment_id']}")
            metrics.append(row)
            curves.extend(threshold_rows)
            runtime.extend(timing)
            bindings[row["experiment_id"]] = binding
            if entry.mode == "raw-fgsm":
                attack = dataset.manifest["attack"]
                row.update(accuracy_reference="fresh_opf_at_attacked_input", epsilon_linf_mw=attack["epsilon_mw"],
                           maximum_observed_linf_mw=attack["maximum_linf_mw"])
                fixed = _fixed_target(entry, dataset, labels, output, writer)
                metrics.append(fixed)
                bindings[fixed["experiment_id"]] = {**binding, "target": "fixed_original_dispatch"}
            if fresh_runtime and entry.split == "ood_test_2" and (
                (entry.run == "reconstruction" and entry.profile == "source-informed"
                 and entry.method == "M5" and entry.mode == "hybrid-q95-rmse")
                or (dataset.network == "germany30-v1" and entry.method == "physics-guided" and entry.mode == "hybrid")
                or (entry.run in clean_plans and entry.method == "M5" and entry.mode == "hybrid-q95-rmse"
                    and row["seed"] == clean_plans[entry.run]["seeds"][0])
            ):
                timing_jobs.append((entry, dataset, labels, seen[key]))
    for dataset in references:
        for split in dataset.manifest["splits"]:
            if split not in ("train", "validation") and (dataset.id, split) not in seen:
                labels = dataset.split(split)
                physics = dataset.physics(split, labels)
                seen[dataset.id, split] = physics
                metrics.append(_reference_result(dataset, split, labels, physics))

    attempted = []
    for dataset in inventory._datasets.values():
        for split, labels in dataset._cached_splits.items():
            attempted.extend({"dataset_id": dataset.id, "network": dataset.network, "scenario": split,
                              **{k: v for k, v in a.items() if k != "limits_mw"}} for a in labels.attempts)
            times = [a["reference_solver_seconds"] for a in labels.attempts]
            if times:
                runtime.append(dict(measurement="archived_reference_worker", component="summed_opf_calls",
                                    dataset_id=dataset.id, network=dataset.network, scenario=split, samples=len(times),
                                    total_seconds=float(sum(times)), mean_latency_seconds=float(np.mean(times)),
                                    median_latency_seconds=float(np.median(times)), p95_latency_seconds=float(np.quantile(times, .95)),
                                    failures=len(times)-len(labels.ids), device="CPU",
                                    workers=dataset.manifest.get("execution", {}).get("workers")))
    training = _training_records(inventory, entries)
    runtime.extend({k: v for k, v in row.items() if k not in ("configuration", "initialization")} for row in training)
    _table(output, "metrics.csv", metrics)
    _table(output, "reference_attempts.csv", attempted)
    _table(output, "threshold_curves.csv", curves)
    _table(output, "comparisons.csv", comparisons(metrics, bindings, output / "matrices"))
    _table(output, "historical_seed_summary.csv", historical_seed_summary(metrics))
    if clean_plans:
        _table(output, "training_seed_summary.csv", clean_seed_summary(metrics, clean_plans))
    for entry, dataset, labels, physics in timing_jobs:
        runtime.extend(benchmark(entry, dataset, labels, physics, repeats=repeats, limit=limit))
    _table(output, "runtime.csv", runtime)
    provenance = {name: {k: v for k, v in data.items() if k not in ("stages", "state", "summary", "source")}
                  for name, data in inventory.runs.items()}
    paper = next((read_json(r["source"])["plan"]["paper_table_ii"] for name, r in inventory.runs.items()
                  if name == "reconstruction"), {})
    datasets = {dataset.id: dataset.provenance for dataset in inventory._datasets.values()}
    manifest = {
        "schema_version": 1, "environment": environment(), "artifact_root": str(inventory.base),
        "source_revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                          check=False, cwd=Path(__file__).resolve().parent).stdout.strip() or None,
        "evaluator_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("report_*.py"))},
        "protocol": {"balance_tolerance_mw": BALANCE_MW, "generator_branch_tolerance_mw": BOUND_MW,
                     "accuracy_threshold_mw": [0.5, 1, 2], "positive_class": "unreliable",
                     "hybrid_curve": "descriptive validation-grid thresholds, not tuned on test labels",
                     "fresh_runtime": fresh_runtime, "runtime_repeats": repeats if fresh_runtime else None,
                     "runtime_limit": limit if fresh_runtime else None},
        "runs": provenance, "datasets": datasets, "inputs": inventory.inputs, "model_training": training,
        "experiment_bindings": bindings, "paper_table_ii": paper, "clean_suites": clean_plans,
        "unavailable": [*(UNAVAILABLE if not clean_plans else [UNAVAILABLE[i] for i in (0, 4, 5)]),
                        *inventory.unavailable],
    }
    if not timing_jobs:
        manifest["unavailable"].append({"item": "Controlled fresh speedups", "reason": "Requested no timing or no matching archived timing arms."})
    write_json(output / "experiment_manifest.json", manifest)
    render(output, metrics, curves, runtime, manifest, figures=figures)
    return {"output": str(output), "saved_evaluations": len(entries), "metric_rows": len(metrics),
            "validated_dataset_splits": len(seen), "controlled_runtime_workloads": len(timing_jobs)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/report_evaluation"))
    parser.add_argument("--runs", nargs="+", choices=list(RUNS), default=list(RUNS))
    parser.add_argument("--clean-suite", type=Path, action="append", default=[],
                        help="Report only completed clean suites, excluding historical experiments")
    parser.add_argument("--reference-dataset", type=Path, action="append", default=[])
    parser.add_argument("--plan", action="store_true", help="Inventory only; no inference, optimization, or output files.")
    parser.add_argument("--benchmark-runtime", action="store_true", help="Re-run matched M0 and hybrid CPU timing workloads.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--runtime-limit", type=int, default=2000)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 2 or args.runtime_limit < 1:
        parser.error("requires at least two repeats and a positive runtime limit")
    default_reference = args.artifacts / "report-germany30-v2-seed2027"
    archived_reference = args.artifacts / "archive" / default_reference.name
    if not default_reference.is_dir() and archived_reference.is_dir():
        default_reference = archived_reference
    references = args.reference_dataset or ([default_reference] if default_reference.is_dir() and not args.clean_suite else [])
    inventory = Inventory(args.artifacts, [] if args.clean_suite else dict.fromkeys(args.runs), clean_suites=args.clean_suite)
    entries = list(inventory.evaluations())
    if not args.clean_suite and not any(path.is_dir() for path in references):
        inventory.unavailable.append({"item": "Germany-30 v2 numerical results", "reason": "Reference validation dataset not found."})
    if args.plan:
        import json
        print(json.dumps({"saved_evaluations": {run: sum(e.run == run for e in entries) for run in inventory.runs},
                          "reference_datasets": list(map(str, references)), "new_training": False,
                          "fresh_timing_requested": args.benchmark_runtime,
                          "unavailable": [*(UNAVAILABLE if not args.clean_suite else []), *inventory.unavailable]}, indent=2))
        return 0
    if not entries and not references:
        parser.error("No frozen evaluations or reference datasets available")
    print(run_report(inventory, entries, args.output, references=references, fresh_runtime=args.benchmark_runtime,
                     repeats=args.repeats, limit=args.runtime_limit, figures=not args.no_figures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
