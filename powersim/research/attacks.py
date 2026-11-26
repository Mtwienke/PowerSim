"""Evaluation-only white-box FGSM, with new OPF labels at perturbed inputs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch

from ..core.config import SimulationConfig
from ..core.dc import DCRelation
from ..core.workers import ordered_pool, validate_workers
from ..instance_generation.data import (_init_reference_worker, _solve_scenario, _write_split, load_dataset,
                                        package_versions, write_json)
from ..instance_generation.network import digest
from ..output_generation.metrics import accuracy
from ..solvers.gao_double_neural.ml import (TrainConfig, artifact_hash, dataset_identity, load_model,
                                           supervised_loss, tensor)


def fgsm_inputs(model, x_mw, target_mw, config, epsilon_mw=1.):
    """Sign of the declared supervised loss; no model update or input clipping."""
    if not np.isfinite(epsilon_mw) or epsilon_mw < 0:
        raise ValueError("FGSM epsilon must be finite and nonnegative")
    x = np.asarray(x_mw, dtype=float)
    target = np.asarray(target_mw, dtype=float)
    if x.ndim != 2 or target.ndim != 2 or len(x) != len(target) or not len(x):
        raise ValueError("Nonempty paired attack inputs/targets required")
    if not np.isfinite(x).all() or not np.isfinite(target).all():
        raise ValueError("Attack inputs/targets must be finite")
    deltas = []
    for start in range(0, len(x), 64):
        bx = tensor(x[start:start+64]).requires_grad_(True)
        loss = supervised_loss(model(bx), tensor(target[start:start+64]), model, config)
        gradient = torch.autograd.grad(loss, bx)[0]
        if not torch.isfinite(gradient).all():
            raise ValueError("Non-finite FGSM input gradient")
        deltas.append(gradient.sign().detach().numpy()*epsilon_mw)
    # Add to original float64 MW inputs, not their float32 round-trip.
    delta = np.concatenate(deltas).astype(float)
    attacked = x+delta
    if np.max(np.abs(attacked-x)) > epsilon_mw+1e-9:
        raise ValueError("FGSM L-infinity budget exceeded")
    return attacked, attacked-x


def generate_fgsm_dataset(dataset, model_path, output, *, split="iid_test", epsilon_mw=1., workers=1):
    validate_workers(workers)
    started = time.perf_counter()
    dataset, output = Path(dataset), Path(output)
    if output.exists() or output.resolve().is_relative_to(dataset.resolve()):
        raise ValueError("Attack output must be new and separate from the source dataset")
    manifest, arrays = load_dataset(dataset, sensitivities=False)
    if split in ("train", "validation") or split not in manifest["splits"]:
        raise ValueError("FGSM generation requires a held-out split")
    simulation_values = manifest["configuration"]["simulation"]
    if simulation_values["benchmark"] != "case30" or simulation_values["physics_mode"] != "dc":
        raise ValueError("Reconstruction attacks are restricted to IEEE-30 DC; Germany is excluded")
    model, checkpoint = load_model(model_path)
    if checkpoint["dataset_id"] != dataset_identity(manifest):
        raise ValueError("FGSM checkpoint/source dataset mismatch")
    source_hashes = {name: artifact_hash(dataset/name) for name in ("manifest.json", "dataset.npz")}
    model_hash = artifact_hash(model_path)
    torch.set_num_threads(1)
    config = TrainConfig(**checkpoint["configuration"])
    x, original_y = arrays[f"{split}_x"], arrays[f"{split}_y"]
    if (set(arrays[f"{split}_scenario_ids"]) & set(checkpoint["training_scenario_ids"]+checkpoint["validation_scenario_ids"])
            or {digest(row.tolist()) for row in x} & set(checkpoint["training_input_ids"]+checkpoint["validation_input_ids"])):
        raise ValueError("FGSM source split overlaps training/validation")
    attacked, delta = fgsm_inputs(model, x, original_y, config, epsilon_mw)
    with torch.no_grad():
        fixed_predictions = np.concatenate([model(tensor(attacked[start:start+64])).numpy()
                                             for start in range(0, len(x), 64)])
    attack_split = {"iid_test": "attack1", "ood_test_1": "attack2"}.get(split, f"attack_{split}")
    records = []
    for i, (load, sid) in enumerate(zip(attacked, arrays[f"{split}_scenario_ids"])):
        records.append({
            "scenario_id": digest({"source_scenario_id": str(sid), "model_hash": model_hash,
                                   "epsilon_mw": epsilon_mw, "x": load.tolist()}),
            "source_scenario_id": str(sid), "number": i, "split": attack_split,
            "net_load_mw": load.tolist(), "input_id": digest(load.tolist()),
            "attack_delta_mw": delta[i].tolist(),
        })
    relation = DCRelation(arrays["m_g"], arrays["m_d"], arrays["branch_constant"],
                          tuple(manifest["branch_ids"]), 0.)
    gens = [SimpleNamespace(**row) for row in manifest["generators"]]
    output.mkdir(parents=True, exist_ok=False)
    attack_arrays = {}
    simulation = SimulationConfig(**manifest["configuration"]["simulation"])
    # Reuse the independently validated reference-label writer; old y is not
    # passed to it and cannot silently become a physical label at new x.
    with ordered_pool(workers, _init_reference_worker, (simulation,)) as map_cases:
        stats = _write_split(output, attack_split, len(records), manifest["splits"][split]["nameplate_penetration"],
                             np.random.SeedSequence(0),
                             map_cases(_solve_scenario, records), attack_arrays,
                              np.asarray(manifest["nominal_gross_load_mw"]), gens, relation,
                              compute_sensitivities=not manifest["configuration"].get("training_sensitivities_only", False),
                              record_format=manifest.get("record_format", "jsonl"))
    original_lookup = {record["scenario_id"]: i for i, record in enumerate(records)}
    valid_rows = [original_lookup[str(sid)] for sid in attack_arrays[f"{attack_split}_scenario_ids"]]
    attack_arrays[f"{attack_split}_original_y"] = original_y[valid_rows]
    attack_arrays.update(
        attack_all_x=attacked, attack_all_delta_mw=delta, attack_all_original_y=original_y,
        attack_all_generation_mw=fixed_predictions, attack_all_source_ids=arrays[f"{split}_scenario_ids"])
    for key in ("x_mean", "x_scale", "y_mean", "y_scale", "m_g", "m_d", "branch_constant", "branch_limits_mw"):
        if key in arrays:
            attack_arrays[key] = arrays[key]
    np.savez_compressed(output/"dataset.npz", **attack_arrays)
    if (dataset/"network.json").exists():
        shutil.copyfile(dataset/"network.json", output/"network.json")
    attack_metadata = {
        "method": "White-box FGSM of declared supervised loss, evaluated without fitting",
        "source_split": split, "split": attack_split, "epsilon_mw": epsilon_mw,
        "maximum_linf_mw": float(np.abs(delta).max()), "model_hash": model_hash,
        "source_dataset": str(dataset.resolve()), "source_dataset_sha256": source_hashes,
        "model_path": str(Path(model_path).resolve()), "model_configuration": asdict(config),
        "fixed_original_target_accuracy_all_attacks": accuracy(fixed_predictions, original_y),
        "original_scenarios": len(x), "new_reference_valid": stats["valid"],
        "label_note": "Split y is freshly re-solved OPF at attacked x. Fixed-original-target metrics are separate and include all attacks, even new reference failures.",
        "versions": package_versions(), "seconds": time.perf_counter()-started,
    }
    attack_manifest = {**manifest, "splits": {attack_split: stats}, "evaluation_only": True,
                       "attack": attack_metadata}
    if artifact_hash(model_path) != model_hash or any(
            artifact_hash(dataset/name) != checksum for name, checksum in source_hashes.items()):
        raise ValueError("Source data/model changed during FGSM generation")
    write_json(output/"manifest.json", attack_manifest)
    write_json(output/"attack.json", attack_metadata)
    return attack_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("model")
    parser.add_argument("output")
    parser.add_argument("--split", default="iid_test")
    parser.add_argument("--epsilon-mw", type=float, default=1.)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    print(json.dumps(generate_fgsm_dataset(
        args.dataset, args.model, args.output, split=args.split, epsilon_mw=args.epsilon_mw,
        workers=args.workers), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
