"""Read-only Gao preprocessing diagnostics, never a training or data-cleaning policy.

Equations are independently expressed from the documented public-example
behavior. Upstream code is not imported or executed. Only train/validation
arrays are read; the paper/source conflicts remain explicit in the report.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import package_versions, write_json


SOURCE_REVISION = "e775ce3a24af0408aed9c02236abd862707cbae0"
POWER_BASE_MW = 100.0
_DATA_ARRAYS = {"x_mean", "x_scale", "y_mean", "y_scale", "m_g", "m_d", "branch_constant", "branch_limits_mw"}


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _jacobians(values):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 3 or min(values.shape[1:]) < 1
            or not np.isfinite(values).all()):
        raise ValueError("Finite sample x generator x input Jacobians required")
    return values


def example_sensitivity_mask(jacobians):
    """Exact signed-max/nonzero-sum predicate on finite supplied Jacobians.

    Not max(abs(J)), not clipping, and not a supervised-sample exclusion rule.
    Nonfinite inputs are rejected by this audit, not repaired.
    """
    j = _jacobians(jacobians)
    return (j.sum(axis=(1, 2)) != 0) & (j.max(axis=(1, 2)) < 10)


def sensitivity_policy_summary(jacobians, available):
    j = _jacobians(jacobians)
    available = np.asarray(available, dtype=bool)
    if available.shape != (len(j),):
        raise ValueError("One availability flag per Jacobian required")
    kept = available & example_sensitivity_mask(j)
    masked = available & ~kept
    energy = np.square(j).sum(axis=(1, 2))
    total = float(energy[available].sum())
    rows = np.flatnonzero(masked)
    top = rows[np.argsort(energy[rows])[::-1][:5]]
    return {
        "supervised_cases_unchanged": len(j),
        "available_jacobians": int(available.sum()),
        "example_kept_jacobians": int(kept.sum()),
        "additionally_masked_jacobians": int(masked.sum()),
        "masked_reference_squared_magnitude_share": float(energy[masked].sum()/total) if total else None,
        "additionally_masked_filtered_rows": rows.tolist(),
        "largest_masked_filtered_rows": top.tolist(),
        "note": "Reference magnitude, not trained loss or causal evidence. L0 cases and labels are unchanged.",
    }


def supervised_loss_scales(prediction_mw, target_mw):
    """Compare error units at identical predictions; no reparameterized training."""
    prediction, target = np.asarray(prediction_mw), np.asarray(target_mw)
    if (prediction.shape != target.shape or prediction.ndim != 2 or not prediction.size
            or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
        raise ValueError("Matching finite nonempty sample x generator arrays required")
    error = prediction-target
    return {
        "mse_mw2": float(np.square(error).mean()),
        "mae_mw": float(np.abs(error).mean()),
        "mse_per_unit2_on_100mw_base": float(np.square(error/POWER_BASE_MW).mean()),
        "example_mae_per_unit_on_100mw_base": float(np.abs(error/POWER_BASE_MW).mean()),
    }


def jacobian_loss_snapshot(predicted, reference, available):
    """Full-population reductions, not an average of minibatch update losses."""
    predicted, reference = _jacobians(predicted), _jacobians(reference)
    available = np.asarray(available, dtype=bool)
    if (predicted.shape != reference.shape or not len(predicted)
            or available.shape != (len(reference),)):
        raise ValueError("Matching nonempty Jacobians and availability flags required")
    kept = available & example_sensitivity_mask(reference)
    residual2 = np.square(predicted-reference)
    balance = predicted.sum(axis=1)-1
    return {
        "pilot_L1_available_entry_mean_square": float(residual2[available].mean()) if available.any() else 0.,
        "example_L1_masked_full_tensor_mean_square": float(residual2[kept].sum()/residual2.size),
        "pilot_L2_mean_square": float(np.square(balance).mean()),
        "example_balance_mean_absolute": float(np.abs(balance).mean()),
        "note": "Fixed saved weights and full-split denominators. No L3/L4 or complete source-objective parity is claimed.",
    }


def _model_snapshot(path, manifest, arrays):
    # Optional PyTorch dependency, and no optimizer or training call.
    import torch
    from .ml import dataset_identity, load_model, physical_jacobian, state_dict_hash, tensor, training_data_hash

    checksum = _sha(path)
    model, checkpoint = load_model(path)
    if (checkpoint["dataset_id"] != dataset_identity(manifest)
            or checkpoint["bus_ids"] != manifest["bus_ids"]
            or checkpoint["generator_ids"] != manifest["generator_ids"]
            or checkpoint.get("training_data_sha256") != training_data_hash(arrays)):
        raise ValueError("Audit checkpoint must have a verified matching train/validation identity")
    state_checksum = state_dict_hash(model.state_dict())
    if checkpoint.get("state_dict_sha256") != state_checksum:
        raise ValueError("Audit checkpoint state checksum mismatch")
    torch.set_num_threads(1)
    splits = {}
    for split in ("train", "validation"):
        predictions, jacobians = [], []
        for start in range(0, len(arrays[f"{split}_x"]), 64):
            x = tensor(arrays[f"{split}_x"][start:start+64])
            with torch.no_grad():
                predictions.append(model(x).numpy())
            jacobians.append(physical_jacobian(model, x, create_graph=False).detach().numpy())
        splits[split] = {
            "supervised_error_units": supervised_loss_scales(
                np.concatenate(predictions), arrays[f"{split}_y"]),
            "jacobian_reductions": jacobian_loss_snapshot(
                np.concatenate(jacobians), arrays[f"{split}_jacobian"], arrays[f"{split}_jacobian_valid"]),
        }
    if _sha(path) != checksum or state_dict_hash(model.state_dict()) != state_checksum:
        raise ValueError("Saved model changed during the read-only audit")
    return {
        "path": str(Path(path).resolve()), "sha256": checksum, "epoch": checkpoint["epoch"],
        "configuration": checkpoint["configuration"], "state_sha256_unchanged": state_checksum,
        "train_validation_data_verified": True, "splits": splits,
    }


def audit_preprocessing(dataset, output, *, model_paths=()):
    """Write a new diagnostic artifact without changing data, models or defaults."""
    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError("Audit output already exists; use a new directory")
    if output.resolve().is_relative_to(dataset.resolve()):
        raise ValueError("Audit output must be separate from the input dataset")
    checksums = {name: _sha(dataset/name) for name in ("manifest.json", "dataset.npz")}
    manifest = json.loads((dataset/"manifest.json").read_text())
    simulation = manifest["configuration"]["simulation"]
    if simulation["benchmark"] != "case30" or simulation["physics_mode"] != "dc":
        raise ValueError("This Gao audit is restricted to IEEE-30 DC; Germany is excluded")
    with np.load(dataset/"dataset.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files
                  if name.startswith(("train_", "validation_")) or name in _DATA_ARRAYS}
    splits = {}
    for split in ("train", "validation"):
        x = arrays[f"{split}_x"]
        if not len(x) or not np.isfinite(x).all():
            raise ValueError("Finite nonempty training and validation inputs required")
        splits[split] = sensitivity_policy_summary(
            arrays[f"{split}_jacobian"], arrays[f"{split}_jacobian_valid"])
        splits[split]["samples_with_active_branch"] = int(np.any(arrays[f"{split}_active_branch"], axis=1).sum())
    conventional = float(sum(g["p_max_mw"] for g in manifest["generators"]))
    nominal = float(sum(manifest["nominal_gross_load_mw"]))
    candidates = []
    for rate in manifest["configuration"]["penetration_rates"]:
        candidates.append({
            "printed_rate": rate,
            "existing_demand_denominator_capacity_mw": rate*nominal,
            "hypothetical_total_capacity_denominator_mw": rate*conventional/(1-rate) if rate < 1 else None,
        })
    perturb_std = .1*np.asarray(arrays["x_scale"])
    report = {
        "schema_version": "1.0", "source_revision": SOURCE_REVISION,
        "scope": "Read-only training/validation diagnostics; no source execution, sample/label edits, training, held-out metrics or Germany.",
        "source_caveat": "Public examples conflict with the journal; see docs/gao-source-audit.md.",
        "dataset": str(dataset.resolve()), "dataset_sha256": checksums,
        "array_scope": "Only train_*, validation_* and shared normalization/DC-map arrays are loaded from NPZ.",
        "splits": splits,
        "units": {
            "example_consumer_power_base_mw": POWER_BASE_MW,
            "example_noise_std_input_units": 1.,
            "inferred_example_noise_std_mw": POWER_BASE_MW,
            "historical_pilot_noise_std_mw_min": float(perturb_std.min()),
            "historical_pilot_noise_std_mw_max": float(perturb_std.max()),
            "jacobian_chain_rule": "Common input/output per-unit base cancels: d(Y/base)/d(X/base) = dY/dX.",
        },
        "penetration_hypothesis": {
            "status": "Arithmetic hypothesis only; no author generator is available.",
            "conventional_pmax_mw": conventional, "nominal_demand_mw": nominal, "candidates": candidates,
        },
        "models": [_model_snapshot(path, manifest, arrays) for path in model_paths],
        "audit_source_sha256": _sha(__file__), "versions": package_versions(),
    }
    if model_paths:
        report["ml_source_sha256"] = _sha(Path(__file__).with_name("ml.py"))
    if any(_sha(dataset/name) != checksum for name, checksum in checksums.items()):
        raise ValueError("Input dataset changed during audit")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"audit.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("output")
    parser.add_argument("--model", action="append", default=[], help="Optional matching checkpoint; repeat for fixed-weight diagnostics")
    args = parser.parse_args(argv)
    report = audit_preprocessing(args.dataset, args.output, model_paths=args.model)
    print(json.dumps({
        "output": str(Path(args.output)/"audit.json"),
        "source_revision": report["source_revision"],
        "splits": {split: {key: value for key, value in row.items() if not key.endswith("_rows")}
                   for split, row in report["splits"].items()},
        "fixed_model_snapshots": len(report["models"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
