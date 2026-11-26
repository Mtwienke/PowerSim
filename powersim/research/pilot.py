"""Opt-in, stage-resumable Gao training pilot on an unchanged existing dataset."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from ..instance_generation.data import load_dataset, package_versions, write_json
from ..solvers.gao_double_neural.ml import (METHOD_WEIGHTS, TrainConfig, artifact_hash, dataset_identity,
                                           train_model, training_data_hash)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PilotConfig:
    seed: int = 2027
    pretrain_epochs: int = 1000
    fine_tune_epochs: int = 200
    learning_rates: tuple[float, ...] = (.01,)
    hidden: tuple[int, ...] = (200, 200, 200)
    activation: str = "tanh"
    batch_size: int = 64
    checkpoint_selection: str = "last"
    perturb_std_normalized: float = .1

    def __post_init__(self):
        if (not self.learning_rates or len(set(self.learning_rates)) != len(self.learning_rates)
                or not np.isfinite(self.learning_rates).all() or min(self.learning_rates) <= 0):
            raise ValueError("Pilot learning rates must be distinct, finite and positive")
        for epochs in (self.pretrain_epochs, self.fine_tune_epochs):
            TrainConfig(hidden=self.hidden, activation=self.activation, epochs=epochs,
                        learning_rate=self.learning_rates[0], batch_size=self.batch_size,
                        checkpoint_selection=self.checkpoint_selection,
                        perturb_std_normalized=self.perturb_std_normalized)


def pilot_plan(dataset, config: PilotConfig | None = None):
    """Validate inputs and return a frozen plan without creating any output."""
    config = config or PilotConfig()
    dataset = Path(dataset).resolve()
    manifest, arrays = load_dataset(dataset)
    if manifest["configuration"]["simulation"]["physics_mode"] != "dc":
        raise ValueError("The paper training pilot requires a DC dataset")
    if not len(arrays["train_x"]) or not len(arrays["validation_x"]):
        raise ValueError("The pilot requires nonempty training and validation splits")
    sources = sorted(PACKAGE_ROOT.rglob("*.py"))
    plan = {
        "schema_version": "1.0",
        "paper_doi": "10.1109/TNNLS.2024.3437741",
        "configuration": asdict(config),
        "dataset": str(dataset),
        "dataset_id": dataset_identity(manifest),
        "training_data_sha256": training_data_hash(arrays),
        "dataset_artifacts": {name: artifact_hash(dataset / name) for name in ("manifest.json", "dataset.npz")},
        "benchmark": manifest["configuration"]["simulation"]["benchmark"],
        "reference_coverage": {split: {key: stats[key] for key in ("requested", "valid")}
                               for split, stats in manifest["splits"].items()},
        "source_sha256": {str(path.relative_to(PACKAGE_ROOT)): artifact_hash(path) for path in sources},
        "versions": package_versions(),
        "planned_stages": len(config.learning_rates)*5,
        "fine_tuning_methods": {
            "M1-continued": METHOD_WEIGHTS["M1"],
            **{name: METHOD_WEIGHTS[name] for name in ("M2", "M4", "M5")},
        },
        "initialization_policy": "Every continuation at one learning rate starts from the same selected M1 checkpoint; fresh Adam state.",
        "checkpoint_policy": "Keep final and best-validation weights; model.pt follows checkpoint_selection.",
        "scope": "Training/validation pilot only. Reuses the dataset without resampling, repairs, held-out evaluation or automatic decoder training.",
        "assumptions": [
            "The 1000 + 200 epoch schedule and 0.01 OPF learning rate are verified for the paper's IEEE-30 study.",
            "Applying that schedule to Germany-30 is a transfer experiment, not a published German result.",
            "Activation, batch size, normalization and perturbation distribution/magnitude remain explicit implementation assumptions.",
            "Final-epoch checkpoint selection is the pilot default, not a verified author checkpoint-selection rule.",
            "M3 is DeepOPF and is not implemented by these loss ablations.",
            "Historical held-out data have already been inspected; new confirmatory claims require fresh reserved tests.",
        ],
    }
    return json.loads(json.dumps(plan))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def run_pilot(dataset, output, config: PilotConfig | None = None, *, resume=False):
    config = config or PilotConfig()
    plan = pilot_plan(dataset, config)
    output = Path(output)
    if resume:
        if not output.is_dir():
            raise ValueError("Cannot resume a missing pilot directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
    # Never steal a lock: after a hard kill its owner must be checked manually.
    lock = output / ".pilot.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        manifest_path, state_path = output / "pilot_manifest.json", output / "status.json"
        if resume:
            if json.loads(manifest_path.read_text()) != plan:
                raise ValueError("Pilot inputs, configuration, source or package versions changed; use a new output")
            state = json.loads(state_path.read_text())
        else:
            _save(manifest_path, plan)
            state = {"started_utc": _now(), "stages": {}}
        state.update(status="running", updated_utc=_now(), current_stage=None)
        state.pop("diagnostic", None)
        _save(state_path, state)

        def stage(name, epochs, learning_rate, method, parent=None):
            for file, checksum in plan["dataset_artifacts"].items():
                if artifact_hash(Path(plan["dataset"]) / file) != checksum:
                    raise ValueError(f"Dataset artifact changed during pilot: {file}")
            for file, checksum in plan["source_sha256"].items():
                if artifact_hash(PACKAGE_ROOT / file) != checksum:
                    raise ValueError(f"Source changed during pilot: {file}")
            if package_versions() != plan["versions"]:
                raise ValueError("Package versions changed during pilot")
            previous = state["stages"].get(name, {})
            required = ("model.pt", "best.pt", "last.pt", "training.json")
            if previous.get("status") == "complete":
                directory = output / previous["directory"]
                if set(previous["sha256"]) != set(required):
                    raise ValueError(f"Incomplete recorded checksums: {name}")
                for file, checksum in previous["sha256"].items():
                    if artifact_hash(directory / file) != checksum:
                        raise ValueError(f"Completed pilot output changed: {name}/{file}")
                return directory
            attempt = previous.get("attempt", 0)+1
            directory = output / "stages" / name / f"attempt-{attempt:03d}"
            while directory.exists():
                attempt += 1
                directory = output / "stages" / name / f"attempt-{attempt:03d}"
            entry = {"status": "running", "attempt": attempt, "started_utc": _now(),
                     "directory": str(directory.relative_to(output))}
            state["stages"][name] = entry
            state.update(current_stage=name, updated_utc=_now())
            _save(state_path, state)
            started = time.perf_counter()
            train_model(dataset, directory, TrainConfig(
                hidden=config.hidden, activation=config.activation, epochs=epochs,
                batch_size=config.batch_size, learning_rate=learning_rate, seed=config.seed,
                perturb_std_normalized=config.perturb_std_normalized,
                checkpoint_selection=config.checkpoint_selection,
                loss_weights=dict(METHOD_WEIGHTS[method])), initialize_from=parent)
            entry.update(status="complete", finished_utc=_now(), seconds=time.perf_counter()-started,
                         sha256={file: artifact_hash(directory / file) for file in required})
            state.update(updated_utc=_now(), current_stage=None)
            _save(state_path, state)
            return directory

        reports = []
        try:
            for rate in config.learning_rates:
                prefix = f"lr-{rate}--"
                m1 = stage(prefix+"M1", config.pretrain_epochs, rate, "M1")
                for name in ("M1", "M1-continued", "M2", "M4", "M5"):
                    if artifact_hash(m1 / "model.pt") != state["stages"][prefix+"M1"]["sha256"]["model.pt"]:
                        raise ValueError("Shared M1 initializer changed during pilot")
                    directory = m1 if name == "M1" else stage(
                        prefix+name, config.fine_tune_epochs, rate,
                        "M1" if name == "M1-continued" else name, m1 / "model.pt")
                    report = json.loads((directory / "training.json").read_text())
                    reports.append({
                        "method": name, "learning_rate": rate,
                        "directory": str(directory.relative_to(output)),
                        "initialization": report["initialization"],
                        "selected_epoch": report["selected_epoch"],
                        "training_accuracy": report["training_accuracy"],
                        "validation_accuracy": report["validation_accuracy"],
                    })
                    _save(output / "summary.json", {
                        "scope": plan["scope"], "held_out_evaluation_performed": False, "training": reports})
            state.update(status="complete", finished_utc=_now(), current_stage=None)
            _save(state_path, state)
        except BaseException as exc:
            current = state["stages"].get(state.get("current_stage"))
            if current is not None and current["status"] == "running":
                current.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                               finished_utc=_now())
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         updated_utc=_now(), diagnostic=f"{type(exc).__name__}: {exc}")
            _save(state_path, state)
            raise
        return {"directory": str(output), "completed_stages": len(state["stages"]),
                "held_out_evaluation_performed": False, "training": reports}
    finally:
        lock.unlink()
