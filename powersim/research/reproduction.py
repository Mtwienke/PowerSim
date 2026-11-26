"""Explicit, frozen-source, stage-resumable IEEE-30 reconstruction suite."""
import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time

from ..core.config import SimulationConfig
from ..core.workers import THREAD_ENV, validate_workers
from ..instance_generation.data import DatasetConfig, package_versions, write_json
from ..solvers.gao_double_neural.ml import METHOD_WEIGHTS, TrainConfig, artifact_hash


PROFILES = ("source-informed", "journal-equations", "public-example")
METHODS = ("M1", "M1-continued", "M2", "M3-example", "M4", "M5")
SPLITS = ("iid_test", "ood_test_1", "ood_test_2")


@dataclass(frozen=True)
class SuiteConfig:
    seed: int = 2027
    test_seed: int = 20281121
    workers: int = 8
    train_cases: int = 10000
    validation_cases: int = 1000
    test_cases: int = 2000
    pretrain_epochs: int = 1000
    fine_tune_epochs: int = 200
    decoder_epochs: int = 1000
    hidden: tuple[int, ...] = (200, 200, 200)
    batch_size: int = 50
    decoder_batch_size: int = 500
    minimum_free_gib: float = 5.

    def __post_init__(self):
        validate_workers(self.workers)
        if (any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                for v in (self.decoder_epochs, self.decoder_batch_size))
                or not 0 <= self.minimum_free_gib < 1e6):
            raise ValueError("Invalid decoder budget or disk reserve")
        data_configuration(self)
        for profile in PROFILES:
            training_configuration(self, profile, "M1")
            training_configuration(self, profile, "M5")

    @classmethod
    def smoke(cls):
        return cls(seed=20261119, test_seed=20261120, workers=1, train_cases=8, validation_cases=3, test_cases=3,
                   pretrain_epochs=2, fine_tune_epochs=1, decoder_epochs=2,
                   hidden=(8, 8, 8), batch_size=4, decoder_batch_size=4)


def data_configuration(config):
    return DatasetConfig(
        simulation=SimulationConfig(benchmark="case30", branch_limit_scale=1.),
        seed=config.seed, placement_seed=config.seed, test_seed=config.test_seed,
        train_cases=config.train_cases, validation_cases=config.validation_cases, test_cases=config.test_cases,
        penetration_basis="total_installed", renewable_capacities_mw=(185., 230., 300.),
        sampling_policy="valid", max_attempt_factor=10)


def training_configuration(config, profile, method):
    if profile not in PROFILES or method not in METHODS:
        raise ValueError("Unknown reconstruction profile/method")
    weights = dict(METHOD_WEIGHTS["M1" if method == "M1-continued" else method])
    if profile == "public-example" and method in ("M4", "M5"):
        weights["L3"] = 0.
    return TrainConfig(
        hidden=config.hidden, activation="relu", seed=config.seed, batch_size=config.batch_size,
        epochs=config.pretrain_epochs if method == "M1" else config.fine_tune_epochs,
        learning_rate=.01 if profile == "journal-equations" else .001,
        loss_weights=weights, checkpoint_selection="last", normalization="per_unit", power_base_mw=100.,
        prediction_loss="mse" if profile == "journal-equations" else "mae", prediction_units="model",
        pattern_norm="mse" if profile == "journal-equations" else "mae",
        mask_reduction="full", sensitivity_policy="example", perturb_std_normalized=1.,
        branch_subset=("line:20",) if profile == "public-example" else (),
        initializer="glorot_uniform", adam_variant="keras_v2", adam_epsilon=1e-7, learning_rate_decay=2e-6)


def decoder_configuration(config, profile):
    return {
        "epochs": config.decoder_epochs, "batch_size": config.decoder_batch_size,
        "hidden": [200]*(3 if profile == "public-example" else 2),
        "activation": "relu",
        "feature_layer": "penultimate_hidden" if profile == "journal-equations" else "last_hidden",
        "initializer": "glorot_uniform", "adam_variant": "keras_v2", "adam_epsilon": 1e-7,
        "learning_rate": .001, "learning_rate_decay": 2e-6,
        "checkpoint_selection": "last", "seed": config.seed, "units": "rmse_mw", "threshold": 6.,
    }


def suite_plan(config=None):
    config = config or SuiteConfig()
    plan = {
        "schema_version": "1.0", "configuration": asdict(config), "benchmark": "case30", "physics": "dc",
        "protocol": "docs/gao-reconstruction-suite.md",
        "upstream_example_revision": "e775ce3a24af0408aed9c02236abd862707cbae0",
        "primary_profile": "source-informed",
        "scope": "Approximate reconstruction with explicit source/paper conflicts; no Germany, author contact, automatic tuning or dispatch repair.",
        "dataset_configuration": asdict(data_configuration(config)),
        "training": {profile: {method: asdict(training_configuration(config, profile, method))
                              for method in METHODS} for profile in PROFILES},
        "decoders": {profile: decoder_configuration(config, profile) for profile in PROFILES},
        "counts": {"datasets": 1, "trainings": 18, "decoders": 18, "calibrations": 1,
                   "attack_datasets": 36, "evaluations": 165, "stages": 239},
        "checkpoint_policy": "Final M1 parents and final primary results; primary-profile best-validation raw checkpoints are a separate secondary arm.",
        "attack_policy": "1 MW self-FGSM on IID and first OOD. Fixed-original targets and freshly re-solved OPF targets reported separately.",
        "gate_policy": "Fixed 6 MW physical RMSE; fixed 6/7 MAE/RMSE/L1/Linf score diagnostics. Primary M5 q95 RMSE is a secondary hybrid.",
        "selection_policy": "No test-dependent profile, checkpoint, norm, threshold, seed or capacity selection.",
        "M3_status": "M3-example is the visible Gao branch-penalty comparator, not the full external security-constrained DeepOPF algorithm.",
        "paper_table_ii": {
            "source": "Gao et al., DOI 10.1109/TNNLS.2024.3437741, p.8055, Table II",
            "columns": ["iid_test", "ood_test_1", "ood_test_2", "attack1", "attack2"],
            "M1": [99.94, 89.59, 69.31, 96.79, 51.72],
            "M2": [99.89, 96.18, 88.95, 98.67, 84.65],
            "M3": [99.94, 88.21, 72.57, 96.53, 46.61],
            "M4": [99.96, 99.15, 92.66, 99.83, 92.58],
            "M5": [99.92, 99.16, 97.94, 99.29, 93.71],
        },
    }
    return json.loads(json.dumps(plan))


def _now():
    return datetime.now(timezone.utc).isoformat()


def _save(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def _configuration_file(path, value):
    value = json.loads(json.dumps(value))
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen configuration changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, value)
    return path


def _freeze(root):
    source, archive, ready = root/"source", root/"source.tar", root/"source.ready"
    if ready.exists():
        if not source.is_dir() or not archive.is_file():
            raise ValueError("Incomplete frozen source; preserve it and use a new run directory")
        return ready.read_text().strip(), source, archive
    if source.exists() or archive.exists():
        raise ValueError("Incomplete source snapshot preserved; use a new run directory")
    repository = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip():
        raise ValueError("Commit project changes before freezing the suite")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    archive.write_bytes(subprocess.check_output(["git", "archive", revision], cwd=repository))
    source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive.read_bytes())) as stream:
        stream.extractall(source, filter="data")
    ready.write_text(revision+"\n")
    return revision, source, archive


def verify_suite(root, state=None):
    root = Path(root)
    state = state or json.loads((root/"status.json").read_text())
    count = 0
    for name, stage in state["stages"].items():
        if stage["status"] != "complete":
            raise ValueError(f"Cannot verify incomplete stage: {name}")
        for filename, checksum in stage["sha256"].items():
            if artifact_hash(root/filename) != checksum:
                raise ValueError(f"Recorded output changed: {filename}")
            count += 1
    return count


class StageRunner:
    def __init__(self, root, source, state, source_hashes, versions, minimum_free_gib=5.):
        self.root, self.source, self.state = root, source, state
        self.source_hashes, self.versions = source_hashes, versions
        self.minimum_free_gib = minimum_free_gib
        self.child = None
        self.environment = {**os.environ, **dict.fromkeys(THREAD_ENV, "1"), "PYTHONPATH": str(source),
                            "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}

    def log(self, message):
        line = f"{_now()} {message}"
        print(line, flush=True)
        with (self.root/"runner.log").open("a") as stream:
            stream.write(line+"\n")

    def save(self):
        self.state["updated_utc"] = _now()
        _save(self.root/"status.json", self.state)

    def execute(self, command, log, directory, required):
        with log.open("x") as stream:
            self.child = subprocess.Popen(command, cwd=self.source, env=self.environment, stdout=stream,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            self.state["child_pid"] = self.child.pid
            self.save()
            code = self.child.wait()
        self.child = None
        self.state["child_pid"] = None
        return code

    def run(self, name, arguments, required, *, inputs=(), module="powersim.cli"):
        if package_versions() != self.versions:
            raise ValueError("Package versions changed during the suite")
        for file, checksum in self.source_hashes.items():
            if artifact_hash(self.source/file) != checksum:
                raise ValueError(f"Frozen source changed: {file}")
        input_hashes = {str(Path(path).resolve()): artifact_hash(path) for path in inputs}
        previous = self.state["stages"].get(name, {})
        if previous.get("status") == "complete":
            if previous["input_sha256"] != input_hashes:
                raise ValueError(f"Completed stage inputs changed: {name}")
            for file, checksum in previous["sha256"].items():
                if artifact_hash(self.root/file) != checksum:
                    raise ValueError(f"Completed stage output changed: {name}/{file}")
            self.log(f"REUSE {name}")
            return self.root/previous["directory"]
        if shutil.disk_usage(self.root).free/2**30 < self.minimum_free_gib:
            raise ValueError("Disk reserve reached; stopping without deleting historical artifacts")
        attempt = previous.get("attempt", 0)+1
        directory = self.root/"stages"/name/f"attempt-{attempt:03d}"
        while directory.exists():
            attempt += 1
            directory = self.root/"stages"/name/f"attempt-{attempt:03d}"
        log = self.root/"logs"/f"{name}-attempt-{attempt:03d}.log"
        command = [sys.executable, "-u", "-m", module, *map(str, arguments(directory))]
        entry = {"status": "running", "attempt": attempt, "directory": str(directory.relative_to(self.root)),
                 "command": command, "started_utc": _now(), "input_sha256": input_hashes,
                 "log": str(log.relative_to(self.root))}
        self.state["stages"][name] = entry
        self.state["current_stage"] = name
        self.save()
        self.log(f"START {name}")
        started = time.perf_counter()
        code = self.execute(command, log, directory, required)
        if code:
            raise RuntimeError(f"{name} exited {code}; inspect {log}")
        for filename, checksum in input_hashes.items():
            if artifact_hash(filename) != checksum:
                raise ValueError(f"Stage modified an input artifact: {filename}")
        paths = [directory/filename for filename in required]
        entry.update(status="complete", finished_utc=_now(), seconds=time.perf_counter()-started,
                     sha256={str(path.relative_to(self.root)): artifact_hash(path) for path in paths})
        self.state.update(current_stage=None,
                          completed_stages=sum(s["status"] == "complete" for s in self.state["stages"].values()))
        self.save()
        self.log(f"COMPLETE {name} ({entry['seconds']:.1f}s)")
        return directory

    def stop_child(self):
        if self.child is None:
            return
        try:
            os.killpg(self.child.pid, signal.SIGTERM)
            self.child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(self.child.pid, signal.SIGKILL)
            self.child.wait()
        except ProcessLookupError:
            pass
        self.child = None


def _summarize(root, entries):
    _save(root/"summary.json", {"updated_utc": _now(), "evaluations": entries})
    rows = []
    for entry in entries:
        r = entry["report"]
        rows.append({
            **{k: entry[k] for k in ("benchmark", "seed", "profile", "method", "checkpoint", "split", "mode") if k in entry},
            "cases": r["labeled_cases"], "raw_generator_accuracy": r["raw_accuracy"]["generator_accuracy_lt_1mw"],
            "raw_joint_accuracy": r["raw_accuracy"]["joint_accuracy_lt_1mw"],
            "raw_mae_mw": r["raw_accuracy"]["mae_mw"], "raw_feasible_rate": r["raw_physics"]["feasible_rate"],
            "accepted_generator_accuracy": r["accepted_accuracy"]["generator_accuracy_lt_1mw"],
            "neural_coverage": r["coverage"]["neural_coverage"],
            "hybrid_generator_accuracy": r["hybrid_accuracy"]["generator_accuracy_lt_1mw"],
            "hybrid_feasible_rate": r["hybrid_physics"]["feasible_rate"],
            "fallback_failures": r["fallback_failures"],
            "fixed_target_attack_accuracy": entry.get("attack", {}).get(
                "fixed_original_target_accuracy_all_attacks", {}).get("generator_accuracy_lt_1mw"),
        })
    if rows:
        temporary = root/"results.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(root/"results.csv")


def run_suite(output, config=None, *, resume=False):
    config = config or SuiteConfig()
    plan = suite_plan(config)
    root = Path(output).resolve()
    if resume:
        if not root.is_dir():
            raise ValueError("Cannot resume a missing suite")
    else:
        root.mkdir(parents=True, exist_ok=False)
    lock = root/".suite.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    runner = None
    old_handlers = {}
    state = {"stages": {}}
    try:
        revision, source, archive = _freeze(root)
        source_hashes = {str(path.relative_to(source)): artifact_hash(path) for path in sorted(source.rglob("*"))
                         if path.is_file() and "__pycache__" not in path.parts}
        manifest = {
            "plan": plan, "source_revision": revision, "source_archive_sha256": artifact_hash(archive),
            "source_file_sha256": source_hashes, "runner_sha256": artifact_hash(__file__),
            "versions": package_versions(), "platform": platform.platform(),
        }
        manifest_path = root/"suite_manifest.json"
        if resume:
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Suite source, runner, inputs or configuration changed; resume using its frozen source/environment")
            state = json.loads((root/"status.json").read_text())
        else:
            _save(manifest_path, manifest)
        (root/"logs").mkdir(exist_ok=True)
        state.setdefault("started_utc", _now())
        state.update(status="running", runner_pid=os.getpid(), child_pid=None)
        state.pop("diagnostic", None)
        runner = StageRunner(root, source, state, source_hashes, manifest["versions"], config.minimum_free_gib)
        runner.save()

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")

        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, interrupted)
        configs = root/"configurations"
        data_file = _configuration_file(configs/"dataset.json", plan["dataset_configuration"])
        dataset = runner.run("dataset", lambda out: [
            "dataset", out, "--config", data_file, "--workers", config.workers],
            ["manifest.json", "dataset.npz", "network.json", "train.jsonl", "validation.jsonl",
             *[f"{split}.jsonl" for split in SPLITS]], inputs=[data_file])
        data_inputs = [dataset/name for name in ("manifest.json", "dataset.npz", "network.json")]
        models, decoders, training_rows = {}, {}, []
        for profile in PROFILES:
            for method in METHODS:
                train_file = _configuration_file(configs/f"{profile}--{method}.json", plan["training"][profile][method])
                parent = models[(profile, "M1")]/"model.pt" if method != "M1" else None
                directory = runner.run(f"{profile}--{method}--train", lambda out, parent=parent: [
                    "train", dataset, out, "--config", train_file,
                    *(["--initialize-from", parent] if parent is not None else [])],
                    ["model.pt", "best.pt", "last.pt", "training.json"],
                    inputs=[*data_inputs, train_file, *([parent] if parent is not None else [])])
                models[(profile, method)] = directory
                report = json.loads((directory/"training.json").read_text())
                training_rows.append({"profile": profile, "method": method,
                                      "directory": str(directory.relative_to(root)),
                                      **{k: v for k, v in report.items() if k != "history"}})
                _save(root/"training_summary.json", {"training": training_rows, "held_out_evaluated": False})
                decoder_file = _configuration_file(configs/f"{profile}--decoder.json", plan["decoders"][profile])
                decoder = runner.run(f"{profile}--{method}--decoder", lambda out: [
                    "decoder", dataset, directory/"model.pt", out, "--config", decoder_file],
                    ["decoder.pt", "decoder-best.pt", "decoder-last.pt", "decoder.json"],
                    inputs=[*data_inputs, directory/"model.pt", decoder_file])
                decoders[(profile, method)] = decoder
        primary_m5 = models[("source-informed", "M5")]/"model.pt"
        calibrated = runner.run("source-informed--M5--q95-calibration", lambda out: [
            "calibrate", dataset, primary_m5, decoders[("source-informed", "M5")]/"decoder.pt", out,
            "--error-units", "rmse_mw", "--quantile", .95],
            ["decoder.pt", "decoder.json"],
            inputs=[*data_inputs, primary_m5, decoders[("source-informed", "M5")]/"decoder.pt"])
        entries = []

        def evaluate(profile, method, split, mode, *, checkpoint="last", gate=None, data=dataset, attack=None):
            model = models[(profile, method)]/("model.pt" if checkpoint == "last" else "best.pt")
            name = f"{profile}--{method}--{checkpoint}--{split}--{mode}"
            directory = runner.run(name, lambda out: [
                "evaluate", data, model, out, "--split", split, "--workers", config.workers,
                "--prediction-format", "jsonl.gz",
                *(["--decoder", gate, "--decoder-norm-sensitivity"] if gate is not None else [])],
                ["evaluation.json", "predictions.npz", "predictions.jsonl.gz"],
                inputs=[data/"manifest.json", data/"dataset.npz", model, *([gate] if gate is not None else [])])
            entry = {"profile": profile, "method": method, "checkpoint": checkpoint,
                     "split": split, "mode": mode, "directory": str(directory.relative_to(root)),
                     "report": json.loads((directory/"evaluation.json").read_text())}
            if attack is not None:
                entry["attack"] = attack
            entries.append(entry)
            _summarize(root, entries)

        for profile in PROFILES:
            for method in METHODS:
                for split in SPLITS:
                    evaluate(profile, method, split, "raw")
                    evaluate(profile, method, split, "hybrid-fixed6-rmse",
                             gate=decoders[(profile, method)]/"decoder.pt")
        for method in METHODS:
            for split in SPLITS:
                evaluate("source-informed", method, split, "raw", checkpoint="best_validation")
        for split in SPLITS:
            evaluate("source-informed", "M5", split, "hybrid-q95-rmse", gate=calibrated/"decoder.pt")
        for profile in PROFILES:
            for method in METHODS:
                model = models[(profile, method)]/"model.pt"
                for source_split, attack_split in (("iid_test", "attack1"), ("ood_test_1", "attack2")):
                    attacked = runner.run(f"{profile}--{method}--{attack_split}--dataset", lambda out: [
                        dataset, model, out, "--split", source_split, "--epsilon-mw", 1.,
                        "--workers", config.workers],
                        ["manifest.json", "dataset.npz", "network.json", "attack.json", f"{attack_split}.jsonl"],
                        inputs=[*data_inputs, model], module="powersim.research.attacks")
                    attack = json.loads((attacked/"attack.json").read_text())
                    evaluate(profile, method, attack_split, "raw-fgsm", data=attacked, attack=attack)
        if len(entries) != plan["counts"]["evaluations"] or len(state["stages"]) != plan["counts"]["stages"]:
            raise ValueError("Executed workload does not match the frozen plan")
        verified = verify_suite(root, state)
        state.update(status="complete", finished_utc=_now(), current_stage=None, child_pid=None,
                     verified_output_hashes=verified, completed_evaluations=len(entries))
        runner.save()
        runner.log(f"SUITE COMPLETE: {len(entries)} evaluations, {verified} output hashes verified")
        return {"root": str(root), "source_revision": revision, "stages": len(state["stages"]),
                "evaluations": len(entries), "verified_output_hashes": verified}
    except BaseException as exc:
        if runner is None and resume:
            # An incompatible resume must not replace an existing status/history.
            raise
        if runner is not None:
            runner.stop_child()
        current = state["stages"].get(state.get("current_stage"))
        if current is not None and current["status"] == "running":
            current.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     child_pid=None, diagnostic=f"{type(exc).__name__}: {exc}", updated_utc=_now())
        _save(root/"status.json", state)
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--smoke", action="store_true", help="Tiny full-workflow plumbing check, not a scientific comparison")
    parser.add_argument("--plan", action="store_true", help="Print settings without writing or executing")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = SuiteConfig.smoke() if args.smoke else SuiteConfig()
    report = suite_plan(config) if args.plan else run_suite(args.output, config, resume=args.resume)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
