from dataclasses import replace
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from powersim.data import package_versions
from powersim.ml import artifact_hash
from powersim.reproduction import (
    METHODS, PROFILES, StageRunner, SuiteConfig, main, run_suite, suite_plan, verify_suite,
)


def test_frozen_plan_distinguishes_all_hypotheses_and_keeps_germany_out(tmp_path, capsys):
    root = tmp_path/"not-created"
    assert main([str(root), "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert not root.exists()
    assert plan["counts"] == {"datasets": 1, "trainings": 18, "decoders": 18, "calibrations": 1,
                              "attack_datasets": 36, "evaluations": 165, "stages": 239}
    assert plan["dataset_configuration"]["simulation"]["benchmark"] == "case30"
    assert plan["dataset_configuration"]["test_seed"] == 20281119
    assert plan["dataset_configuration"]["renewable_capacities_mw"] == [185, 230, 300]
    training = plan["training"]
    assert training["source-informed"]["M5"]["loss_weights"]["L3"] == 1
    assert training["public-example"]["M5"]["loss_weights"]["L3"] == 0
    assert training["public-example"]["M5"]["branch_subset"] == ["line:20"]
    assert training["journal-equations"]["M5"]["pattern_norm"] == "mse"
    assert training["source-informed"]["M5"]["pattern_norm"] == "mae"
    assert plan["decoders"]["public-example"]["hidden"] == [200, 200, 200]
    assert set(METHODS) == {"M1", "M1-continued", "M2", "M3-example", "M4", "M5"}


def fake_freeze(root):
    source, archive = root/"source", root/"source.tar"
    if not source.exists():
        source.mkdir()
        (source/"pin.txt").write_text("frozen source\n")
        archive.write_bytes(b"test-only archive")
    return "test-frozen-revision", source, archive


class FakeCommands(StageRunner):
    fail_once = None

    def execute(self, command, log, directory, required):
        directory.mkdir(parents=True)
        log.write_text("unit-test command, no subprocess launched\n")
        if self.fail_once and self.fail_once in str(directory):
            FakeCommands.fail_once = None
            (directory/"partial.txt").write_text("preserve failed attempt")
            raise RuntimeError("controlled interruption")
        for filename in required:
            if filename == "training.json":
                value = {"configuration": {}, "training_accuracy": {}, "validation_accuracy": {}, "history": []}
            elif filename == "evaluation.json":
                value = {
                    "labeled_cases": 3, "raw_accuracy": {"generator_accuracy_lt_1mw": .5,
                    "joint_accuracy_lt_1mw": .2, "mae_mw": 1.},
                    "hybrid_accuracy": {"generator_accuracy_lt_1mw": .5},
                    "accepted_accuracy": {"generator_accuracy_lt_1mw": .5},
                    "raw_physics": {"feasible_rate": 0}, "hybrid_physics": {"feasible_rate": 0},
                    "coverage": {"neural_coverage": 1.}, "fallback_failures": 0,
                }
            elif filename == "attack.json":
                value = {"fixed_original_target_accuracy_all_attacks": {"generator_accuracy_lt_1mw": .5}}
            else:
                value = {}
            (directory/filename).write_text(json.dumps(value))
        return 0


def test_full_workload_resume_and_hash_validation_with_no_real_training(tmp_path, monkeypatch):
    import powersim.reproduction as module
    monkeypatch.setattr(module, "_freeze", fake_freeze)
    monkeypatch.setattr(module, "StageRunner", FakeCommands)
    root = tmp_path/"suite"
    config = SuiteConfig.smoke()
    result = run_suite(root, config)
    assert result["stages"] == 239 and result["evaluations"] == 165
    assert result["verified_output_hashes"] == verify_suite(root)
    assert len(json.loads((root/"summary.json").read_text())["evaluations"]) == 165
    assert not (root/".suite.lock").exists()
    before = (root/"status.json").read_bytes()
    with pytest.raises(ValueError, match="configuration changed"):
        run_suite(root, replace(config, test_seed=1234), resume=True)
    assert (root/"status.json").read_bytes() == before
    assert run_suite(root, config, resume=True)["verified_output_hashes"] == result["verified_output_hashes"]
    state = json.loads((root/"status.json").read_text())
    first = next(iter(state["stages"].values()))
    path = root/next(iter(first["sha256"]))
    path.write_text(path.read_text()+"changed")
    with pytest.raises(ValueError, match="output changed"):
        run_suite(root, config, resume=True)
    assert path.read_text().endswith("changed")


def test_failed_attempt_is_preserved_and_restarts_in_new_directory(tmp_path, monkeypatch):
    import powersim.reproduction as module
    monkeypatch.setattr(module, "_freeze", fake_freeze)
    monkeypatch.setattr(module, "StageRunner", FakeCommands)
    FakeCommands.fail_once = "source-informed--M2--train"
    root = tmp_path/"suite"
    with pytest.raises(RuntimeError, match="controlled"):
        run_suite(root, SuiteConfig.smoke())
    partial = next(root.rglob("partial.txt"))
    assert not (root/".suite.lock").exists()
    assert json.loads((root/"status.json").read_text())["status"] == "failed"
    result = run_suite(root, SuiteConfig.smoke(), resume=True)
    assert result["stages"] == 239
    assert partial.read_text() == "preserve failed attempt"
    stage = json.loads((root/"status.json").read_text())["stages"]["source-informed--M2--train"]
    assert stage["attempt"] == 2


def test_stage_rejects_changed_source_before_execution(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    pin = source/"pin.txt"
    pin.write_text("original")
    checksum = artifact_hash(pin)
    root = tmp_path/"run"
    root.mkdir()
    runner = StageRunner(root, source, {"stages": {}}, {"pin.txt": checksum}, package_versions(), 0.)
    pin.write_text("changed")
    with pytest.raises(ValueError, match="Frozen source changed"):
        runner.run("not-started", lambda out: [], [])
    assert not (root/"stages").exists()
