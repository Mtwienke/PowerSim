from dataclasses import replace
import json

import numpy as np
import pytest

from powersim.config import SimulationConfig
from powersim.data import DatasetConfig, generate_dataset, installed_capacity, load_dataset


def test_total_capacity_hypothesis_matches_printed_percentages_not_demand():
    config = DatasetConfig(penetration_basis="total_installed",
                           renewable_capacities_mw=(185., 230., 300.))
    for rate, capacity in zip(config.penetration_rates, config.renewable_capacities_mw):
        value, actual = installed_capacity(config, rate, 189.2, 335., capacity)
        assert value == capacity
        assert round(100*actual, 2) == round(100*rate, 2)
        inferred, _ = installed_capacity(config, rate, 189.2, 335.)
        assert inferred == pytest.approx(335*rate/(1-rate))
    with pytest.raises(ValueError, match="printed regime"):
        installed_capacity(config, .3558, 189.2, 335., 67.31736)
    assert installed_capacity(DatasetConfig(), .3558, 189.2, 335.)[0] == pytest.approx(67.31736)


@pytest.mark.parametrize("kwargs", [
    {"penetration_basis": "unknown"},
    {"penetration_basis": "total_installed", "penetration_rates": (1.,)},
    {"renewable_capacities_mw": (185.,)},
    {"renewable_capacities_mw": (185., np.inf, 300.)},
    {"sampling_policy": "hidden_resample"},
    {"max_attempt_factor": 0},
    {"test_seed": -1},
    {"placement_seed": True},
])
def test_new_sampling_and_capacity_settings_are_validated(kwargs):
    with pytest.raises(ValueError):
        DatasetConfig(**kwargs)


def test_valid_count_replaces_failures_but_records_every_attempt(tmp_path, monkeypatch):
    import powersim.instance_generation.data as module
    from powersim.solver import SolveResult

    original = module.solve_reference
    calls = 0

    def first_failure(net, config):
        nonlocal calls
        calls += 1
        if calls == 1:
            return SolveResult("failed", False, "dc", "opf", .01, ("controlled failure",))
        return original(net, config)

    monkeypatch.setattr(module, "solve_reference", first_failure)
    config = DatasetConfig(train_cases=4, validation_cases=2, test_cases=2, sampling_policy="valid")
    manifest = generate_dataset(tmp_path/"data", config)
    _, arrays = load_dataset(tmp_path/"data")
    rows = [json.loads(line) for line in (tmp_path/"data/train.jsonl").read_text().splitlines()]
    assert manifest["splits"]["train"]["valid"] == len(arrays["train_x"]) == 4
    assert manifest["splits"]["train"]["requested"] == len(rows) >= 5
    assert manifest["splits"]["train"]["target_valid_cases"] == 4
    assert rows[0]["reference"]["status"] == "failed"
    assert rows[0]["scenario_id"] not in arrays["train_scenario_ids"]
    assert arrays["branch_limits_mw"].shape == (len(manifest["branch_ids"]),)


def test_attempt_cap_is_finite_and_preserves_failure_diagnostics(tmp_path, monkeypatch):
    from powersim.solver import SolveResult
    monkeypatch.setattr("powersim.instance_generation.data.solve_reference",
                        lambda *args: SolveResult("failed", False, "dc", "opf", .01, ("always fails",)))
    config = DatasetConfig(train_cases=2, validation_cases=1, test_cases=1,
                           sampling_policy="valid", max_attempt_factor=2)
    directory = tmp_path/"capped"
    with pytest.raises(ValueError, match="attempt cap exhausted"):
        generate_dataset(directory, config)
    assert len((directory/"train.jsonl").read_text().splitlines()) == 4
    assert "attempt cap" in json.loads((directory/"generation_error.json").read_text())["diagnostic"]
    assert not (directory/"dataset.npz").exists()


def test_fresh_test_streams_leave_train_and_validation_and_sites_unchanged(tmp_path):
    config = DatasetConfig(simulation=SimulationConfig(branch_limit_scale=1.),
                           train_cases=4, validation_cases=2, test_cases=2, sampling_policy="valid",
                           penetration_basis="total_installed", renewable_capacities_mw=(185., 230., 300.))
    first = generate_dataset(tmp_path/"old-streams", config)
    second = generate_dataset(tmp_path/"new-streams", replace(config, test_seed=20281119))
    _, a = load_dataset(tmp_path/"old-streams")
    _, b = load_dataset(tmp_path/"new-streams")
    for key in a:
        if key.startswith(("train_", "validation_")) or key in ("x_mean", "x_scale", "y_mean", "y_scale"):
            np.testing.assert_array_equal(a[key], b[key])
    assert first["installations"] == second["installations"]
    assert not np.array_equal(a["iid_test_x"], b["iid_test_x"])
    assert second["capacity_regimes"][0]["renewable_capacity_mw"] == 185.


def test_valid_sampling_is_worker_count_invariant(tmp_path):
    config = DatasetConfig(train_cases=4, validation_cases=2, test_cases=2, sampling_policy="valid")
    generate_dataset(tmp_path/"one", config, workers=1)
    generate_dataset(tmp_path/"two", config, workers=2)
    _, a = load_dataset(tmp_path/"one")
    _, b = load_dataset(tmp_path/"two")
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
