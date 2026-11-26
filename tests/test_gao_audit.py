import json

import numpy as np
import pytest

from powersim.research.gao_audit import (
    audit_preprocessing, example_sensitivity_mask, jacobian_loss_snapshot,
    sensitivity_policy_summary, supervised_loss_scales,
)


def test_example_mask_is_signed_max_strict_cutoff_and_nonzero_sum():
    j = np.array([[[9.999, 0]], [[10, 1]], [[-1000, .5]], [[1, -1]], [[0, 0]]])
    before = j.copy()
    np.testing.assert_array_equal(example_sensitivity_mask(j), [True, False, True, False, False])
    np.testing.assert_array_equal(j, before)


@pytest.mark.parametrize("values", [np.zeros((2, 3)), np.zeros((1, 0, 2)), [[[np.nan]]], [[[np.inf]]]])
def test_mask_rejects_malformed_or_nonfinite_arrays(values):
    with pytest.raises(ValueError, match="Finite"):
        example_sensitivity_mask(values)


def test_masking_is_not_sample_deletion_and_reference_energy_is_not_loss():
    reference = np.array([[[1., 0]], [[20, 0]]])
    report = sensitivity_policy_summary(reference, [True, True])
    assert report["supervised_cases_unchanged"] == 2
    assert report["available_jacobians"] == 2
    assert report["example_kept_jacobians"] == 1
    assert report["additionally_masked_jacobians"] == 1
    assert report["additionally_masked_filtered_rows"] == [1]
    assert report["masked_reference_squared_magnitude_share"] == pytest.approx(400/401)
    loss = jacobian_loss_snapshot(np.zeros_like(reference), reference, [True, True])
    assert loss["pilot_L1_available_entry_mean_square"] == pytest.approx(401/4)
    assert loss["example_L1_masked_full_tensor_mean_square"] == pytest.approx(1/4)
    assert loss["pilot_L2_mean_square"] == loss["example_balance_mean_absolute"] == 1


def test_empty_and_unavailable_sensitivities_are_explicit():
    report = sensitivity_policy_summary(np.zeros((0, 2, 3)), [])
    assert report["supervised_cases_unchanged"] == 0
    assert report["masked_reference_squared_magnitude_share"] is None
    report = sensitivity_policy_summary(np.zeros((2, 2, 3)), [False, False])
    assert report["available_jacobians"] == report["additionally_masked_jacobians"] == 0
    loss = jacobian_loss_snapshot(np.zeros((2, 2, 3)), np.zeros((2, 2, 3)), [False, False])
    assert loss["example_L1_masked_full_tensor_mean_square"] == 0
    assert loss["pilot_L1_available_entry_mean_square"] == 0
    with pytest.raises(ValueError, match="availability"):
        sensitivity_policy_summary(np.zeros((2, 2, 3)), [True])
    with pytest.raises(ValueError, match="Matching"):
        jacobian_loss_snapshot(np.zeros((2, 2, 3)), np.zeros((3, 2, 3)), [True, True])


def test_prediction_loss_units_do_not_change_the_predictions():
    prediction, target = np.array([[99., 202.]]), np.array([[100., 200.]])
    before = prediction.copy()
    report = supervised_loss_scales(prediction, target)
    assert report["mse_mw2"] == 2.5
    assert report["mae_mw"] == 1.5
    assert report["mse_per_unit2_on_100mw_base"] == pytest.approx(.00025)
    assert report["example_mae_per_unit_on_100mw_base"] == pytest.approx(.015)
    np.testing.assert_array_equal(prediction, before)
    with pytest.raises(ValueError, match="Matching"):
        supervised_loss_scales([[np.inf]], [[1]])
    with pytest.raises(ValueError, match="Matching"):
        supervised_loss_scales([[1, 2]], [[1]])


@pytest.fixture
def simple_dataset(tmp_path):
    directory = tmp_path/"data"
    directory.mkdir()
    manifest = {
        "configuration": {"simulation": {"benchmark": "case30", "physics_mode": "dc"},
                          "penetration_rates": [.3558, .4071, .4724]},
        "generators": [{"p_max_mw": 335}], "nominal_gross_load_mw": [100, 89.2],
    }
    (directory/"manifest.json").write_text(json.dumps(manifest))
    arrays = {"x_scale": np.ones(2)}
    for split in ("train", "validation"):
        arrays.update({f"{split}_x": [[1., 2.], [3., 4.]],
                       f"{split}_y": [[3.], [7.]],
                       f"{split}_jacobian": [[[1., 1.]], [[20., -19.]]],
                       f"{split}_jacobian_valid": [True, True],
                       f"{split}_active_branch": [[False], [True]]})
    arrays["ood_test_2_y"] = [[np.nan]]  # Not consulted, even for validity.
    np.savez_compressed(directory/"dataset.npz", **arrays)
    return directory


def test_audit_loads_only_train_validation_and_preserves_inputs(simple_dataset, tmp_path, monkeypatch):
    from powersim.research.gao_audit import _sha
    original = np.load
    loaded = []

    class Guard:
        def __init__(self, *args, **kwargs):
            self.archive = original(*args, **kwargs)
            self.files = self.archive.files

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            assert key.startswith(("train_", "validation_")) or key == "x_scale"
            loaded.append(key)
            return self.archive[key]

    monkeypatch.setattr("powersim.research.gao_audit.np.load", Guard)
    before = {name: _sha(simple_dataset/name) for name in ("manifest.json", "dataset.npz")}
    output = tmp_path/"audit"
    report = audit_preprocessing(simple_dataset, output)
    assert report["dataset_sha256"] == before
    assert before == {name: _sha(simple_dataset/name) for name in before}
    assert all(not key.startswith("ood_") for key in loaded)
    assert report["models"] == []
    assert report["splits"]["train"]["additionally_masked_jacobians"] == 1
    rates = report["penetration_hypothesis"]["candidates"]
    assert rates[0]["existing_demand_denominator_capacity_mw"] == pytest.approx(67.31736)
    assert rates[0]["hypothetical_total_capacity_denominator_mw"] == pytest.approx(185.024837)
    assert (output/"audit.json").is_file()
    with pytest.raises(ValueError, match="already exists"):
        audit_preprocessing(simple_dataset, output)


def test_audit_rejects_germany_and_in_place_outputs(simple_dataset, tmp_path):
    with pytest.raises(ValueError, match="separate"):
        audit_preprocessing(simple_dataset, simple_dataset/"audit")
    assert not (simple_dataset/"audit").exists()
    path = simple_dataset/"manifest.json"
    manifest = json.loads(path.read_text())
    manifest["configuration"]["simulation"]["benchmark"] = "germany30"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="IEEE-30 DC"):
        audit_preprocessing(simple_dataset, tmp_path/"rejected")
    assert not (tmp_path/"rejected").exists()
