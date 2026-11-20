from copy import deepcopy
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from powersim.data import DatasetConfig, generate_dataset, load_dataset
from powersim.ml import (
    METHOD_WEIGHTS, Decoder, OPFNetwork, TrainConfig, artifact_hash, branch_flow_penalty,
    jacobian_losses, load_decoder, load_model, reconstruction_errors, supervised_loss,
    tensor, train_decoder, train_model,
)
from powersim.optim import KerasV2Adam


def test_keras_adam_epsilon_hat_decay_and_saved_iterations_match_equations():
    parameter = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
    optimizer = KerasV2Adam([parameter], lr=.001, epsilon=1e-7, decay=.01)
    expected, m, v = parameter.detach().numpy().copy(), np.zeros(2), np.zeros(2)
    gradients = [np.array([1e-9, -.3]), np.array([.2, .1]), np.array([-.1, 0.])]
    for t, gradient in enumerate(gradients, start=1):
        parameter.grad = torch.tensor(gradient)
        optimizer.step()
        m = .9*m+.1*gradient
        v = .999*v+.001*gradient**2
        rate = .001/(1+.01*(t-1))
        expected -= rate*np.sqrt(1-.999**t)/(1-.9**t)*m/(np.sqrt(v)+1e-7)
        np.testing.assert_allclose(parameter.detach().numpy(), expected, rtol=1e-13, atol=1e-13)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(rate)
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = KerasV2Adam([restored_parameter])
    restored.load_state_dict(deepcopy(optimizer.state_dict()))
    assert restored.iterations == optimizer.iterations == 3
    for item in (parameter, restored_parameter):
        item.grad = torch.tensor([.2, -.1], dtype=torch.float64)
    optimizer.step()
    restored.step()
    torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [
    {"normalization": "hidden"},
    {"prediction_loss": "unknown"},
    {"pattern_norm": "unknown"},
    {"power_base_mw": 0},
    {"adam_epsilon": -1},
    {"learning_rate_decay": float("inf")},
    {"sensitivity_policy": "clip"},
    {"mask_reduction": "silently_discard"},
    {"initializer": "unknown"},
    {"adam_variant": "unknown"},
    {"branch_subset": ("line:1", "line:1")},
])
def test_numerical_settings_are_explicit_and_validated(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_model_unit_losses_full_mask_denominator_and_branch_penalty():
    prediction = tensor([[99, 202]])
    target = tensor([[100, 200]])
    model = SimpleNamespace(y_scale=tensor([100, 100]))
    config = TrainConfig(prediction_units="model", prediction_loss="mae")
    assert supervised_loss(prediction, target, model, config) == pytest.approx(.015)
    assert supervised_loss(prediction, target, model, replace(config, prediction_loss="mse")) == pytest.approx(.00025)
    j = tensor([[[1, 1]], [[2, 2]]])
    losses = jacobian_losses(j, torch.zeros_like(j), tensor([1, 0]).bool(),
                             tensor([[1], [0]]).bool(), tensor([[1], [0]]).bool(),
                             tensor([[1]]), tensor([[0, 0]]), pattern_norm="mae", reduction="full")
    assert losses["L1"] == .5
    assert losses["L2"] == .5
    assert losses["L3"] == losses["L4"] == .5
    p = tensor([[5, 0]]).requires_grad_(True)
    penalty = branch_flow_penalty(p, tensor([[0, 0]]), tensor([[1, 0]]), tensor([[0, 0]]),
                                  tensor([1]), tensor([2]))
    assert penalty.item() == pytest.approx(8.)
    penalty.backward()
    assert p.grad[0, 0] == 3
    assert "M3" not in METHOD_WEIGHTS
    assert METHOD_WEIGHTS["M3-example"] == {"L0": 1., "branch_penalty": 1.}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    directory = tmp_path_factory.mktemp("reconstruction-ml")/"data"
    generate_dataset(directory, DatasetConfig(train_cases=5, validation_cases=2, test_cases=2,
                                              sampling_policy="valid"))
    return directory


def source_config(**kwargs):
    return TrainConfig(**{
        "hidden": (8, 5, 3), "activation": "relu", "normalization": "per_unit",
        "prediction_loss": "mae", "prediction_units": "model", "pattern_norm": "mae",
        "mask_reduction": "full", "sensitivity_policy": "example", "initializer": "glorot_uniform",
        "adam_variant": "keras_v2", "adam_epsilon": 1e-7, "learning_rate_decay": 2e-6,
        "batch_size": 2, "epochs": 2, "checkpoint_selection": "last", "perturb_std_normalized": 1.,
        **kwargs,
    })


def test_source_training_uses_per_unit_and_checked_common_parent_without_dropping_l0(dataset, tmp_path):
    defaults = asdict(TrainConfig())
    parent = tmp_path/"M1"
    cfg = source_config()
    train_model(dataset, parent, cfg)
    parent_hash = artifact_hash(parent/"model.pt")
    model, checkpoint = load_model(parent/"model.pt")
    torch.testing.assert_close(model.x_mean, torch.zeros(30))
    torch.testing.assert_close(model.x_scale, torch.full((30,), 100.))
    assert checkpoint["epoch"] == 2
    for method in ("M2", "M4", "M5", "M3-example"):
        report = train_model(dataset, tmp_path/method, replace(cfg, loss_weights=METHOD_WEIGHTS[method]),
                             initialize_from=parent/"model.pt")
        assert report["initialization"]["checkpoint_sha256"] == parent_hash
        assert report["training_accuracy"]["evaluated_scenarios"] == 5
        assert report["configuration"]["normalization"] == "per_unit"
        assert all(np.isfinite(list(row.values())).all() for row in report["history"])
    assert artifact_hash(parent/"model.pt") == parent_hash
    assert asdict(TrainConfig()) == defaults
    with pytest.raises(ValueError, match="normalization"):
        train_model(dataset, tmp_path/"bad-parent", replace(cfg, normalization="standard"),
                    initialize_from=parent/"model.pt")
    with pytest.raises(ValueError, match="branch_subset"):
        train_model(dataset, tmp_path/"bad-branch", replace(cfg, branch_subset=("line:unknown",)))
    assert not (tmp_path/"bad-branch").exists()


def test_example_mask_only_changes_l1_learning_availability(dataset, tmp_path):
    import shutil
    changed = tmp_path/"changed"
    shutil.copytree(dataset, changed)
    _, arrays = load_dataset(changed)
    arrays["train_jacobian"][:] = .1
    arrays["train_jacobian"][0, 0, 0] = 100
    np.savez_compressed(changed/"dataset.npz", **arrays)
    report = train_model(changed, tmp_path/"masked", source_config(loss_weights=METHOD_WEIGHTS["M2"]))
    assert report["sensitivity_learning"]["available"] == 5
    assert report["sensitivity_learning"]["kept_for_L1"] == 4
    assert report["training_accuracy"]["evaluated_scenarios"] == 5
    _, after = load_dataset(changed)
    np.testing.assert_array_equal(after["train_jacobian"], arrays["train_jacobian"])


def test_last_hidden_relu_decoder_has_correct_width_and_preserves_final_and_best(dataset, tmp_path):
    parent = tmp_path/"M1"
    train_model(dataset, parent, source_config())
    before = artifact_hash(parent/"model.pt")
    metadata = train_decoder(dataset, parent/"model.pt", tmp_path/"decoder", epochs=2,
                             hidden=(5, 5), activation="relu", feature_layer="last_hidden",
                             initializer="glorot_uniform", adam_variant="keras_v2",
                             adam_epsilon=1e-7, learning_rate_decay=2e-6,
                             checkpoint_selection="last", units="rmse_mw", threshold=6)
    model, _ = load_model(parent/"model.pt")
    decoder, loaded = load_decoder(tmp_path/"decoder/decoder.pt", parent/"model.pt", model)
    assert decoder.layers[0].in_features == 3
    assert loaded["feature_layer_name"] == metadata["feature_layer_name"] == "last_hidden"
    assert loaded["selected_epoch"] == 2 and loaded["activation"] == "relu"
    report = json.loads((tmp_path/"decoder/decoder.json").read_text())
    assert set(report["checkpoints"]) == {"decoder.pt", "decoder-best.pt", "decoder-last.pt"}
    assert artifact_hash(parent/"model.pt") == before
    assert sum(isinstance(layer, torch.nn.ReLU) for layer in decoder.layers) == 2


def test_reconstruction_l1_linf_are_not_mae_or_rmse():
    model = SimpleNamespace(x_mean=tensor([0, 0]), x_scale=tensor([2, 4]), features=lambda x: x)
    decoder = lambda x: torch.zeros_like(x)
    x = tensor([[2, 8]])
    assert reconstruction_errors(model, decoder, x, "l1_mw") == 10
    assert reconstruction_errors(model, decoder, x, "linf_mw") == 8
    assert reconstruction_errors(model, decoder, x, "mae_mw") == 5


def test_legacy_checkpoint_configuration_remains_loadable(dataset, tmp_path):
    directory = tmp_path/"legacy"
    train_model(dataset, directory, TrainConfig(hidden=(8, 5, 3), epochs=1))
    model, checkpoint = load_model(directory/"model.pt")
    for key in ("normalization", "power_base_mw", "prediction_loss", "prediction_units",
                "pattern_norm", "mask_reduction", "sensitivity_policy", "branch_subset",
                "initializer", "adam_variant", "adam_epsilon", "learning_rate_decay"):
        checkpoint["configuration"].pop(key)
    path = tmp_path/"legacy-config.pt"
    torch.save(checkpoint, path)
    loaded, _ = load_model(path)
    _, arrays = load_dataset(dataset)
    with torch.no_grad():
        torch.testing.assert_close(model(tensor(arrays["validation_x"])),
                                   loaded(tensor(arrays["validation_x"])), rtol=0, atol=0)
