import gzip
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from powersim.instance_generation.data import DatasetConfig, generate_dataset, load_dataset, split_record_file
from powersim.output_generation.report_inputs import Inventory
from powersim.research.clean_suite import BENCHMARKS, CleanConfig, data_config, main, plan
from powersim.solvers.gao_double_neural.ml import (
    OPFNetwork, active_branch_loss, chain_jacobian, masked_residual_loss, physical_jacobian,
)


def test_full_plan_fresh_rates_workers_and_demand_shift(tmp_path, capsys):
    output = tmp_path/"planned"
    assert main([str(output), "--plan"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert not output.exists()
    assert list(data["networks"]) == list(BENCHMARKS)
    assert data["configuration"]["workers"] == 4
    assert len(set(data["seeds"])) == 3
    assert data["networks"]["case57"]["full"]["penetration_rates"] == [.3428, .3951, .4596, .5031]
    assert data["networks"]["case118"]["full"]["penetration_rates"] == [.3211, .3391, .3878, .4426]
    assert data["networks"]["case300"]["full"]["preserve_fixed_injections"]
    assert data["networks"]["germany30"]["full"]["simulation"]["branch_limit_scale"] == .9
    assert data["networks"]["case1354pegase"]["held_out"][-1] == "ood_demand"
    assert data_config(CleanConfig(), "case30").test_seed != data_config(CleanConfig(), "case30", probe=True).test_seed
    assert plan(CleanConfig(smoke=True))["networks"]["case30"]["full"]["test_seed"] != data["configuration"]["test_seed"]
    assert {row["training"]["M5"]["cpu_threads"] for row in data["settings"].values()} == {4}


@pytest.mark.parametrize("activation", ["relu", "tanh", "silu"])
def test_chain_jacobian_matches_autograd_values_and_weight_gradients(activation):
    torch.manual_seed(2)
    model = OPFNetwork([0, 2, 0], [1, 2, 3], [4, 0], [2, 4], hidden=(4, 5, 3), activation=activation)
    x = torch.tensor([[1., 20., 3.], [2., 2., 4.]], requires_grad=True)
    first, second = physical_jacobian(model, x), chain_jacobian(model, x)
    torch.testing.assert_close(second, first)
    a = torch.autograd.grad(first.square().sum(), model.parameters(), allow_unused=True)
    b = torch.autograd.grad(second.square().sum(), model.parameters(), allow_unused=True)
    for p, left, right in zip(model.parameters(), a, b):
        torch.testing.assert_close(left if left is not None else torch.zeros_like(p),
                                   right if right is not None else torch.zeros_like(p))


@pytest.mark.parametrize("reduction", ["active", "full"])
@pytest.mark.parametrize("norm", ["mse", "mae"])
def test_active_branch_reduction_matches_dense_formula(reduction, norm):
    torch.manual_seed(3)
    j = torch.randn(3, 2, 4, requires_grad=True)
    mg, md = torch.randn(5, 2), torch.randn(5, 4)
    active = torch.tensor([[False]*5, [True, True, False, True, False], [True, False, False, False, False]])
    dense = masked_residual_loss(torch.einsum("bg,ngd->nbd", mg, j)-md, active, norm, reduction)
    sparse = active_branch_loss(j, active, mg, md, norm, reduction)
    torch.testing.assert_close(sparse, dense)
    torch.testing.assert_close(torch.autograd.grad(sparse, j)[0], torch.autograd.grad(dense, j)[0])
    assert active_branch_loss(j, ~torch.ones_like(active), mg, md, norm, reduction).item() == 0.


def test_compressed_demand_split_retains_labels_and_skips_test_sensitivities(tmp_path):
    root = tmp_path/"data"
    config = DatasetConfig(train_cases=3, validation_cases=2, test_cases=3, seed=2026,
                           demand_shift_mean_factor=1.10, training_sensitivities_only=True, record_format="jsonl.gz")
    manifest = generate_dataset(root, config)
    iid, demand = (manifest["splits"][s] for s in ("iid_test", "ood_demand"))
    assert iid["renewable_capacity_mw"] == demand["renewable_capacity_mw"]
    assert iid["load_mean_factor"] == 1 and demand["load_mean_factor"] == 1.10
    assert iid["seed_spawn_key"] != demand["seed_spawn_key"]
    path = split_record_file(root, "ood_demand", manifest)
    with gzip.open(path, "rt") as f:
        assert len(list(f)) == 3
    _, with_j = load_dataset(root)
    assert with_j["train_jacobian"].shape[0] == manifest["splits"]["train"]["valid"]
    assert with_j["ood_demand_jacobian"].shape[0] == 0
    _, without_j = load_dataset(root, sensitivities=False)
    assert "train_jacobian" not in without_j
    np.testing.assert_array_equal(without_j["ood_demand_y"], with_j["ood_demand_y"])


def test_archived_absolute_suite_paths_are_remapped(tmp_path):
    root = tmp_path/"archive"/"full-suite-20261118T044204Z"
    root.mkdir(parents=True)
    inv = Inventory(tmp_path, [])
    assert inv.remap("/previous/artifacts/full-suite-20261118T044204Z/stages/a") == root/"stages/a"
