from dataclasses import replace
import json

import numpy as np
import pytest

from powersim.config import SimulationConfig
from powersim.contracts import DispatchDecision
from powersim.data import (DatasetConfig, RenewableConfig, generate_dataset, load_dataset, load_dataset_instance,
                           valid_label, wind_capacity_factor)
from powersim.dc import dc_relation, reference_jacobian
from powersim.network import apply_nodal_loads, build_dispatch_inputs, make_case, network_fingerprint, nodal_loads, observation
from powersim.solver import evaluate_decision, solve_reference


def test_defaults_and_validation():
    assert SimulationConfig().physics_mode == "dc"
    assert len(make_case().bus) == 30
    assert len(make_case(SimulationConfig(benchmark="case14")).bus) == 14
    for values in ({"physics_mode": "bad"}, {"method": "bad"}, {"benchmark": "bad"},
                   {"physics_mode": "ac", "method": "learned"}, {"branch_limit_scale": float("nan")}):
        with pytest.raises(ValueError):
            SimulationConfig(**values)


@pytest.mark.parametrize("physics,opf,pf", [("dc", "rundcopp", "rundcpp"), ("ac", "runopp", "runpp")])
def test_separate_routing(monkeypatch, physics, opf, pf):
    import powersim.solver as module
    calls = []
    def stop(name):
        def called(net, **kwargs):
            calls.append(name)
            raise RuntimeError("routing probe")
        return called
    for name in ("rundcopp", "rundcpp", "runopp", "runpp"):
        monkeypatch.setattr(module.pp, name, stop(name))
    net = make_case()
    c = SimulationConfig(physics_mode=physics)
    solve_reference(net, c)
    assert calls == [opf]
    calls.clear()
    evaluate_decision(net, DispatchDecision(active_power_mw={g.asset_id: 0.0 for g in build_dispatch_inputs(net)["generators"]}), c)
    assert calls == [pf]


def test_contract_costs_order_and_structural_identity():
    net = make_case()
    inputs = build_dispatch_inputs(net)
    gens = inputs["generators"]
    assert gens[0].asset_id == "ext_grid:0"
    assert gens[0].cost_quadratic == .02
    assert len({g.marginal_cost_per_mwh for g in gens}) > 1
    obs = observation(net)
    assert obs.bus_ids == tuple(f"bus:{i}" for i in range(30))
    assert sum(obs.net_load_mw) == pytest.approx(obs.total_demand_mw)
    fp = network_fingerprint(net)
    net.load.p_mw *= 1.1
    assert network_fingerprint(net) == fp
    assert observation(net).scenario_id != obs.scenario_id
    net.poly_cost.at[0, "cp2_eur_per_mw2"] *= 2
    assert network_fingerprint(net) != fp
    net = make_case(SimulationConfig(benchmark="case14"))
    fp = network_fingerprint(net)
    net.trafo.at[0, "tap_pos"] += 1
    assert network_fingerprint(net) != fp


@pytest.mark.parametrize("physics,benchmark", [("dc", "case30"), ("dc", "case14"), ("ac", "case30"), ("ac", "case14")])
def test_real_reference(physics, benchmark):
    c = SimulationConfig(physics, benchmark)
    net = make_case(c)
    r = solve_reference(net, c)
    assert r.status == "ok", r.diagnostics
    assert r.metrics["feasible"]
    assert "ext_grid:0" in r.generation_mw
    assert r.objective > 0
    assert r.objective == pytest.approx(net.res_cost)
    assert r.model_id and r.configuration_id and r.input_id
    json.dumps(r.to_dict(), allow_nan=False)
    if physics == "dc":
        assert r.bus_voltage_pu is None
        assert r.metrics["voltage_violations"] is None
        assert r.metrics["reactive_violations"] is None


def test_slack_does_not_hide_raw_prediction_error():
    net = make_case()
    r = solve_reference(net)
    p = dict(r.generation_mw)
    p["ext_grid:0"] += 10
    e = evaluate_decision(net, DispatchDecision(active_power_mw=p))
    assert e.status == "ok"
    assert e.raw_balance_residual_mw == pytest.approx(10)
    assert e.slack_adjustment_mw["ext_grid:0"] == pytest.approx(-10)
    assert not e.metrics["feasible"]
    assert e.proposed_generation_mw == p


def test_ac_objective_includes_actual_reactive_quadratic_costs():
    config = SimulationConfig(physics_mode="ac")
    net = make_case(config)
    net.poly_cost.loc[:, "cq2_eur_per_mvar2"] = .01
    reference = solve_reference(net, config)
    assert reference.status == "ok", reference.diagnostics
    assert reference.objective == pytest.approx(net.res_cost)
    inputs = build_dispatch_inputs(net, config)
    targets = {g.asset_id: float(net.res_bus.at[int(g.bus_id.split(":")[1]), "vm_pu"])
               for g in inputs["generators"]}
    evaluation = evaluate_decision(net, DispatchDecision(active_power_mw=reference.generation_mw,
                                                         voltage_targets_pu=targets), config)
    assert evaluation.status == "ok"
    assert evaluation.objective == pytest.approx(reference.objective, rel=1e-6)


def test_finite_predictions_and_label_rejection():
    net = make_case()
    inputs = build_dispatch_inputs(net)
    p = {g.asset_id: 0.0 for g in inputs["generators"]}
    p["ext_grid:0"] = float("nan")
    failed = evaluate_decision(net, DispatchDecision(active_power_mw=p))
    assert failed.status == "failed"
    json.dumps(failed.to_dict(), allow_nan=False)
    r = solve_reference(net)
    ids = [b.asset_id for b in r.branches]
    for broken in (replace(r, status="failed"), replace(r, objective=float("nan")),
                   replace(r, generation_mw={}), replace(r, branches=()),
                   replace(r, metrics={"feasible": False})):
        assert not valid_label(broken, inputs["generators"], ids)


def test_dc_transformer_limits_and_relation():
    c = SimulationConfig(benchmark="case14", branch_limit_scale=.8)
    net = make_case(c)
    r = solve_reference(net, c)
    relation = dc_relation(net)
    p = np.array(list(r.generation_mw.values()))
    assert any(b.asset_id.startswith("trafo:") for b in r.branches)
    np.testing.assert_allclose(relation.m_g @ p-relation.m_d @ nodal_loads(net)+relation.constant,
                               [b.p_from_mw for b in r.branches], atol=1e-6)
    assert all(b.limit_percent == 80 for b in r.branches)
    assert relation.branch_ids == tuple(b.asset_id for b in r.branches)
    assert relation.branch_limits_mw == tuple(b.limit_mw for b in r.branches)


def test_transformer_pf_violation_respects_configured_limit():
    c = SimulationConfig(benchmark="case14")
    net = make_case(c)
    r = solve_reference(net, c)
    # Set a tight transformer limit, but evaluate the UNCHANGED decision with PF.
    net.trafo.at[2, "max_loading_percent"] = .1
    result = evaluate_decision(net, DispatchDecision(active_power_mw=r.generation_mw), c)
    assert result.status == "ok"
    transformer = next(b for b in result.branches if b.asset_id == "trafo:2")
    assert transformer.limit_percent == .1
    assert transformer.violated
    assert not result.metrics["feasible"]


def test_invalid_converged_reference_is_not_a_label(monkeypatch):
    import powersim.solver as module
    real = module.pp.rundcopp
    def corrupt(net, **kwargs):
        real(net, **kwargs)
        net.res_gen.at[0, "p_mw"] = float("nan")
    monkeypatch.setattr(module.pp, "rundcopp", corrupt)
    result = solve_reference(make_case())
    assert result.status == "failed"
    assert "Non-finite" in result.diagnostics[0]


@pytest.mark.parametrize("unsupported", ["impedance", "controllable_load"])
def test_unsupported_devices_or_optimized_inputs_fail_explicitly(unsupported):
    import pandapower as pp
    net = make_case()
    if unsupported == "impedance":
        pp.create_impedance(net, 0, 1, rft_pu=.01, xft_pu=.1, sn_mva=100)
    else:
        net.load["controllable"] = True
    result = solve_reference(net)
    assert result.status == "failed"
    assert ("Unsupported" if unsupported == "impedance" else "fixed exogenous") in result.diagnostics[0]


def test_failed_samples_are_preserved_but_not_training_labels(tmp_path, monkeypatch):
    import powersim.data as module
    from powersim.solver import SolveResult
    real = module.solve_reference
    calls = []
    def sometimes(net, config):
        calls.append(True)
        if len(calls) == 1:
            return SolveResult("failed", False, "dc", "opf", .01, ("controlled failure",))
        return real(net, config)
    monkeypatch.setattr(module, "solve_reference", sometimes)
    path = tmp_path/"failures"
    manifest = generate_dataset(path, DatasetConfig(train_cases=4, validation_cases=2, test_cases=2))
    records = [json.loads(line) for line in (path/"train.jsonl").read_text().splitlines()]
    assert records[0]["reference"]["status"] == "failed"
    assert manifest["splits"]["train"]["failures"][0]["diagnostics"] == ("controlled failure",)
    _, a = load_dataset(path)
    assert records[0]["scenario_id"] not in a["train_scenario_ids"]
    recovered = load_dataset_instance(path, "train", 0)
    assert recovered.net_load_mw == pytest.approx(records[0]["net_load_mw"])


def test_kkt_sensitivity_matches_finite_differences():
    net = make_case()
    r = solve_reference(net)
    relation = dc_relation(net)
    g = build_dispatch_inputs(net)["generators"]
    j, cond = reference_jacobian([x.cost_quadratic for x in g], relation, r.active_lower,
                                r.active_upper, [b.active for b in r.branches])
    assert cond < 1e10
    np.testing.assert_allclose(j.sum(axis=0), 1, atol=1e-8)
    x = np.asarray(nodal_loads(net))
    # Interior case: perturb an existing load, then compare the optimal vector.
    delta = np.zeros(len(x)); delta[1] = .05
    apply_nodal_loads(net, x + delta)
    r2 = solve_reference(net)
    np.testing.assert_allclose(np.array(list(r2.generation_mw.values())) - list(r.generation_mw.values()),
                               j @ delta, atol=1e-4)


def test_active_generator_mask_and_kkt_row():
    net = make_case()
    net.gen.at[0, "max_p_mw"] = 20
    result = solve_reference(net)
    assert result.status == "ok" and result.active_upper[1]
    relation = dc_relation(net)
    gens = build_dispatch_inputs(net)["generators"]
    j, _ = reference_jacobian([g.cost_quadratic for g in gens], relation,
                              result.active_lower, result.active_upper, [b.active for b in result.branches])
    np.testing.assert_allclose(j[1], 0, atol=1e-9)
    np.testing.assert_allclose(j.sum(axis=0), 1, atol=1e-9)


def test_weakly_active_kkt_is_unavailable():
    from powersim.dc import DCRelation
    relation = DCRelation(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros(1), ("line:0",), 0)
    j, condition = reference_jacobian([.5, .5], relation, [True, False], [False, False], [False],
                                     generation=np.array([0, 1]), cost_linear=np.array([1, 0]))
    assert condition < 1e10 and j is None


def test_dataset_reproducibility_splits_and_normalization(tmp_path):
    c = DatasetConfig(train_cases=6, validation_cases=2, test_cases=2)
    m = generate_dataset(tmp_path / "a", c)
    generate_dataset(tmp_path / "b", c, workers=2)
    _, a = load_dataset(tmp_path / "a"); _, b = load_dataset(tmp_path / "b")
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    np.testing.assert_allclose(a["x_mean"], a["train_x"].mean(axis=0))
    seen = set()
    for split in m["splits"]:
        ids = set(a[f"{split}_scenario_ids"])
        assert not ids & seen
        seen |= ids
    assert m["installations"]
    assert m["versions"]["pandapower"]
    assert m["splits"]["train"]["valid"] > 0
    for split in m["splits"]:
        def records(root):
            values = [json.loads(line) for line in (tmp_path/root/f"{split}.jsonl").read_text().splitlines()]
            for entry in values:
                entry["reference"].pop("solver_seconds")
            return values
        assert records("a") == records("b")


def test_congested_kkt_branch_identity_and_finite_difference(tmp_path):
    directory = tmp_path/"congested"
    config = DatasetConfig(train_cases=12, validation_cases=2, test_cases=2)
    generate_dataset(directory, config)
    _, a = load_dataset(directory)
    candidates = np.flatnonzero(a["train_active_branch"].any(axis=1) & a["train_jacobian_valid"])
    assert len(candidates), "Preset must actually exercise branch constraints"
    i = candidates[0]
    j = a["train_jacobian"][i]
    active = a["train_active_branch"][i]
    np.testing.assert_allclose((a["m_g"] @ j)[active], a["m_d"][active], atol=1e-8)
    net = make_case(config.simulation)
    x = a["train_x"][i].copy()
    delta = np.zeros(len(x)); delta[1] = .01
    apply_nodal_loads(net, x+delta)
    perturbed = solve_reference(net, config.simulation)
    assert perturbed.status == "ok"
    np.testing.assert_array_equal([b.active for b in perturbed.branches], active)
    np.testing.assert_allclose(np.array(list(perturbed.generation_mw.values()))-a["train_y"][i],
                               j @ delta, atol=1e-4)


def test_conversion_and_nett_load_no_double_count():
    config = RenewableConfig()
    np.testing.assert_allclose(wind_capacity_factor([0, 3, 12, 24, 25], config), [0, 0, 1, 1, 0])
    net = make_case()
    import pandapower as pp
    pp.create_sgen(net, 3, p_mw=4)
    target = np.asarray(nodal_loads(net))
    apply_nodal_loads(net, target)
    np.testing.assert_allclose(nodal_loads(net), target)
