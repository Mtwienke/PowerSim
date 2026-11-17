from dataclasses import replace
import json

import numpy as np
import pytest

from powersim.config import SimulationConfig
from powersim.contracts import DispatchDecision
from powersim.data import DatasetConfig, RenewableConfig, generate_dataset, load_dataset, valid_label, wind_capacity_factor
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
        def called(net):
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
    r = solve_reference(make_case(c), c)
    assert r.status == "ok", r.diagnostics
    assert r.metrics["feasible"]
    assert "ext_grid:0" in r.generation_mw
    assert r.objective > 0
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


def test_finite_predictions_and_label_rejection():
    net = make_case()
    inputs = build_dispatch_inputs(net)
    p = {g.asset_id: 0.0 for g in inputs["generators"]}
    p["ext_grid:0"] = float("nan")
    assert evaluate_decision(net, DispatchDecision(active_power_mw=p)).status == "failed"
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


def test_dataset_reproducibility_splits_and_normalization(tmp_path):
    c = DatasetConfig(train_cases=6, validation_cases=2, test_cases=2)
    m = generate_dataset(tmp_path / "a", c)
    generate_dataset(tmp_path / "b", c)
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


def test_conversion_and_nett_load_no_double_count():
    config = RenewableConfig()
    np.testing.assert_allclose(wind_capacity_factor([0, 3, 12, 24, 25], config), [0, 0, 1, 1, 0])
    net = make_case()
    import pandapower as pp
    pp.create_sgen(net, 3, p_mw=4)
    target = np.asarray(nodal_loads(net))
    apply_nodal_loads(net, target)
    np.testing.assert_allclose(nodal_loads(net), target)
