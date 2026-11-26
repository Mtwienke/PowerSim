from dataclasses import replace
import json

import pandapower as pp
import pytest

from powersim.config import SimulationConfig
from powersim.contracts import DispatchDecision, DispatchFailure
from powersim.problem import OPFInstance
from powersim.solver import solve_reference


@pytest.fixture
def instance():
    net = pp.create_empty_network(sn_mva=100)
    source = pp.create_bus(net, vn_kv=110)
    demand = pp.create_bus(net, vn_kv=110)
    pp.create_line_from_parameters(net, source, demand, 1, .1, .2, 0, .5, max_loading_percent=100)
    ext_grid = pp.create_ext_grid(net, source, min_p_mw=0, max_p_mw=80)
    gen = pp.create_gen(net, demand, p_mw=0, min_p_mw=0, max_p_mw=80)
    pp.create_poly_cost(net, ext_grid, "ext_grid", 0, cp2_eur_per_mw2=1)
    pp.create_poly_cost(net, gen, "gen", 0, cp2_eur_per_mw2=3)
    pp.create_load(net, demand, p_mw=40)
    snapshot = OPFInstance(net)
    net.load.loc[:, "p_mw"] = 150
    return snapshot


def test_reuse_one_instance_with_independent_solver(instance, monkeypatch):
    reference = solve_reference(instance)
    assert reference.status == "ok", reference.diagnostics

    def unexpected(*args, **kwargs):
        pytest.fail("A custom solver must not invoke pandapower OPF")

    monkeypatch.setattr(pp, "rundcopp", unexpected)

    def economic_dispatch(case):
        assert case is instance
        scratch = case.copy_network()
        assert scratch.load.p_mw.iloc[0] == 40
        assert not scratch.OPF_converged
        scratch.load.loc[:, "p_mw"] = 100
        # Closed-form optimum for these costs, assuming no bounds bind.
        inverse_costs = {g.asset_id: 1/g.cost_quadratic for g in case.generators}
        factor = sum(case.net_load_mw) / sum(inverse_costs.values())
        return DispatchDecision(active_power_mw={k: v*factor for k, v in inverse_costs.items()})

    for _ in range(2):
        result = solve_reference(instance, solver=economic_dispatch)
        assert result.status == "ok", result.diagnostics
        assert result.operation == "opf"
        assert result.to_dict().keys() == reference.to_dict().keys()
        assert result.generation_mw == pytest.approx(reference.generation_mw)
        assert result.objective == pytest.approx(reference.objective)
        assert [b.p_from_mw for b in result.branches] == pytest.approx([b.p_from_mw for b in reference.branches])
        assert (result.model_id, result.configuration_id, result.input_id) == (
            reference.model_id, reference.configuration_id, reference.input_id)
        json.dumps(result.to_dict(), allow_nan=False)


def test_custom_solver_errors_and_infeasible_dispatch(instance):
    failed = solve_reference(instance, solver=lambda _: DispatchFailure("infeasible", "No solution found"))
    assert failed.status == "failed" and not failed.converged
    assert failed.diagnostics == ("infeasible: No solution found",)

    incomplete = solve_reference(instance, solver=lambda _: DispatchDecision(active_power_mw={"gen:0": 10.}))
    assert incomplete.status == "failed" and incomplete.operation == "opf"
    assert "exactly controllable generator IDs" in incomplete.diagnostics[0]

    proposal = DispatchDecision(active_power_mw={"gen:0": 10., "ext_grid:0": 35.}, diagnostics=("test",))
    invalid = solve_reference(instance, solver=lambda _: proposal)
    assert invalid.status == "invalid" and invalid.converged
    assert invalid.diagnostics[-1] == "test"
    assert invalid.proposed_generation_mw == dict(proposal.active_power_mw)
    assert invalid.slack_adjustment_mw == pytest.approx({"ext_grid:0": -5.})
    assert not invalid.metrics["feasible"]

    wrong_config = replace(SimulationConfig(), branch_limit_scale=.8)
    mismatch = solve_reference(instance, wrong_config)
    assert mismatch.status == "failed"
    assert "Configuration must match" in mismatch.diagnostics[0]
