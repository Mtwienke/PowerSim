import pytest

from controllers.my_dispatch import MyDispatchPolicy
from powersim.contracts import DispatchDecision, DispatchFailure
from powersim.simulator import build_dispatch_inputs, evaluate, make_case
from powersim.config import SimulationConfig


def policy_for(response):
    class FixedPolicy:
        def reset(self, context) -> None:
            pass

        def decide(self, observation):
            return response

    return FixedPolicy()


def test_case_runs_through_policy_and_dc_evaluator() -> None:
    result = evaluate(make_case(), MyDispatchPolicy())

    assert result["status"] == "evaluated"
    assert "feasible" in result
    assert result["bus_voltage_pu"] is None
    assert "slack_p_mw" in result


def test_invalid_policy_response_is_rejected() -> None:
    class InvalidPolicy:
        def reset(self, context) -> None:
            pass

        def decide(self, observation):
            return {"status": "not-a-contract-result"}

    result = evaluate(make_case(), InvalidPolicy())  # type: ignore[arg-type]

    assert result["status"] == "proposal_rejected"
    assert "neither DispatchDecision nor DispatchFailure" in result["diagnostic"]


def test_policy_failure_is_reported_without_running_power_flow() -> None:
    result = evaluate(make_case(), policy_for(DispatchFailure("unavailable", "policy cannot dispatch")))

    assert result["status"] == "policy_failure"
    assert result["diagnostic"] == "policy cannot dispatch"


@pytest.mark.parametrize("setpoints", [{}, {"gen:unknown": 1.0}])
def test_decision_rejects_missing_or_unexpected_generator_keys(setpoints) -> None:
    result = evaluate(make_case(), policy_for(DispatchDecision(active_power_mw=setpoints)))

    assert result["status"] == "proposal_rejected"
    assert result["proposal"]["active_power_mw"] == setpoints


def test_solver_exception_is_reported_with_type_and_diagnostic(monkeypatch) -> None:
    import powersim.solver as simulator

    def raise_solver_error(net) -> None:
        raise RuntimeError("controlled AC failure")

    monkeypatch.setattr(simulator.pp, "runpp", raise_solver_error)
    net = make_case()
    inputs = build_dispatch_inputs(net)
    decision = DispatchDecision(active_power_mw={asset_id: 0.0 for asset_id in inputs["context"].controllable_assets})

    result = evaluate(net, policy_for(decision), config=SimulationConfig(physics_mode="ac"))

    assert result["status"] == "power_flow_failed"
    assert result["error_type"] == "RuntimeError"
    assert result["diagnostic"] == "controlled AC failure"


def test_voltage_and_line_violations_mark_evaluation_infeasible(monkeypatch) -> None:
    import powersim.solver as simulator

    real_runpp = simulator.pp.runpp

    def run_with_violations(net) -> None:
        real_runpp(net)
        net.res_bus.loc[:, "vm_pu"] = 1.2
        net.res_line.loc[:, "loading_percent"] = 150.0

    monkeypatch.setattr(simulator.pp, "runpp", run_with_violations)

    result = evaluate(make_case(), MyDispatchPolicy(), config=SimulationConfig(physics_mode="ac"))

    assert result["status"] == "evaluated"
    assert result["feasible"] is False
    assert result["voltage_violations"] > 0
    assert result["line_overloads"] > 0
    assert result["proposal"]["diagnostics"]
