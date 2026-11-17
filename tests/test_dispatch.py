from controllers.my_dispatch import MyDispatchPolicy, choose_dispatch
from powersim.contracts import DispatchContext, DispatchDecision, DispatchFailure, DispatchObservation, Generator


def observation(demand: float, renewable: float, generators: tuple[Generator, ...]) -> DispatchObservation:
    return DispatchObservation(
        schema_version="1.0",
        context_fingerprint="network-test",
        timestamp="2026-01-01T00:00:00Z",
        step_id=0,
        interval_hours=0.25,
        total_demand_mw=demand,
        renewable_available_mw=renewable,
        generators=generators,
    )


def test_dispatch_uses_merit_order_and_renewables_first() -> None:
    generators = (
        Generator("coal", 0.0, 100.0, 20.0),
        Generator("gas", 0.0, 100.0, 40.0),
    )

    decision = choose_dispatch(observation(150.0, 30.0, generators))

    assert isinstance(decision, DispatchDecision)
    assert decision.active_power_mw == {"coal": 100.0, "gas": 20.0}


def test_dispatch_reports_capacity_shortfall() -> None:
    generators = (Generator("small", 0.0, 40.0),)

    result = choose_dispatch(observation(100.0, 10.0, generators))

    assert isinstance(result, DispatchFailure)
    assert result.status == "insufficient_capacity"
    assert "50.000 MW" in result.diagnostic


def test_policy_rejects_decision_before_reset() -> None:
    policy = MyDispatchPolicy()
    result = policy.decide(observation(10.0, 0.0, (Generator("coal", 0.0, 20.0),)))

    assert isinstance(result, DispatchFailure)
    assert result.status == "policy_not_initialized"


def test_policy_rejects_observation_from_different_network() -> None:
    policy = MyDispatchPolicy()
    policy.reset(DispatchContext("1.0", "other-network", (), ("coal",)))

    result = policy.decide(observation(10.0, 0.0, (Generator("coal", 0.0, 20.0),)))

    assert isinstance(result, DispatchFailure)
    assert result.status == "context_mismatch"


def test_policy_rejects_generator_assets_outside_context() -> None:
    policy = MyDispatchPolicy()
    policy.reset(DispatchContext("1.0", "network-test", (), ("gas",)))

    result = policy.decide(observation(10.0, 0.0, (Generator("coal", 0.0, 20.0),)))

    assert isinstance(result, DispatchFailure)
    assert result.status == "asset_mismatch"


def test_dispatch_context_units_are_immutable_and_json_friendly() -> None:
    context = DispatchContext("1.0", "fingerprint", (("power", "MW"),), ("gen:1",))

    assert context.units == (("power", "MW"),)
    try:
        context.units[0] = ("power", "kW")  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("DispatchContext.units unexpectedly allowed mutation")


def test_dispatch_decision_snapshots_mutable_inputs_and_exposes_read_only_mappings() -> None:
    targets = {"gen:1": 12.0}
    diagnostics = ["scheduled"]
    decision = DispatchDecision(active_power_mw=targets, diagnostics=diagnostics)

    targets["gen:1"] = 99.0
    diagnostics.append("changed")

    assert decision.active_power_mw == {"gen:1": 12.0}
    assert decision.diagnostics == ("scheduled",)
    try:
        decision.active_power_mw["gen:1"] = 50.0  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("DispatchDecision.active_power_mw unexpectedly allowed mutation")
