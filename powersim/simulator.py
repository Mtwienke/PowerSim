"""Compatibility policy adapter. Numerical references live in powersim.solver."""
from dataclasses import asdict

from .config import SimulationConfig
from .contracts import DispatchDecision, DispatchFailure
from .network import build_dispatch_inputs, make_case, network_fingerprint, observation
from .solver import evaluate_decision, solve_reference


def evaluate(net, policy, step_id=0, config: SimulationConfig | None = None) -> dict:
    config = config or SimulationConfig(method="heuristic")
    inputs = build_dispatch_inputs(net, config)
    policy.reset(inputs["context"])
    proposal = policy.decide(observation(net, config, step_id))
    if isinstance(proposal, DispatchFailure):
        return {"status": "policy_failure", "diagnostic": proposal.diagnostic, "proposal": asdict(proposal)}
    if not isinstance(proposal, DispatchDecision):
        return {"status": "proposal_rejected", "diagnostic": "Policy returned neither DispatchDecision nor DispatchFailure"}
    payload = {"schema_version": proposal.schema_version, "active_power_mw": dict(proposal.active_power_mw),
               "voltage_targets_pu": dict(proposal.voltage_targets_pu), "diagnostics": list(proposal.diagnostics)}
    if set(proposal.active_power_mw) != set(inputs["context"].controllable_assets):
        return {"status": "proposal_rejected", "diagnostic": "Setpoints must cover exactly controllable generator IDs",
                "proposal": payload}
    result = evaluate_decision(net, proposal, config)
    if result.status != "ok":
        diagnostic = result.diagnostics[0]
        error_type, _, message = diagnostic.partition(": ")
        return {"status": "power_flow_failed", "error_type": error_type, "diagnostic": message, "proposal": payload}
    data = result.to_dict()
    data.update(status="evaluated", feasible=result.metrics["feasible"], proposal=payload,
                executed_setpoints_mw=dict(proposal.active_power_mw),
                realised_generation_mw={int(k.split(":")[1]): v for k, v in result.generation_mw.items() if k.startswith("gen:")},
                slack_p_mw={int(k.split(":")[1]): v for k, v in result.generation_mw.items() if k.startswith("ext_grid:")},
                line_loading_percent={int(b.asset_id.split(":")[1]): b.loading_percent for b in result.branches if b.asset_id.startswith("line:")},
                voltage_violations=result.metrics["voltage_violations"],
                line_overloads=sum(b.violated for b in result.branches if b.asset_id.startswith("line:")),
                physical_solve=f"{config.physics_mode.upper()} power flow")
    if result.bus_voltage_pu is not None:
        data["bus_voltage_pu"] = {int(k.split(":")[1]): v for k, v in result.bus_voltage_pu.items()}
    return data
