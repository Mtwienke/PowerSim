"""Physical adapter. Policies see immutable DTOs, never the pandapower net."""
import hashlib
import json
import logging
import math
from typing import Any, Literal, NotRequired, TypedDict
import pandapower as pp
import pandapower.networks as pn
from .contracts import DispatchContext, DispatchDecision, DispatchFailure, DispatchObservation, DispatchPolicy, Generator

logger = logging.getLogger(__name__)


class DispatchDecisionPayload(TypedDict):
    schema_version: str
    active_power_mw: dict[str, float]
    voltage_targets_pu: dict[str, float]
    diagnostics: list[str]


class DispatchFailurePayload(TypedDict):
    status: str
    diagnostic: str


class PolicyFailureResult(TypedDict):
    status: Literal["policy_failure"]
    diagnostic: str
    proposal: DispatchFailurePayload


class ProposalRejectedResult(TypedDict):
    status: Literal["proposal_rejected"]
    diagnostic: str
    proposal: NotRequired[DispatchDecisionPayload]


class PowerFlowFailedResult(TypedDict):
    status: Literal["power_flow_failed"]
    error_type: str
    diagnostic: str
    proposal: DispatchDecisionPayload


class EvaluatedResult(TypedDict):
    status: Literal["evaluated"]
    feasible: bool
    proposal: DispatchDecisionPayload
    executed_setpoints_mw: dict[str, float]
    realised_generation_mw: dict[int, float]
    bus_voltage_pu: dict[int, float]
    voltage_violations: int
    line_loading_percent: dict[int, float]
    line_overloads: int
    slack_p_mw: dict[int, float]
    physical_solve: str


EvaluationResult = PolicyFailureResult | ProposalRejectedResult | PowerFlowFailedResult | EvaluatedResult


class DispatchInputs(TypedDict):
    context: DispatchContext
    generators: tuple[Generator, ...]
    generator_indices: dict[str, int]

def make_case() -> pp.pandapowerNet:
    return pn.case14()

def network_fingerprint(net: pp.pandapowerNet) -> str:
    payload = {"bus": net.bus.to_dict(), "line": net.line.to_dict(), "gen": net.gen.to_dict()}
    return hashlib.sha256(json.dumps(payload, default=str, sort_keys=True).encode()).hexdigest()

def build_dispatch_inputs(net: pp.pandapowerNet) -> DispatchInputs:
    fp = network_fingerprint(net)
    generators = []
    for index, row in net.gen.iterrows():
        min_p_mw = float(row.min_p_mw)
        max_p_mw = float(row.max_p_mw)
        generators.append(
            Generator(
                asset_id=f"gen:{index}",
                p_min_mw=0.0 if math.isnan(min_p_mw) else min_p_mw,
                p_max_mw=float(row.p_mw) + 100 if math.isnan(max_p_mw) else max_p_mw,
                marginal_cost_per_mwh=1.0,
            )
        )
    gens = tuple(generators)
    ctx = DispatchContext(
        schema_version="1.0",
        network_fingerprint=fp,
        units=(("power", "MW"), ("voltage", "pu"), ("duration", "h")),
        controllable_assets=tuple(generator.asset_id for generator in gens),
        assumptions=("Balanced positive-sequence quasi-static AC power flow",),
    )
    return {"context": ctx, "generators": gens,
            "generator_indices": {f"gen:{idx}": int(idx) for idx in net.gen.index}}


def _decision_payload(decision: DispatchDecision) -> DispatchDecisionPayload:
    """Return a JSON-friendly snapshot of a validated decision DTO."""
    return {
        "schema_version": decision.schema_version,
        "active_power_mw": dict(decision.active_power_mw),
        "voltage_targets_pu": dict(decision.voltage_targets_pu),
        "diagnostics": list(decision.diagnostics),
    }


def _failure_payload(failure: DispatchFailure) -> DispatchFailurePayload:
    """Return a JSON-friendly snapshot of a policy failure DTO."""
    return {"status": failure.status, "diagnostic": failure.diagnostic}

def _validate_and_apply_setpoints(
    net: pp.pandapowerNet,
    inputs: DispatchInputs,
    proposal: DispatchDecision,
) -> ProposalRejectedResult | None:
    """Reject malformed dispatches or apply their validated generator setpoints."""
    expected = set(inputs["context"].controllable_assets)
    if set(proposal.active_power_mw) != expected:
        return {
            "status": "proposal_rejected",
            "diagnostic": "Setpoints must cover exactly controllable generator IDs",
            "proposal": _decision_payload(proposal),
        }
    for asset_id, value in proposal.active_power_mw.items():
        index = inputs["generator_indices"][asset_id]
        net.gen.at[index, "p_mw"] = value
    return None


def _evaluated_result(net: pp.pandapowerNet, proposal: DispatchDecision) -> EvaluatedResult:
    """Assemble metrics after a successful AC power flow."""
    vm = net.res_bus.vm_pu
    loading = net.res_line.loading_percent if len(net.line) else []
    voltage_violations = int(
        ((vm < net.bus.min_vm_pu.fillna(.9)) | (vm > net.bus.max_vm_pu.fillna(1.1))).sum()
    )
    line_overloads = int((loading > 100).sum()) if len(net.line) else 0
    return {
        "status": "evaluated",
        "feasible": voltage_violations == 0 and line_overloads == 0,
        "proposal": _decision_payload(proposal),
        "executed_setpoints_mw": dict(proposal.active_power_mw),
        "realised_generation_mw": net.res_gen.p_mw.to_dict(),
        "bus_voltage_pu": vm.to_dict(),
        "voltage_violations": voltage_violations,
        "line_loading_percent": loading.to_dict() if len(net.line) else {},
        "line_overloads": line_overloads,
        "slack_p_mw": net.res_ext_grid.p_mw.to_dict(),
        "physical_solve": "AC power flow",
    }


def evaluate(net: pp.pandapowerNet, policy: DispatchPolicy, step_id: int = 0) -> EvaluationResult:
    inputs = build_dispatch_inputs(net)
    ctx, gens = inputs["context"], inputs["generators"]
    policy.reset(ctx)
    demand = float(net.load.p_mw.sum())
    obs = DispatchObservation("1.0", ctx.network_fingerprint, "2026-01-01T00:00:00Z", step_id, .25, demand, 0.0, gens)
    proposal = policy.decide(obs)
    if isinstance(proposal, DispatchFailure):
        return {
            "status": "policy_failure",
            "diagnostic": proposal.diagnostic,
            "proposal": _failure_payload(proposal),
        }
    if not isinstance(proposal, DispatchDecision):
        return {"status": "proposal_rejected", "diagnostic": "Policy returned neither DispatchDecision nor DispatchFailure"}
    rejection = _validate_and_apply_setpoints(net, inputs, proposal)
    if rejection is not None:
        return rejection
    try:
        pp.runpp(net)
    except Exception as exc:
        logger.exception("Pandapower AC power flow failed")
        return {"status": "power_flow_failed", "error_type": type(exc).__name__,
                "diagnostic": str(exc), "proposal": _decision_payload(proposal)}
    return _evaluated_result(net, proposal)
