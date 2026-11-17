"""OPF labels and independent PF evaluation. No optimizer is called by evaluate_decision."""
from dataclasses import asdict, dataclass, field
import math
import time
from typing import Literal

import numpy as np
import pandapower as pp
from pandapower.pypower.idx_brch import PF, PT, RATE_A
from pandapower.pypower.idx_bus import GS

from .config import SimulationConfig
from .contracts import DispatchDecision
from .network import build_dispatch_inputs, digest, nodal_loads, ordered_buses

# All active masks are local, tolerance-based, and aligned to explicit IDs.
BOUND_TOL_MW = 1e-3
BALANCE_TOL_MW = 1e-4
LOADING_TOL_PERCENT = 1e-3
VOLTAGE_TOL_PU = 1e-5


@dataclass(frozen=True)
class BranchResult:
    asset_id: str
    p_from_mw: float
    p_to_mw: float
    limit_mw: float | None
    loading_percent: float
    limit_percent: float
    active: bool
    violated: bool


@dataclass(frozen=True)
class SolveResult:
    status: Literal["ok", "failed", "invalid"]
    converged: bool
    physics_mode: str
    operation: str
    solver_seconds: float
    diagnostics: tuple[str, ...] = ()
    generation_mw: dict[str, float] = field(default_factory=dict)
    objective: float | None = None
    branches: tuple[BranchResult, ...] = ()
    active_lower: tuple[bool, ...] = ()
    active_upper: tuple[bool, ...] = ()
    proposed_generation_mw: dict[str, float] | None = None
    raw_balance_residual_mw: float | None = None
    slack_adjustment_mw: dict[str, float] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    bus_voltage_pu: dict[str, float] | None = None
    reactive_generation_mvar: dict[str, float] | None = None
    active_reactive_lower: tuple[bool, ...] | None = None
    active_reactive_upper: tuple[bool, ...] | None = None
    active_voltage_lower: tuple[bool, ...] | None = None
    active_voltage_upper: tuple[bool, ...] | None = None
    model_id: str | None = None
    configuration_id: str | None = None
    input_id: str | None = None
    generator_ids: tuple[str, ...] = ()
    bus_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def polynomial_objective(net, generation, reactive=None) -> float:
    total = 0.0
    for _, row in net.poly_cost.iterrows():
        key = f"{row.et}:{int(row.element)}"
        if key not in generation:
            raise ValueError(f"Uncovered cost source {key}")
        p = generation[key]
        total += float(row.cp0_eur + row.cp1_eur_per_mw * p + row.cp2_eur_per_mw2 * p**2)
        if reactive is not None:
            q = reactive[key]
            total += float(row.cq0_eur + row.cq1_eur_per_mvar * q + row.cq2_eur_per_mvar2 * q**2)
    return total


def _branches(net, dc: bool) -> tuple[BranchResult, ...]:
    output = []
    for kind in ("line", "trafo"):
        start, _ = net._pd2ppc_lookups["branch"].get(kind, (0, 0))
        for offset, (index, row) in enumerate(net[kind].iterrows()):
            if not row.in_service:
                continue
            result = net[f"res_{kind}"].loc[index]
            branch = net._ppc["branch"][start + offset]
            limit_percent = float(row.get("max_loading_percent", 100))
            limit_percent = 100.0 if not math.isfinite(limit_percent) else limit_percent
            rating = float(branch[RATE_A])
            limit = rating if 0 < rating < 1e8 else None
            p_from, p_to = float(branch[PF]), float(branch[PT])
            # DC limits are RATE_A in MW, NOT AC current loading at an assumed voltage.
            loading = (abs(p_from) / limit * limit_percent if limit else 0.0) if dc else float(result.loading_percent)
            active = bool(limit and abs(abs(p_from) - limit) <= BOUND_TOL_MW) if dc else abs(loading-limit_percent) <= LOADING_TOL_PERCENT
            violated = bool(limit and abs(p_from) > limit + BOUND_TOL_MW) if dc else loading > limit_percent + LOADING_TOL_PERCENT
            output.append(BranchResult(f"{kind}:{index}", p_from, p_to, limit if dc else None,
                                       loading, limit_percent, active, violated))
    return tuple(output)


def _collect(net, config, operation, elapsed, proposed=None) -> SolveResult:
    inputs = build_dispatch_inputs(net, config)
    gens, ctx = inputs["generators"], inputs["context"]
    dc = config.physics_mode == "dc"
    realized, reactive = {}, {}
    for g in gens:
        kind, idx = g.asset_id.split(":")
        row = net[f"res_{kind}"].loc[int(idx)]
        realized[g.asset_id] = float(row.p_mw)
        reactive[g.asset_id] = float(row.q_mvar)
    branches = _branches(net, dc)
    values = list(realized.values()) + [v for b in branches for v in (b.p_from_mw, b.p_to_mw, b.loading_percent)]
    if not dc:
        values += list(reactive.values()) + net.res_bus.vm_pu.tolist()
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Non-finite or missing solver outputs")
    expected_branches = int(net.line.in_service.sum() + net.trafo.in_service.sum())
    if len(branches) != expected_branches or len(net.res_bus) != len(net.bus):
        raise ValueError("Incomplete solver output coverage")
    target = realized if proposed is None else proposed
    raw_balance = sum(target.values()) - sum(nodal_loads(net)) - float(net._ppc["bus"][:, GS].sum())
    # AC includes losses; raw balance is diagnostic, not a lossless AC feasibility condition.
    pviol = sum(p < g.p_min_mw-BOUND_TOL_MW or p > g.p_max_mw+BOUND_TOL_MW
                for g in gens for p in (target[g.asset_id],))
    realized_pviol = sum(realized[g.asset_id] < g.p_min_mw-BOUND_TOL_MW or
                         realized[g.asset_id] > g.p_max_mw+BOUND_TOL_MW for g in gens)
    qviol = None if dc else sum(reactive[g.asset_id] < g.q_min_mvar-BOUND_TOL_MW or
                               reactive[g.asset_id] > g.q_max_mvar+BOUND_TOL_MW for g in gens)
    vm = None if dc else {f"bus:{i}": float(net.res_bus.at[i, "vm_pu"]) for i in ordered_buses(net)}
    vviol = None if dc else sum(v < float(net.bus.at[int(k.split(":")[1]), "min_vm_pu"])-VOLTAGE_TOL_PU or
                               v > float(net.bus.at[int(k.split(":")[1]), "max_vm_pu"])+VOLTAGE_TOL_PU for k, v in vm.items())
    slack = {} if proposed is None else {g.asset_id: realized[g.asset_id]-target[g.asset_id]
                                        for g in gens if g.source_type == "ext_grid" or
                                        bool(net.gen.at[int(g.asset_id.split(":")[1]), "slack"])}
    feasible = (pviol == realized_pviol == 0 and not any(b.violated for b in branches)
                and all(abs(v) <= BALANCE_TOL_MW for v in slack.values())
                and (abs(raw_balance) <= BALANCE_TOL_MW if dc else qviol == vviol == 0))
    objective = polynomial_objective(net, target, None if dc or proposed is not None else reactive)
    if not math.isfinite(objective):
        raise ValueError("Non-finite objective")
    return SolveResult(
        "ok", True, config.physics_mode, operation, elapsed,
        generation_mw=realized, objective=objective, branches=branches,
        active_lower=tuple(abs(realized[g.asset_id]-g.p_min_mw) <= BOUND_TOL_MW for g in gens),
        active_upper=tuple(abs(realized[g.asset_id]-g.p_max_mw) <= BOUND_TOL_MW for g in gens),
        proposed_generation_mw=proposed, raw_balance_residual_mw=raw_balance, slack_adjustment_mw=slack,
        metrics={"feasible": feasible, "generator_bound_violations": pviol,
                 "realized_generator_bound_violations": realized_pviol,
                 "branch_violations": sum(b.violated for b in branches),
                 "voltage_violations": vviol, "reactive_violations": qviol,
                 "balance_applicable": dc},
        bus_voltage_pu=vm,
        reactive_generation_mvar=None if dc else reactive,
        active_reactive_lower=None if dc else tuple(abs(reactive[g.asset_id]-g.q_min_mvar) <= BOUND_TOL_MW for g in gens),
        active_reactive_upper=None if dc else tuple(abs(reactive[g.asset_id]-g.q_max_mvar) <= BOUND_TOL_MW for g in gens),
        active_voltage_lower=None if dc else tuple(abs(vm[f"bus:{i}"]-float(net.bus.at[i, "min_vm_pu"])) <= VOLTAGE_TOL_PU for i in ordered_buses(net)),
        active_voltage_upper=None if dc else tuple(abs(vm[f"bus:{i}"]-float(net.bus.at[i, "max_vm_pu"])) <= VOLTAGE_TOL_PU for i in ordered_buses(net)),
        model_id=ctx.network_fingerprint, configuration_id=ctx.configuration_id,
        input_id=digest({"p": nodal_loads(net), "q": net.load.q_mvar.tolist()}),
        generator_ids=ctx.controllable_assets, bus_ids=ctx.bus_ids,
    )


def solve_reference(net, config: SimulationConfig | None = None) -> SolveResult:
    config = config or SimulationConfig()
    started = time.perf_counter()
    try:
        build_dispatch_inputs(net, config)  # fail early on unsupported contracts
        (pp.rundcopp if config.physics_mode == "dc" else pp.runopp)(net, numba=False)
        if not net.OPF_converged:
            raise ValueError("OPF did not converge")
        result = _collect(net, config, "opf", time.perf_counter()-started)
        if not result.metrics["feasible"]:
            return SolveResult(**{**result.__dict__, "status": "invalid",
                                  "diagnostics": ("Converged OPF failed independent label validation",)})
        return result
    except Exception as exc:
        return SolveResult("failed", False, config.physics_mode, "opf", time.perf_counter()-started,
                           (f"{type(exc).__name__}: {exc}",))


def evaluate_decision(net, decision: DispatchDecision, config: SimulationConfig | None = None) -> SolveResult:
    config = config or SimulationConfig()
    started = time.perf_counter()
    try:
        inputs = build_dispatch_inputs(net, config)
        proposed = dict(decision.active_power_mw)
        if set(proposed) != set(inputs["context"].controllable_assets):
            raise ValueError("Setpoints must cover exactly controllable generator IDs (including ext_grid)")
        if not all(isinstance(p, (int, float)) and math.isfinite(p) for p in proposed.values()):
            raise ValueError("All proposed setpoints must be finite")
        if set(decision.voltage_targets_pu) - set(proposed):
            raise ValueError("Unknown voltage target IDs")
        if config.physics_mode == "dc" and decision.voltage_targets_pu:
            raise ValueError("Voltage targets are not applicable in DC")
        for key, value in proposed.items():
            kind, index = key.split(":")
            if kind == "gen":
                scale = float(net.gen.at[int(index), "scaling"])
                if scale <= 0:
                    raise ValueError("Controlled generator scaling must be positive")
                net.gen.at[int(index), "p_mw"] = value/scale
        for key, value in decision.voltage_targets_pu.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Voltage targets must be finite and positive")
            kind, index = key.split(":")
            net[kind].at[int(index), "vm_pu"] = value
        # PF balances at the slack. Its proposal is retained and discrepancies reported.
        (pp.rundcpp if config.physics_mode == "dc" else pp.runpp)(net, numba=False)
        if not net.converged:
            raise ValueError("Power flow did not converge")
        return _collect(net, config, "pf", time.perf_counter()-started, proposed)
    except Exception as exc:
        snapshot = dict(decision.active_power_mw)
        safe_snapshot = snapshot if all(isinstance(v, (int, float)) and math.isfinite(v) for v in snapshot.values()) else None
        return SolveResult("failed", False, config.physics_mode, "pf", time.perf_counter()-started,
                           (f"{type(exc).__name__}: {exc}",),
                           proposed_generation_mw=safe_snapshot)
