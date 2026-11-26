"""Stable ordering and structural identity, independent of scenario load values."""
from dataclasses import asdict
import hashlib
import json
import math

import numpy as np
import pandapower as pp
import pandapower.networks as pn

from ..core.config import SimulationConfig
from ..core.contracts import DispatchContext, DispatchObservation, Generator


def digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def make_case(config: SimulationConfig | None = None) -> pp.pandapowerNet:
    config = config or SimulationConfig()
    if config.benchmark == "germany30":
        from .german_case import make_germany30
        net = make_germany30()
    else:
        net = getattr(pn, config.benchmark)()
    for table in (net.line, net.trafo):
        if len(table):
            if "max_loading_percent" not in table:
                table["max_loading_percent"] = 100.0
            table["max_loading_percent"] = table.max_loading_percent.fillna(100) * config.branch_limit_scale
    return net


def network_fingerprint(net) -> str:
    # Operating points (load P/Q, generator P, results) are NOT model identity.
    # Include complete supported topology, transformer taps, costs and constraints.
    payload = {"sn_mva": float(net.sn_mva), "f_hz": float(net.f_hz)}
    excluded = {"gen": ["p_mw"], "sgen": ["p_mw", "q_mvar"]}
    for name in ("bus", "line", "trafo", "trafo3w", "switch", "gen", "ext_grid",
                 "sgen", "shunt", "poly_cost", "pwl_cost"):
        table = net[name].drop(columns=excluded.get(name, []), errors="ignore")
        payload[name] = json.loads(table.sort_index().to_json(orient="split", double_precision=15))
    return digest(payload)


def ordered_buses(net) -> tuple[int, ...]:
    return tuple(int(i) for i in net.bus.index[net.bus.in_service].sort_values())


def generators(net) -> tuple[Generator, ...]:
    result = []
    unsupported = [name for name in ("trafo3w", "dcline", "line_dc", "vsc", "impedance",
                                     "storage", "ward", "xward", "motor") if name in net and len(net[name])]
    if unsupported:
        raise ValueError(f"Unsupported research-adapter devices: {', '.join(unsupported)}")
    if "controllable" in net.load and bool(net.load.controllable.fillna(False).any()):
        raise ValueError("OPF loads must be fixed exogenous nodal inputs, not controllable loads")
    if len(net.pwl_cost):
        raise ValueError("Research adapter supports polynomial costs, not piecewise-linear costs")
    if len(net.sgen) and "controllable" in net.sgen and bool(net.sgen.controllable.fillna(False).any()):
        raise ValueError("Controllable sgen is unsupported; renewables must be fixed nodal injections")
    for kind in ("ext_grid", "gen"):
        for index, row in net[kind].sort_index().iterrows():
            if not row.in_service:
                continue
            cost = net.poly_cost[(net.poly_cost.et == kind) & (net.poly_cost.element == index)]
            if len(cost) != 1:
                raise ValueError(f"Exactly one polynomial cost required for {kind}:{index}")
            c = cost.iloc[0]
            bounds = [float(row.get(k, float("nan"))) for k in ("min_p_mw", "max_p_mw")]
            if not all(math.isfinite(x) for x in bounds) or bounds[0] > bounds[1]:
                raise ValueError(f"Finite ordered active bounds required for {kind}:{index}")
            result.append(Generator(
                f"{kind}:{index}", *bounds, float(c.cp1_eur_per_mw),
                f"bus:{int(row.bus)}", kind, float(c.cp2_eur_per_mw2), float(c.cp0_eur),
                float(row.get("min_q_mvar", -1e9)), float(row.get("max_q_mvar", 1e9)),
                float(c.cq0_eur), float(c.cq1_eur_per_mvar), float(c.cq2_eur_per_mvar2),
            ))
    return tuple(result)


def nodal_power(net, element) -> tuple[float, ...]:
    values = dict.fromkeys(ordered_buses(net), 0.0)
    for row in net[element].itertuples():
        if row.in_service and row.bus in values:
            values[row.bus] += float(row.p_mw) * float(row.scaling)
    return tuple(values.values())


def nodal_loads(net) -> tuple[float, ...]:
    return tuple(load - injection for load, injection in zip(nodal_power(net, "load"), nodal_power(net, "sgen")))


def apply_nodal_loads(net, values) -> None:
    """Scenario inputs already include renewables. Never also dispatch sgen."""
    buses = ordered_buses(net)
    if len(values) != len(buses) or not np.isfinite(values).all():
        raise ValueError("One finite net load required per ordered bus")
    if len(net.sgen):
        net.sgen.loc[:, "p_mw"] = 0.0
        net.sgen.loc[:, "q_mvar"] = 0.0
    net.load.loc[:, ["p_mw", "q_mvar"]] = 0.0
    for bus, value in zip(buses, values):
        indices = net.load.index[(net.load.bus == bus) & net.load.in_service]
        if len(indices):
            net.load.at[indices[0], "p_mw"] = float(value) / float(net.load.at[indices[0], "scaling"])
        else:
            pp.create_load(net, bus, p_mw=float(value), q_mvar=0)


class NodalLoadWriter:
    """Prepare one private DC scratch network; subsequent cases update only P."""

    def __init__(self, net):
        self.net = net
        apply_nodal_loads(net, nodal_loads(net))
        first = net.load[net.load.in_service].reset_index().drop_duplicates("bus").set_index("bus")
        rows = first.loc[list(ordered_buses(net))]
        self.indices = rows["index"].to_list()
        scaling = rows.scaling.to_numpy(dtype=float)
        if not np.isfinite(scaling).all() or (scaling <= 0).any():
            raise ValueError("Scenario load row scaling must be finite and positive")
        self.scaling = scaling

    def apply(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != self.scaling.shape or not np.isfinite(values).all():
            raise ValueError("One finite net load required per ordered bus")
        self.net.load.loc[self.indices, "p_mw"] = values / self.scaling


def build_dispatch_inputs(net, config: SimulationConfig | None = None) -> dict:
    config = config or SimulationConfig()
    fp = network_fingerprint(net)
    gens = generators(net)
    ctx = DispatchContext(
        "2.0", fp, (("power", "MW"), ("reactive_power", "MVAr"), ("cost", "EUR/h")),
        tuple(g.asset_id for g in gens),
        ("Fixed renewables are negative nodal load; all conventional sources include slack.",),
        tuple(f"bus:{i}" for i in ordered_buses(net)), config.physics_mode,
        digest({"model": fp, "configuration": asdict(config)}),
    )
    return {"context": ctx, "generators": gens,
            "generator_indices": {g.asset_id: int(g.asset_id.split(":")[1]) for g in gens}}


def observation(net, config: SimulationConfig | None = None, step_id=0) -> DispatchObservation:
    inputs = build_dispatch_inputs(net, config)
    ctx = inputs["context"]
    loads = nodal_loads(net)
    return DispatchObservation(
        "2.0", ctx.network_fingerprint, "2026-01-01T00:00:00Z", step_id, .25,
        sum(loads), 0.0, inputs["generators"], ctx.bus_ids, loads,
        digest({"p": loads, "q": net.load.q_mvar.tolist(), "step": step_id}),
        ctx.physics_mode, ctx.configuration_id,
    )
