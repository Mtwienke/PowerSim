"""Linear DC branch map and local, equality-constrained QP sensitivities."""
from dataclasses import dataclass

import numpy as np
import pandapower as pp
from pandapower.pypower.makeBdc import makeBdc
from pandapower.pypower.idx_brch import RATE_A
from pandapower.pypower.idx_bus import GS, BUS_TYPE, REF

from .network import generators, ordered_buses


@dataclass(frozen=True)
class DCRelation:
    m_g: np.ndarray
    m_d: np.ndarray
    constant: np.ndarray
    branch_ids: tuple[str, ...]
    balance_constant_mw: float
    branch_limits_mw: tuple[float | None, ...] = ()


def dc_branch_limit_mw(rating: float) -> float | None:
    return rating if 0 < rating < 1e8 else None


def dc_relation(net) -> DCRelation:
    """Compile DC flows and limits in the dispatch contract's bus/source/branch order."""
    pp.rundcpp(net, numba=False)
    ppc = net._ppc
    buses = ordered_buses(net)
    lookup = net._pd2ppc_lookups["bus"]
    indices = np.array([lookup[b] for b in buses], dtype=int)
    if len(set(indices)) != len(buses) or len(buses) != len(ppc["bus"]):
        raise ValueError("DC research map requires connected, unfused, in-service buses")
    if len(net.trafo3w):
        raise ValueError("Three-winding transformers are not supported by the research map")
    bbus, bf, pinj, pfinj, _ = makeBdc(ppc["bus"], ppc["branch"])
    reference = int(np.flatnonzero(ppc["bus"][:, BUS_TYPE] == REF)[0])
    keep = np.array([i for i in range(len(buses)) if i != reference])
    h = np.zeros((len(ppc["branch"]), len(buses)))
    h[:, keep] = np.linalg.solve(bbus.toarray()[np.ix_(keep, keep)].T, bf.toarray()[:, keep].T).T
    cgen = np.zeros((len(buses), len(generators(net))))
    for i, g in enumerate(generators(net)):
        cgen[lookup[int(g.bus_id.split(":")[1])], i] = 1
    gs = ppc["bus"][:, GS]
    constant = ppc["baseMVA"] * (np.asarray(pfinj) - h @ np.asarray(pinj)) - h @ gs
    branch_ids = tuple(f"{kind}:{i}" for kind in ("line", "trafo") for i, r in net[kind].iterrows() if r.in_service)
    if len(branch_ids) != len(h):
        raise ValueError("Unsupported DC branch topology")
    limits = tuple(dc_branch_limit_mw(float(rating)) for rating in ppc["branch"][:, RATE_A].real)
    return DCRelation(h @ cgen, h[:, indices], constant, branch_ids, float(gs.sum()), limits)


def reference_jacobian(cost_quadratic, relation, active_lower, active_upper,
                       active_branch, *, condition_limit=1e10,
                       generation=None, cost_linear=None, branch_flows=None, dual_tolerance=1e-5):
    """KKT derivative of strictly convex DC QP with a locally fixed active set.

    Returns (J or None, condition number). Undefined/redundant active sets are
    explicitly unavailable; no pseudoinverse is silently used as a label.
    """
    ng = len(cost_quadratic)
    nd = relation.m_d.shape[1]
    rows = [np.ones(ng)]
    rhs = [np.ones(nd)]
    signs = []
    for i, active in enumerate(np.asarray(active_lower) | np.asarray(active_upper)):
        if active:
            rows.append(np.eye(ng)[i])
            rhs.append(np.zeros(nd))
            signs.append(-1 if active_lower[i] else 1)
    for i, active in enumerate(active_branch):
        if active:
            rows.append(relation.m_g[i])
            rhs.append(relation.m_d[i])
            signs.append(1 if branch_flows is None or branch_flows[i] >= 0 else -1)
    a, b = np.asarray(rows), np.asarray(rhs)
    hessian = np.diag(2 * np.asarray(cost_quadratic))
    kkt = np.block([[hessian, a.T], [a, np.zeros((len(a), len(a)))]])
    condition = float(np.linalg.cond(kkt))
    if not np.isfinite(condition) or condition > condition_limit:
        return None, condition
    if generation is not None and cost_linear is not None and signs:
        # A is full row rank here. Strict complementarity excludes weakly active
        # inequalities at which the fixed-active-set derivative may be undefined.
        multipliers = np.linalg.solve(a @ a.T, -a @ (hessian @ generation + cost_linear))
        if np.any(np.asarray(signs)*multipliers[1:] <= dual_tolerance):
            return None, condition
    j = np.linalg.solve(kkt, np.vstack([np.zeros((ng, nd)), b]))[:ng]
    return j, condition
