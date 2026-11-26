"""Render a solved scenario from its stable bus/source/branch IDs, not map proximity."""
from html import escape
import json
import math

import networkx as nx
import plotly.graph_objects as go

from .network import generators, network_fingerprint, nodal_power, ordered_buses

FLOW_TOL_MW = .0001
NODE_STYLES = {
    "Slack / external grid": ("#7c3aed", "diamond"),
    "Conventional source": ("#c2410c", "triangle-up"),
    "Fixed renewables": ("#0f766e", "square"),
    "Demand / other bus": ("#2563eb", "circle"),
}
EDGE_STYLES = {
    "In service": "#3b82f6",
    "Near limit / active": "#d97706",
    "Over limit": "#dc2626",
    "Idle": "#94a3b8",
    "Unavailable / out of service": "#cbd5e1",
}


def _edge_specs(net, buses):
    for kind, a, b in (("line", "from_bus", "to_bus"), ("trafo", "hv_bus", "lv_bus")):
        for index, row in net[kind].iterrows():
            u, v = int(row[a]), int(row[b])
            if u in buses and v in buses:
                yield f"{kind}:{index}", u, v, kind, bool(row.in_service)


def _positions(net, buses, edges):
    if net.get("source_provenance", {}).get("source_doi") == "10.5281/zenodo.18619025":
        coords = {bus: json.loads(net.bus.at[bus, "geo"])["coordinates"] for bus in buses}
        ref_lat = sum(v[1] for v in coords.values()) / len(coords)
        # Local equal scaling for visual geometry only. Not a physical network model.
        aspect = math.cos(math.radians(ref_lat))
        return {bus: (lon * aspect, lat) for bus, (lon, lat) in coords.items()}, "Regional centroids, approximate geography"
    graph = nx.Graph()
    graph.add_nodes_from(buses)
    graph.add_edges_from((u, v) for _, u, v, _, _ in edges)
    return nx.spring_layout(graph, seed=2026, iterations=150), "Electrical schematic, not geographic"


def _branch_style(branch, in_service):
    if branch is None or not in_service:
        return "Unavailable / out of service", 0.
    p, q = branch["p_from_mw"], branch["p_to_mw"]
    flow = max(abs(p), abs(q))
    if branch["violated"]:
        return "Over limit", flow
    if flow <= FLOW_TOL_MW:
        return "Idle", flow
    if branch["active"] or branch["loading_percent"] >= .9 * branch["limit_percent"]:
        return "Near limit / active", flow
    return "In service", flow


def _add_branches(fig, positions, specs, flows, physics_mode):
    max_flow = max((max(abs(b["p_from_mw"]), abs(b["p_to_mw"])) for b in flows.values()), default=0.)
    seen = set()
    groups = {}
    for edge in specs:
        groups.setdefault(tuple(sorted(edge[1:3])), []).append(edge)
    for group in groups.values():
        for i, (asset, u, v, kind, in_service) in enumerate(group):
            x0, y0 = positions[u]
            x1, y1 = positions[v]
            if u > v:  # use the same perpendicular for opposed parallel circuits
                sign = -1
            else:
                sign = 1
            k = .10 * (i-(len(group)-1)/2) * sign
            xm, ym = (x0+x1)/2-k*(y1-y0), (y0+y1)/2+k*(x1-x0)
            branch = flows.get(asset)
            label, flow = _branch_style(branch, in_service)
            width = 1 + 5*(flow/max_flow if max_flow else 0)
            if branch is None:
                detail = "No solved flow available"
            else:
                p, q = branch["p_from_mw"], branch["p_to_mw"]
                rated = branch["limit_mw"]
                applicable = rated is not None or physics_mode == "ac"
                loading = (f"{100*branch['loading_percent']/branch['limit_percent']:.1f}% of limit"
                           if applicable and branch["limit_percent"] else "Loading unavailable")
                rating = (f"{rated:,.2f} MW" if rated is not None else
                          f"{branch['limit_percent']:g}% allowed loading" if applicable else "no MW rating")
                detail = (f"From-end injection: {p:,.3f} MW<br>To-end injection: {q:,.3f} MW<br>"
                          f"{loading} · limit {rating}")
                # Same-sign withdrawals at both terminals do not define one through-flow.
                if p > FLOW_TOL_MW and q < -FLOW_TOL_MW:
                    start, end = (x0, y0), (x1, y1)
                elif q > FLOW_TOL_MW and p < -FLOW_TOL_MW:
                    start, end = (x1, y1), (x0, y0)
                else:
                    start = end = None
                if start is not None:
                    f = .18
                    fig.add_annotation(x=xm+f*(end[0]-xm), y=ym+f*(end[1]-ym),
                                       ax=xm+f*(start[0]-xm), ay=ym+f*(start[1]-ym),
                                       xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                                       arrowhead=3, arrowsize=1, arrowwidth=min(width, 2.5),
                                       arrowcolor=EDGE_STYLES[label])
            fig.add_trace(go.Scatter(
                x=[x0, xm, x1], y=[y0, ym, y1], mode="lines+markers", name=label,
                legendgroup=label, showlegend=label not in seen,
                meta={"asset_id": asset, "from_bus": f"bus:{u}", "to_bus": f"bus:{v}"},
                line={"color": EDGE_STYLES[label], "width": width, "dash": "dash" if kind == "trafo" else "solid"},
                marker={"size": [0, 11, 0], "color": "rgba(0,0,0,0.01)"},
                hovertemplate=f"<b>{asset}</b> · bus {u} ↔ bus {v}<br>{detail}<extra></extra>"))
            seen.add(label)


def _add_buses(fig, net, buses, positions, result, demand, renewables):
    by_bus = {bus: [] for bus in buses}
    for g in generators(net):
        by_bus[int(g.bus_id.split(":")[1])].append(g)
    actual = result.get("generation_mw", {})
    proposed = result.get("proposed_generation_mw") or {}
    slack = result.get("slack_adjustment_mw", {})
    groups = {name: [] for name in NODE_STYLES}
    co_located = []
    for i, bus in enumerate(buses):
        sources = by_bus[bus]
        is_slack = any(g.source_type == "ext_grid" or
                       bool(net.gen.at[int(g.asset_id.split(":")[1]), "slack"]) for g in sources)
        role = ("Slack / external grid" if is_slack else "Conventional source" if sources else
                "Fixed renewables" if renewables[i] > 0 else "Demand / other bus")
        name = escape(str(net.bus.at[bus, "name"]))
        text = [f"<b>Bus {bus}</b> · {name}", f"Gross demand: {demand[i]:,.2f} MW",
                f"Fixed injection: {renewables[i]:,.2f} MW",
                f"Net demand: {demand[i]-renewables[i]:,.2f} MW"]
        vm = result.get("bus_voltage_pu")
        if vm is not None:
            voltage = vm.get(f"bus:{bus}", vm.get(bus))
            if voltage is not None:
                text.append(f"Voltage: {voltage:.4f} pu")
        for source in sources:
            asset = source.asset_id
            actual_text = f"{actual[asset]:,.3f} MW" if asset in actual else "unavailable"
            line = f"{asset} realized: {actual_text}"
            if asset in proposed:
                line += f" · proposed: {proposed[asset]:,.3f} MW"
            if asset in slack:
                line += f" · slack adjustment: {slack[asset]:+,.3f} MW"
            text.append(line)
        magnitude = max(abs(demand[i]), abs(renewables[i]),
                        sum(abs(actual[g.asset_id]) for g in sources if g.asset_id in actual))
        groups[role].append((bus, magnitude, "<br>".join(text)))
        if renewables[i] > 0 and sources:
            co_located.append((bus, renewables[i], magnitude))
    max_magnitude = max((m for group in groups.values() for _, m, _ in group), default=0.) or 1.

    def marker_size(magnitude):
        return 10 + 18 * math.sqrt(magnitude / max_magnitude)

    if co_located:
        fig.add_trace(go.Scatter(
            x=[positions[bus][0] for bus, _, _ in co_located],
            y=[positions[bus][1] for bus, _, _ in co_located],
            mode="markers", name="Renewables at source bus",
            hovertext=[f"Bus {bus} · Fixed injection: {power:,.2f} MW" for bus, power, _ in co_located],
            hovertemplate="%{hovertext}<extra></extra>",
            marker={"size": [marker_size(m)+10 for _, _, m in co_located],
                    "symbol": "circle-open", "color": NODE_STYLES["Fixed renewables"][0],
                    "line": {"width": 2}}))
    for label, rows in groups.items():
        if not rows:
            continue
        color, symbol = NODE_STYLES[label]
        fig.add_trace(go.Scatter(
            x=[positions[bus][0] for bus, _, _ in rows], y=[positions[bus][1] for bus, _, _ in rows],
            mode="markers+text", name=label, text=[str(bus) for bus, _, _ in rows],
            customdata=[f"bus:{bus}" for bus, _, _ in rows],
            hovertext=[text for _, _, text in rows], hovertemplate="%{hovertext}<extra></extra>",
            textposition="top center", marker={"size": [marker_size(m) for _, m, _ in rows],
                                               "symbol": symbol, "color": color, "line": {"color": "white", "width": 1}}))


def solution_figure(net, result, *, demand=None, renewables=None, label=None):
    """Plot an OPF/PF result, preserving the chosen/raw distinction at the caller."""
    if result.get("model_id") and result["model_id"] != network_fingerprint(net):
        raise ValueError("Solution topology does not match the network")
    buses = ordered_buses(net)
    if result.get("bus_ids") and tuple(result["bus_ids"]) != tuple(f"bus:{b}" for b in buses):
        raise ValueError("Solution bus ordering does not match the network")
    demand = nodal_power(net, "load") if demand is None else demand
    renewables = nodal_power(net, "sgen") if renewables is None else renewables
    if any(len(a) != len(buses) or not all(math.isfinite(v) for v in a) for a in (demand, renewables)):
        raise ValueError("Scenario graph requires one finite demand/injection per ordered bus")
    edges = tuple(_edge_specs(net, set(buses)))
    positions, layout = _positions(net, buses, edges)
    flows = {b["asset_id"]: b for b in result.get("branches", [])}
    if set(flows) - {asset for asset, *_ in edges}:
        raise ValueError("Solution contains branches absent from the network")
    fig = go.Figure()
    _add_branches(fig, positions, edges, flows, result.get("physics_mode"))
    _add_buses(fig, net, buses, positions, result, demand, renewables)
    mode = result.get("physics_mode", "").upper()
    operation = result.get("operation", "").upper()
    feasibility = result.get("metrics", {}).get("feasible")
    status = "Feasible" if feasibility else "Not feasible" if feasibility is not None else "Solution unavailable"
    provenance = net.get("source_provenance", {})
    attribution = f"<br>{escape(provenance['attribution'])} · {escape(provenance['license'])}" if provenance else ""
    title = f"{escape(label or str(net.name or 'Network solution'))} · {mode} {operation}".strip()
    fig.update_layout(template="plotly_white", height=650, hovermode="closest",
                      title={"text": f"{title}<br><sup>{status} · {layout}{attribution}</sup>"},
                      xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
                      yaxis={"visible": False}, margin={"l": 25, "r": 25, "t": 100, "b": 25},
                      legend={"orientation": "h", "y": -.04})
    return fig


def dataset_solution(dataset, split, sample):
    """Read one originally attempted sample and its recorded reference; no re-solve."""
    from .data import load_dataset_attempt

    net, manifest, record = load_dataset_attempt(dataset, split, sample)
    bus_ids = manifest["bus_ids"]
    demand = record["gross_load_mw"]
    renewable = [0.] * len(bus_ids)
    for site, power in zip(manifest["installations"], record["renewable_mw"], strict=True):
        renewable[bus_ids.index(site["bus_id"])] += power
    return net, record["reference"], demand, renewable
