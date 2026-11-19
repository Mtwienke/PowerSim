"""Rebuild the small ODbL-licensed German transmission equivalent from pinned CSVs."""
import argparse
from collections import defaultdict
from hashlib import file_digest
import heapq
import json
import math
from pathlib import Path

import networkx as nx
import pandas as pd

SOURCE_MD5 = {
    "buses.csv": "031c30f04dc24e99210b3f5ad8a01c94",
    "lines.csv": "04d1e7cb33835de9c7d8be4716c6a8bf",
    "transformers.csv": "b6730b52097d2691d49231bdd53c7bd4",
}
BASE_KV = 380.0
N_REGIONS = 30
GROSS_MW = 48000.0


def read_source(root):
    frames = {}
    for name, expected in SOURCE_MD5.items():
        path = root / name
        with path.open("rb") as source:
            if file_digest(source, "md5").hexdigest() != expected:
                raise ValueError(f"{path}: checksum differs from Zenodo release 0.7")
        frames[name] = pd.read_csv(path, quotechar="'")
    return frames


def transmission_graph(frames):
    buses = frames["buses.csv"]
    domestic = buses[buses.country.eq("DE") & buses.dc.eq("f") & buses.under_construction.eq("f")]
    coordinates = {r.bus_id: (r.x, r.y) for r in domestic.itertuples()}
    graph = nx.Graph()
    graph.add_nodes_from(sorted(coordinates))
    elements = []

    def add(kind, asset, a, b, x, rating, km):
        if not all(math.isfinite(v) and v > 0 for v in (x, rating)):
            raise ValueError(f"Invalid electrical parameters: {asset}")
        previous = graph.get_edge_data(a, b, default={}).get("km", math.inf)
        graph.add_edge(a, b, km=min(previous, max(0.1, km)))
        elements.append((kind, asset, a, b, x, rating))

    for r in frames["lines.csv"].sort_values("line_id").itertuples():
        if r.bus0 in graph and r.bus1 in graph and r.under_construction == "f" and r.bus0 != r.bus1:
            add("line", r.line_id, r.bus0, r.bus1, r.x * (BASE_KV / r.voltage)**2, r.s_nom, r.length/1000)
    for r in frames["transformers.csv"].sort_values("transformer_id").itertuples():
        if r.bus0 in graph and r.bus1 in graph and r.bus0 != r.bus1:
            # Source has no transformer impedance. 12% at source rating is an assumption.
            add("transformer", r.transformer_id, r.bus0, r.bus1, .12 * BASE_KV**2 / r.s_nom, r.s_nom, .1)
    largest = min(nx.connected_components(graph), key=lambda nodes: (-len(nodes), min(nodes)))
    return graph.subgraph(largest), coordinates, elements, len(graph)


def connected_regions(graph):
    """Farthest-first seeds by shortest-path km, then multi-source graph Voronoi."""
    nearest = dict.fromkeys(graph, math.inf)
    seeds = []
    seed = min(graph, key=lambda n: (-graph.degree(n), n))
    for _ in range(N_REGIONS):
        seeds.append(seed)
        for node, distance in nx.single_source_dijkstra_path_length(graph, seed, weight="km").items():
            nearest[node] = min(nearest[node], distance)
        seed = min(nearest, key=lambda n: (-nearest[n], n))
    seeds.sort()
    queue = [(0., i, seed) for i, seed in enumerate(seeds)]
    heapq.heapify(queue)
    region = {}
    while queue:
        distance, i, node = heapq.heappop(queue)
        if node in region:
            continue
        region[node] = i
        for neighbor in graph[node]:
            if neighbor not in region:
                heapq.heappush(queue, (distance + graph[node][neighbor]["km"], i, neighbor))
    return seeds, region


def build(frames):
    graph, positions, elements, all_domestic = transmission_graph(frames)
    seeds, membership = connected_regions(graph)
    members = defaultdict(list)
    for bus in sorted(graph):
        members[membership[bus]].append(bus)
    buses = []
    for i, seed in enumerate(seeds):
        points = [positions[node] for node in members[i]]
        lat = sum(y for _, y in points)/len(points)
        lon = sum(x for x, _ in points)/len(points)
        buses.append(dict(index=i, name=f"DE region {i+1:02d}", seed_source_bus=seed,
                          source_buses=members[i], longitude=round(lon, 7), latitude=round(lat, 7),
                          gross_load_mw=round(GROSS_MW*len(points)/len(graph), 6), wind_mw=0., pv_mw=0.))

    # One explicitly assumed operating snapshot, not historical telemetry.
    from_north = sorted(buses, key=lambda row: (-row["latitude"], row["index"]))
    for row in from_north[:6]:
        row["wind_mw"] = 700.
    for row in from_north[-6:]:
        row["pv_mw"] = 400.

    groups = defaultdict(list)
    internal = 0
    for kind, asset, a, b, reactance, rating in elements:
        if a not in membership or b not in membership:
            continue
        if membership[a] == membership[b]:
            internal += 1
        else:
            groups[tuple(sorted((membership[a], membership[b])))].append((kind, asset, reactance, rating))
    corridors = []
    for (a, b), components in sorted(groups.items()):
        inverse_x = sum(1/x for _, _, x, _ in components)
        rating = min(capacity * x * inverse_x for _, _, x, capacity in components)
        corridors.append(dict(from_bus=a, to_bus=b, x_ohm_at_base=round(1/inverse_x, 8),
                              rating_mva=round(rating, 6),
                              source_elements=[f"{kind}:{asset}" for kind, asset, _, _ in components]))

    # Multiple controllable sources spread across the graph, with convex assumed costs.
    # Use the largest 18 demand regions.
    ranked = sorted(buses, key=lambda row: (-row["gross_load_mw"], row["index"]))
    slack = ranked[0]["index"]
    sources = [dict(bus=row["index"], is_slack=row["index"] == slack,
                    max_mw=round(1.85*row["gross_load_mw"], 6),
                    linear_cost=33. + (row["index"]*7 % 24),
                    quadratic_cost=.005 + .001*(row["index"] % 5))
               for row in sorted(ranked[:18], key=lambda row: row["index"])]
    return {
        "schema_version": 1, "base_voltage_kv": BASE_KV,
        "provenance": {
            "source": "PyPSA-Eur prebuilt electricity network based on OpenStreetMap, v0.7",
            "source_doi": "10.5281/zenodo.18619025", "license": "ODbL-1.0",
            "attribution": "Xiong, Fioriti, Neumann, Riepin & Brown; © OpenStreetMap contributors",
            "source_md5": SOURCE_MD5, "domestic_ac_candidate_buses": all_domestic,
            "source_buses_in_component": len(graph), "internal_elements_not_modeled": internal,
            "scope": "Domestic AC connected component, represented as a lossless 380 kV DC equivalent. No HVDC or cross-border exchange.",
            "operating_point": "Illustrative assumption: 48 GW gross demand by source bus count; 4.2 GW fixed wind and 2.4 GW fixed PV. Costs, dispatch capacities and transformer X are assumptions.",
        },
        "buses": buses, "corridors": corridors, "sources": sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Folder containing the three CSVs from Zenodo v0.7")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = build(read_source(args.source))
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
