"""Rebuild the small ODbL-licensed German transmission equivalent from pinned CSVs."""
import argparse
from collections import defaultdict
from hashlib import file_digest
import heapq
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

SOURCE_MD5 = {
    "buses.csv": "031c30f04dc24e99210b3f5ad8a01c94",
    "lines.csv": "04d1e7cb33835de9c7d8be4716c6a8bf",
    "transformers.csv": "b6730b52097d2691d49231bdd53c7bd4",
    "links.csv": "2ddd515a30cfc19205e6681def579fbd",
}
BASE_KV = 380.0
N_REGIONS = 30
GROSS_MW = 70000.0
WIND_MW = 4200.0
PV_MW = 2400.0
TRANSFORMER_UK = .12
CAPACITY_FACTOR = 1.25
N_SOURCES = 18
MAX_REGION_FACTOR = 1.35


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
            # x, r and s_nom are already bundle values: the source divides x by circuits.
            add("line", r.line_id, r.bus0, r.bus1, r.x * (BASE_KV / r.voltage)**2, r.s_nom, r.length/1000)
    for r in frames["transformers.csv"].sort_values("transformer_id").itertuples():
        if r.bus0 in graph and r.bus1 in graph and r.bus0 != r.bus1:
            # Source has no transformer impedance. 12% at nameplate rating is an assumption;
            # per-unit uk is invariant to the listed voltage, so only uk and S enter.
            add("transformer", r.transformer_id, r.bus0, r.bus1,
                TRANSFORMER_UK * BASE_KV**2 / r.s_nom, r.s_nom, .1)
    largest = min(nx.connected_components(graph), key=lambda nodes: (-len(nodes), min(nodes)))
    return graph.subgraph(largest), coordinates, elements, len(graph)


def _projected(positions, nodes):
    """Longitude scaled by mean latitude so Ward distances are locally isotropic."""
    mean_lat = math.radians(np.mean([positions[node][1] for node in nodes]))
    return {node: np.array([positions[node][0] * math.cos(mean_lat), positions[node][1]])
            for node in nodes}


def _connected(graph, members):
    if not members:
        return False
    start = next(iter(members))
    seen, stack = {start}, [start]
    while stack:
        node = stack.pop()
        for neighbour in graph[node]:
            if neighbour in members and neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return len(seen) == len(members)


def balanced_regions(graph, positions):
    """Capacity-constrained agglomerative Ward clustering with a balance pass.

    The previous farthest-first seeds left one region with 216 of 780 buses while
    four held only four.  Merging adjacent clusters by Ward distance (with a hard
    size ceiling) keeps every region connected and spatially compact; the final
    pass moves boundary buses from overfull to underfull neighbours.  Region order
    is stable: by the highest-degree member's source bus id.
    """
    nodes = sorted(graph)
    if len(nodes) < N_REGIONS:
        raise ValueError("Not enough source buses for the requested regions")
    ceiling = math.ceil(MAX_REGION_FACTOR * len(nodes) / N_REGIONS)
    target = round(len(nodes) / N_REGIONS)
    points = _projected(positions, nodes)

    clusters = {node: {node} for node in nodes}
    centroids = {node: points[node].copy() for node in nodes}
    neighbours = {node: set(graph[node]) for node in nodes}
    alive = set(nodes)
    counter = 0

    def ward(a, b):
        na, nb = len(clusters[a]), len(clusters[b])
        return na * nb / (na + nb) * float(((centroids[a] - centroids[b]) ** 2).sum())

    queue = [(ward(a, b), min(a, b), max(a, b))
             for a in nodes for b in graph[a] if a < b]
    heapq.heapify(queue)

    def merge(a, b):
        nonlocal counter
        c = f"cluster-{counter}"
        counter += 1
        na, nb = len(clusters[a]), len(clusters[b])
        clusters[c] = clusters[a] | clusters[b]
        centroids[c] = (centroids[a] * na + centroids[b] * nb) / (na + nb)
        neighbours[c] = (neighbours[a] | neighbours[b]) - {a, b}
        for other in neighbours[c]:
            neighbours[other].discard(a)
            neighbours[other].discard(b)
            neighbours[other].add(c)
        alive.discard(a)
        alive.discard(b)
        alive.add(c)
        for old in (a, b):
            clusters.pop(old)
            centroids.pop(old)
            neighbours.pop(old)
        for other in neighbours[c]:
            heapq.heappush(queue, (ward(c, other), min(c, other), max(c, other)))

    while len(alive) > N_REGIONS:
        while True:
            _, a, b = heapq.heappop(queue)
            if a in alive and b in alive and b in neighbours[a]:
                break
        if len(clusters[a]) + len(clusters[b]) <= ceiling:
            merge(a, b)
            continue
        pending, merged = [(ward(a, b), a, b)], False
        while queue:
            item = heapq.heappop(queue)
            _, a2, b2 = item
            if a2 in alive and b2 in alive and b2 in neighbours[a2]:
                if len(clusters[a2]) + len(clusters[b2]) <= ceiling:
                    merge(a2, b2)
                    merged = True
                    break
                pending.append(item)
        for item in pending:
            heapq.heappush(queue, item)
        if not merged:
            merge(a, b)

    owner = {node: cluster for cluster, members in clusters.items() for node in members}
    for _ in range(4 * len(nodes)):
        sizes = {cluster: len(members) for cluster, members in clusters.items()}
        small = min(sizes, key=lambda cluster: (sizes[cluster], cluster))
        if sizes[small] >= target:
            break
        candidates = []
        for node in clusters[small]:
            for neighbour in graph[node]:
                donor = owner[neighbour]
                if donor != small and sizes[donor] > target:
                    candidates.append((donor, neighbour))
        if not candidates:
            break
        donor, node = min(candidates, key=lambda pair: (
            float(((points[pair[1]] - centroids[small]) ** 2).sum()), pair[0], pair[1]))
        if not _connected(graph, clusters[donor] - {node}):
            break
        clusters[donor].discard(node)
        clusters[small].add(node)
        owner[node] = small
        centroids[donor] = np.mean([points[n] for n in clusters[donor]], axis=0)
        centroids[small] = np.mean([points[n] for n in clusters[small]], axis=0)

    seeds = {cluster: min(members, key=lambda node: (-graph.degree(node), node))
             for cluster, members in clusters.items()}
    ordered = sorted(clusters, key=lambda cluster: seeds[cluster])
    result = [sorted(clusters[cluster]) for cluster in ordered]
    if len(result) != N_REGIONS or not all(_connected(graph, set(members)) for members in result):
        raise ValueError("Region construction produced an invalid partition")
    return result, [seeds[cluster] for cluster in ordered]


def border_statistics(frames, domestic_ids):
    lines = frames["lines.csv"]
    in_de = lines.bus0.isin(domestic_ids) | lines.bus1.isin(domestic_ids)
    crossing = lines[in_de & ~(lines.bus0.isin(domestic_ids) & lines.bus1.isin(domestic_ids))
                     & lines.under_construction.eq("f")]
    all_de = set(frames["buses.csv"].loc[frames["buses.csv"].country.eq("DE"), "bus_id"])
    links = frames["links.csv"]
    links_de = links[links.bus0.isin(all_de) | links.bus1.isin(all_de)]
    return {
        "border_ac_lines_excluded": int(len(crossing)),
        "border_ac_rating_mva_excluded": round(float(crossing.s_nom.sum()), 1),
        "hvdc_links_excluded": int(len(links_de)),
        "hvdc_capacity_mw_excluded": round(float(links_de.p_nom.sum()), 1),
    }


def build(frames):
    graph, positions, elements, all_domestic = transmission_graph(frames)
    regions, seeds = balanced_regions(graph, positions)
    membership = {bus: region for region, members in enumerate(regions) for bus in members}
    buses = []
    for i, members in enumerate(regions):
        points = [positions[node] for node in members]
        lat = sum(y for _, y in points)/len(points)
        lon = sum(x for x, _ in points)/len(points)
        buses.append(dict(index=i, name=f"DE region {i+1:02d}", seed_source_bus=seeds[i],
                          source_buses=members, longitude=round(lon, 7), latitude=round(lat, 7),
                          gross_load_mw=round(GROSS_MW*len(members)/len(graph), 6),
                          wind_mw=0., pv_mw=0.))

    # Illustrative near-peak snapshot: 70 GW gross demand is allocated by source-bus
    # count. Wind/PV nameplate is spread by a smooth latitude gradient weighted by
    # region size instead of a flat north/south cut-off.
    latitudes = np.array([row["latitude"] for row in buses])
    span = latitudes.max() - latitudes.min()
    north = (latitudes - latitudes.min())/span if span else np.full(len(buses), .5)
    sizes = np.array([len(row["source_buses"]) for row in buses], dtype=float)
    wind_weight = north**2 * sizes
    pv_weight = (1 - north)**2 * sizes
    wind_weight /= wind_weight.sum()
    pv_weight /= pv_weight.sum()
    for row, wind, pv in zip(buses, WIND_MW*wind_weight, PV_MW*pv_weight):
        row["wind_mw"] = round(float(wind), 6)
        row["pv_mw"] = round(float(pv), 6)

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
        # Exact DC limit of the parallel bundle when flow splits as 1/x: the
        # tightest member fixes the equivalent rating.
        rating = min(capacity * x * inverse_x for _, _, x, capacity in components)
        corridors.append(dict(from_bus=a, to_bus=b, x_ohm_at_base=round(1/inverse_x, 8),
                              rating_mva=round(rating, 6),
                              source_elements=[f"{kind}:{asset}" for kind, asset, _, _ in components]))

    # 18 sources in the largest-demand regions, total capacity at a documented
    # reserve margin. Costs follow a monotone illustrative merit order by unit
    # size (larger units cheaper); the slack is an expensive domestic balancer.
    ranked = sorted(buses, key=lambda row: (-row["gross_load_mw"], row["index"]))
    chosen = sorted(ranked[:N_SOURCES], key=lambda row: row["index"])
    slack = ranked[0]["index"]
    share = sum(len(row["source_buses"]) for row in chosen)
    by_size = sorted(chosen, key=lambda row: (-row["gross_load_mw"], row["index"]))
    rank = {row["index"]: position for position, row in enumerate(by_size)}
    sources = []
    for row in chosen:
        position = rank[row["index"]]
        linear, quadratic = 38. + 2.*position, .004 + .0004*position
        if row["index"] == slack:
            linear, quadratic = 85., .01
        sources.append(dict(bus=row["index"], is_slack=row["index"] == slack,
                            max_mw=round(CAPACITY_FACTOR*GROSS_MW*len(row["source_buses"])/share, 6),
                            linear_cost=linear, quadratic_cost=quadratic))

    sizes = [len(members) for members in regions]
    provenance = {
        "source": "PyPSA-Eur prebuilt electricity network based on OpenStreetMap, v0.7",
        "source_doi": "10.5281/zenodo.18619025", "license": "ODbL-1.0",
        "attribution": "Xiong, Fioriti, Neumann, Riepin & Brown; © OpenStreetMap contributors",
        "source_md5": SOURCE_MD5, "domestic_ac_candidate_buses": all_domestic,
        "source_buses_in_component": len(graph), "internal_elements_not_modeled": internal,
        "region_partition": ("capacity-constrained agglomerative Ward clustering with a "
                             f"{MAX_REGION_FACTOR:.2f}x size ceiling and a balance pass"),
        "region_bus_count_min": min(sizes), "region_bus_count_max": max(sizes),
        "corridor_impedance": ("cut-set parallel equivalent of boundary elements; intra-region "
                               "series impedance is not represented (see docs/germany30.md)"),
        "scope": ("Domestic AC connected component, represented as a lossless 380 kV DC equivalent. "
                  "No HVDC or cross-border exchange."),
        "operating_point": ("Illustrative near-peak assumption: 70 GW gross demand by source bus "
                            "count; 4.2 GW wind and 2.4 GW PV spread by a latitude-weighted region "
                            "size. Costs, dispatch capacities and transformer X are assumptions."),
        "transformer_impedance": "12% on listed nameplate MVA; the source has no transformer impedance.",
        "cost_model": ("Illustrative monotone merit order by unit size, 38-72 EUR/MWh plus a small "
                       "quadratic term; the slack is an 85 EUR/MWh domestic balancer."),
        **border_statistics(frames, set(positions)),
    }
    return {
        "schema_version": 2, "base_voltage_kv": BASE_KV,
        "provenance": provenance, "buses": buses, "corridors": corridors, "sources": sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Folder containing the CSVs from Zenodo v0.7")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = build(read_source(args.source))
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()
