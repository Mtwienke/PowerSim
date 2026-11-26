# Germany-30: reduced domestic transmission benchmark

`germany30` is an offline, **DC-only** benchmark with 30 equivalent buses,
54 transmission corridors, and 18 conventional sources including one slack.
It is derived from a public representation of the German grid, not 30 actual
substations, live grid state, or a model fit for system operation. The existing
IEEE cases and their AC options are unchanged.

```sh
powersim reference --benchmark germany30
powersim dataset artifacts/germany-data --benchmark germany30 --preset smoke --workers 4
powersim plot-solution artifacts/germany-solution.html --benchmark germany30
powersim plot-solution artifacts/germany-sampled.html --dataset artifacts/germany-data --split train --sample 0
```

In Streamlit, select **germany30** and **Solve reference OPF**. The Solution
Flow view uses approximate regional centroids. The HTML export includes its
JavaScript and works without basemap tiles or internet. The selected dataset
sample is the original zero-based attempt number, not its row in the filtered
NPZ; export replays its recorded result and will not silently re-solve a failed
attempt.

## Source and reduction

The electrical source is Xiong, Fioriti, Neumann, Riepin and Brown,
[*Prebuilt Electricity Network for PyPSA-Eur based on OpenStreetMap Data*,
v0.7](https://doi.org/10.5281/zenodo.18619025), ODbL 1.0, with
[© OpenStreetMap contributors](https://www.openstreetmap.org/copyright).
See the [data license notice](../powersim/instance_generation/benchmarks/germany30.LICENSE.md) and
[source methodology paper](https://doi.org/10.1038/s41597-025-04550-7).
The source's line ratings and reactances are themselves estimated/modelled
parameters; they are not certified operator limits.

1. Select German (`country=DE`) domestic AC buses and lines not marked under
   construction, plus transformers with both endpoints in that set. Keep
   the largest connected component: 780 of 781 candidate buses. Cross-border
   branches and exchanges, HVDC links and their converters are **excluded**:
   42 in-service AC border lines (74.6 GVA) and four HVDC links (3.6 GW) are
   not represented. The modeled country is electrically closed; the slack is a
   domestic balancing source, not a tie-line import.
2. Partition the 780-bus graph into 30 connected regions with
   capacity-constrained agglomerative Ward clustering on projected bus
   coordinates (a 1.35× size ceiling plus a boundary-bus balance pass).
   Regions hold **16–36 source buses** (mean 26). Region IDs are ordered by the
   highest-degree member's source ID; a map point is the mean coordinates of
   its members, not a substation. The earlier farthest-first/Voronoi partition
   left one region with 216 buses and four with only four; the balance pass
   removes that loss of spatial resolution.
3. Collapse intra-region branches (1,026 elements). Refer inter-region
   reactances to a common 380 kV base via `X_380 = X_source * (380/V_source)^2`.
   Transformer impedance is absent in the source and is assumed to be 12% on
   its listed MVA base. Boundary components between each pair form one
   equivalent, with `X_eq = 1 / sum(1/X_i)` and rating
   `min_i(S_i * X_i / X_eq)`. The latter respects every member's rating **under
   the assumed parallel flow split after bus aggregation**. It does not protect
   omitted internal constraints or certify the original network. The compiled
   JSON retains regional membership and contributing source element IDs.
4. Represent each equivalent as a lossless pandapower line at 380 kV (`length=1`
   is only a storage convention for series X). Resistance, shunt charging,
   original transformer taps and reactive behavior are not preserved. Since
   this is not an AC equivalent, AC solves are explicitly unavailable here.

The source's `x`, `r` and `s_nom` are already **bundle values**: for a line with
`circuits=c`, `s_nom = c * sqrt(3) * V * i_nom` and the listed reactance is
divided by `c`. The builder therefore uses them directly; dividing again would
double-count parallel circuits. This was verified against all 1,019 in-service
German lines in the v0.7 release.

## Assumption audit (2026-11 revision)

The network derivation is data-driven; the operating point is not. This table
separates the two and records what was checked during the rework.

| Assumption | Evidence / check | Consequence |
| --- | --- | --- |
| Region partition | Old farthest-first: 4–216 buses per region, one region held 28% of demand. New: 16–36. | The 30-bus model now has comparable spatial resolution across Germany. |
| Corridor impedance | Cut-set parallel equivalent only. An indicative DC Kron reduction with one representative bus per region gives median **5.4× larger** pairwise reactances (range 0.4–24×). | Intra-region series impedance is omitted, so modeled transfers are optimistic. This is a documented reduction error, not a calibrated correction. |
| Transformer impedance | The source has no transformer `x`. 12% on nameplate MVA is assumed for every listed unit. | A ±2 percentage-point change shifts transformer-dominated corridor X by roughly ±17%; no transformer-specific data is available to improve this. |
| Branch ratings | Source `s_nom` is thermal (per-circuit `i_nom` × circuits). The equivalent uses the tightest member under the DC parallel split. | Ratings are not N-1 security limits; the dataset's default 0.9 scale is a separate stress convention. |
| Cross-border exchange | 42 AC border lines (74.6 GVA) and 4 HVDC links (3.6 GW) excluded. | Germany is a closed island; import/export flexibility and border congestion are absent. |
| Demand | 70 GW gross is a near-peak snapshot, about 1.2× the 2024 average gross load (≈519 TWh / 8760 h ≈ 59 GW). It is allocated by source-bus count. | Bus count is a neutral proxy for substation density, not measured consumption; it does not represent population or industry. |
| Conventional fleet | 18 sources in the largest-demand regions, total 87.5 GW (1.25× gross demand). | This is close to the scale of Germany's dispatchable fleet, but the spatial allocation and unit sizes are illustrative. |
| Costs | Monotone merit order by unit size (38–72 EUR/MWh plus a small quadratic term); the slack is an 85 EUR/MWh balancer. | No plant-level cost data is used. The ordering is transparent and independent of bus index, but it is not a market model. |
| Renewables | 4.2 GW wind and 2.4 GW PV, spread by a smooth latitude-weighted region size. | A low-renewable near-peak snapshot (9.4% of gross demand), not historical weather or installed capacity. The stochastic generator replaces these injections. |
| Voltage levels | 220 kV and 380 kV are merged on a 380 kV base; 110 kV distribution is absent. | The equivalent cannot reproduce voltage- or reactive-driven constraints. |

## Illustrative operating point

70,000 MW gross demand is allocated by original bus count. Wind and PV
nameplate are spread across regions by a smooth north–south gradient weighted
by region size (4,200 MW wind, 2,400 MW PV in total), replacing the earlier flat
"six northernmost / six southernmost" split. Conventional capacity is added in
the largest 18 demand regions at 1.25× their share of gross demand (87.5 GW
total); the largest hosts the balancing source. Its and other sources' positive
quadratic cost curves are deterministic, illustrative modeling assumptions.
**No generation, cost, demand or renewable dispatch values are taken from
market telemetry.**

The nominal DC reference supplies **63,400 MW** of conventional output beyond
the **6,600 MW** fixed injections, with no violations and a maximum corridor
loading of **76.4%** at default limits. The stochastic generator instead starts
from **gross** nominal demand and replaces the illustrative fixed injections
with its seeded wind/PV regime (rather than adding renewables twice). Its
default branch rating scale is still 0.9; do not confuse that with the unscaled
reference. In a 200-attempt-per-split smoke at the documented 0.9 scale, all
five splits retained valid reference labels; the pre-revision bundle retained
75–95%. That smoke is not a full-suite feasibility result. At the same 0.9
scale, 128 of 200 train samples had one binding corridor, so the benchmark still
exercises active transmission constraints even though the unscaled reference
does not.

## Rebuild the bundle

The derived database is bundled in the Python package. Download `buses.csv`,
`lines.csv`, `transformers.csv` and `links.csv` from the versioned Zenodo record
into a local directory, then run:

```sh
python scripts/build_germany30.py path/to/source-csvs artifacts/germany30-rebuilt.json
cmp powersim/instance_generation/benchmarks/germany30.json artifacts/germany30-rebuilt.json
```

The builder checks source checksums and refuses to overwrite its destination.
The full raw CSVs are not bundled. The database and its transformations are
available for attribution and reproducibility; the ODbL applies to the
derived data separately from the application code.

The current bundle is schema version 2 (2026-11-20). Results recorded before
that date, including the seed-2027 suite, used schema version 1 and are not
reproducible from this file.
