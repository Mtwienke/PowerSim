# Germany-30: reduced domestic transmission benchmark

`germany30` is an offline, **DC-only** benchmark with 30 equivalent buses,
53 transmission corridors, and 18 conventional sources including one slack.
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
See the [data license notice](../powersim/benchmarks/germany30.LICENSE.md) and
[source methodology paper](https://doi.org/10.1038/s41597-025-04550-7).
The source's line ratings and reactances are themselves estimated/modelled
parameters; they are not certified operator limits.

1. Select German (`country=DE`) domestic AC buses and lines not marked under
   construction, plus transformers with both endpoints in that set. Keep
   the largest connected component: 780 of 781 candidate buses. Cross-border
   branches and exchanges, HVDC links and their converters are **excluded**.
   The modeled country is electrically closed; the slack is a domestic
   balancing source, not a tie-line import.
2. Use shortest-path distance (line length, with a 0.1 km weight for transformer
   connections) for deterministic farthest-first region seeds and a
   multi-source shortest-path assignment. Each region is connected; IDs
   are sorted by seed source ID. A map point is the mean coordinates of its
   members, not a substation.
3. Collapse intra-region branches (1,050 elements). Refer inter-region
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

## Illustrative operating point

48,000 MW gross demand is allocated by original bus count. The six
northernmost regional centroids receive fixed wind output (700 MW each), and the
six southernmost receive fixed PV output (400 MW each). Conventional capacity
is added in the largest 18 demand regions at 1.85× their demand; the largest
hosts the balancing source. Its and other sources' positive quadratic cost
curves are deterministic, illustrative modeling assumptions. **No generation,
cost, demand or renewable dispatch values are taken from market telemetry.**

The nominal DC reference supplies **41,400 MW** of conventional output beyond
the **6,600 MW** fixed injections, with one active corridor and no violations
at default limits. The stochastic generator instead starts from **gross**
nominal demand and replaces the illustrative fixed injections with its seeded
wind/PV regime (rather than adding renewables twice). Its default branch rating
scale is still 0.9; do not confuse that with the unscaled reference.

## Rebuild the bundle

The 50 KB derived database is bundled in the Python package. Download `buses.csv`,
`lines.csv` and `transformers.csv` from the versioned Zenodo record into a
local directory, then run:

```sh
python scripts/build_germany30.py path/to/source-csvs artifacts/germany30-rebuilt.json
cmp powersim/benchmarks/germany30.json artifacts/germany30-rebuilt.json
```

The builder checks source checksums and refuses to overwrite its destination.
The full raw CSVs are not bundled. The database and its transformations are
available for attribution and reproducibility; the ODbL applies to the
derived data separately from the application code.
