# PowerSim

**Research default: IEEE 30, DC physics, numerical DC-OPF.** AC and IEEE 14
remain selectable. The headless reference/data pipeline is available with:

```sh
python -m powersim.cli reference
python -m powersim.cli reference --physics ac --benchmark case14
python -m powersim.cli dataset artifacts/data --preset smoke
```

See [research protocol](docs/research.md) for ordering, validation, local
sensitivities, reproducibility, and explicit differences from Gao et al.
The older architecture notes below are superseded by that protocol.

## Change the dispatch logic here: `controllers/my_dispatch.py`

Edit `choose_dispatch(observation)` to implement a policy. It receives immutable typed values in MW and returns a `DispatchDecision` keyed by stable generator IDs, or a typed `DispatchFailure`. The simulator snapshots decisions into JSON-friendly result values, checks authority and setpoint coverage, applies accepted setpoints, runs pandapower AC power flow, then reports realised values and constraint violations. Policy code never receives the live network.

The included policy uses available renewable output first, then fills conventional capacity in increasing marginal-cost order. It is a heuristic, not OPF or a feasibility guarantee; the independent AC evaluator reports voltage and line-limit issues. The slack source remains visible in results.

## Install and run

Use Python 3.11–3.14. Python 3.15 is currently excluded because native dependencies in this stack (the reported PyO3 0.27.1 build, via `cachebox`) support Python only through 3.14. Use Python 3.12 or 3.13 for the broadest prebuilt-wheel availability. From this directory:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
streamlit run app.py
```

For the validated macOS Apple Silicon / Python 3.13 environment, install with the committed resolved constraints:

```sh
python -m pip install -e '.[dev]' -c constraints-macos-arm64-py313.txt
```

That constraints file captures the working package versions for this platform and interpreter. Regenerate it from a clean Python 3.13 arm64 environment, review the full diff, and rerun the project checks before updating it; it is not a cross-platform lock file.

The app loads pandapower's bundled IEEE 14 bus network. This initial vertical slice provides manual case edits, dispatch DTOs, a replaceable heuristic, real AC power flow, voltage/branch inspection, and explicit evaluation status.

## Architecture and milestones

1. Runnable AC evaluation and dispatch contract (current): `controllers/`, `powersim/contracts.py`, `powersim/simulator.py`, Streamlit UI.
2. Time-series scenarios, AC-OPF adapter, persistence/replay and comparison.
3. Validated JSON/MATPOWER imports and profile ingestion.
4. Dataset generation and learned dispatch evaluation.

## Assumptions and limitations

The model is balanced, positive-sequence, quasi-static AC power flow. The current observation is a single-step summary and the bundled case has no renewable profile. The evaluator checks bus voltage and line loading, while transformer limits and full generator P/Q/slack-bound enforcement are not yet implemented. There is not yet a time-series runner, AC-OPF, import pipeline, external-process protocol, dataset workflow, or replay store. The Python package range is broad; lock a platform-specific environment for published experiments.

## Geographic network view

The optional geographic layer is defined in `powersim/geography.py`; it consumes a separate GeoJSON mapping with explicit `EPSG:4326`, provenance, geometry kind, license, and stable asset IDs. The dashboard includes an illustrative synthetic feeder mapping at `examples/geography/synthetic_feeder.geojson`, clearly separate from the IEEE 14 electrical case. It never treats proximity as an electrical/service association, and unknown results are shown as unknown. See [docs/geospatial.md](docs/geospatial.md) for format, attribution, internet requirements, limitations, and the SMART-DS/OpenDSS feasibility policy. At present time sliders, controller comparisons, and SMART-DS conversion are not implemented.
