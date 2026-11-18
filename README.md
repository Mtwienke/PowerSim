# PowerSim

A reproducible, **paper-inspired** research environment for Gao et al.,
*A Trustable Data-Driven Optimal Power Flow Computational Method With Robust
Generalization Ability*, DOI
[10.1109/TNNLS.2024.3437741](https://doi.org/10.1109/TNNLS.2024.3437741).

**Defaults: IEEE 30 · DC physics · numerical DC-OPF.** IEEE 14 and AC remain
selectable. PF evaluates prescribed decisions; OPF generates references.
Learned predictions are never silently optimized.

## Install

Python 3.11–3.14 (3.12/3.13 recommended):

```sh
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'       # reference/data/dashboard, no PyTorch required
pip install -e '.[ml]'        # optional CPU training and learned inference
```

`constraints-macos-arm64-py313.txt` preserves the original basic-simulation
environment. The verified optional ML extension is
`constraints-ml-macos-arm64-py313.txt`; install using
`pip install -e '.[dev,ml]' -c constraints-ml-macos-arm64-py313.txt`.
Neither is a cross-platform lock. Every experiment records package versions.

## Reproduce a short CPU smoke experiment

Use fresh output directories; commands refuse to overwrite experiment artifacts.

```sh
powersim reference
powersim reference --physics ac --benchmark case14
powersim dataset artifacts/data --preset smoke --seed 2026
powersim train artifacts/data artifacts/baseline --method baseline --epochs 3 --hidden 16,16,16 --batch-size 16
powersim train artifacts/data artifacts/physics --method physics-guided --epochs 3 --hidden 16,16,16 --batch-size 16
powersim decoder artifacts/data artifacts/physics/model.pt artifacts/decoder --epochs 3 --hidden 16
powersim evaluate artifacts/data artifacts/baseline/model.pt artifacts/baseline-iid
powersim evaluate artifacts/data artifacts/physics/model.pt artifacts/hybrid-iid --decoder artifacts/decoder/decoder.pt
powersim evaluate artifacts/data artifacts/physics/model.pt artifacts/hybrid-ood --decoder artifacts/decoder/decoder.pt --split ood_test_2
pytest -q
streamlit run app.py
```

Without an installed entry point, use `python -m powersim.cli` in place of
`powersim`. Three epochs verify plumbing, **not useful predictive accuracy**.
The paper-inspired large preset is opt-in:
`powersim dataset artifacts/paper --preset paper` (10,000 train attempts,
1,000 validation, 2,000 test attempts per renewable regime).
No full experiment suite is launched automatically.
See [verified smoke results](docs/smoke-results.md) for measured outcomes and
their intentionally limited interpretation.
The explicitly requested [full 2026-11-18 suite](docs/full-experiment-results.md)
also completed: all 57 stages and 42 evaluations. Its results expose important
feasibility limitations and do not establish reproduction of the paper's claims.

## What is implemented

- Validated DC/AC configuration, immutable nodal policy inputs, actual polynomial
  costs, stable ordering, structural/configuration/scenario identities.
- Numerical OPF references including external-grid generation; separate PF
  audits with raw balance error, explicit slack adjustment, source and
  line/**transformer** constraints. DC voltage/Q checks are unavailable, not zero.
- Seeded load/wind/PV scenarios, documented conversion curves, failed-solve
  records, separate train/validation/IID/OOD splits, training-only normalization.
- Configurable three-hidden-layer baseline (default 200 neurons/layer) and
  individually configurable L0–L5 Jacobian losses; KKT reference sensitivities.
- Frozen-OPF decoder training, validation-only calibration, learned/OPF hybrid
  routing with explicit fallback failure and separate production rejection logs.
- Measured accuracy, physical feasibility, feasible-only objective gaps,
  acceptance/fallback rates, CPU latency and separately measured batch throughput.

The dashboard exposes physics, benchmark and numerical/heuristic methods.
Learned inference is disabled until matching trained model/decoder paths are
provided; match the artifact's branch limit scale (dataset default 0.9).
Trained **DC-only** inference is also available headlessly:

```sh
# scenario.json contains one array of ordered nodal net loads in MW
powersim infer artifacts/data artifacts/physics/model.pt artifacts/decoder/decoder.pt scenario.json --production-log artifacts/production/rejected.jsonl
```

Rejection logging never updates a model. Retraining requires an explicit new
dataset and `train` invocation. Decoder acceptance is **not** a proof of
feasibility or optimality.

## Protocol, configuration and limitations

See [docs/research.md](docs/research.md) for loss equations, normalization,
sensitivity regularity, metric denominators, CLI configuration, renewable
penetration definitions and remaining differences from the paper. Full-text
M1–M5 combinations were not verified: methods use descriptive names.
Conversion curves, validation size, activation and training settings are explicit
assumptions. The stress preset scales branch limits to 90% of bundled limits;
manifests report actual active sets and label failures rather than assuming
congestion. No repair/projection or automatic retraining is implemented.

Custom heuristics remain in `controllers/my_dispatch.py`, receiving frozen DTOs,
never a mutable pandapower network. Only the bundled balanced quasi-static
networks and polynomial conventional-source costs are supported by the research
adapter; it is not a production grid operating tool.

## Geographic view

The existing optional [geospatial layer](docs/geospatial.md) remains separate.
`examples/geography/synthetic_feeder.geojson` is illustrative, not IEEE network
geography. DC has no voltage-magnitude prediction; the map shows unavailable
values rather than manufacturing AC measurements.
