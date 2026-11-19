# PowerSim

A reproducible, **paper-inspired** research environment for Gao et al.,
*A Trustable Data-Driven Optimal Power Flow Computational Method With Robust
Generalization Ability*, DOI
[10.1109/TNNLS.2024.3437741](https://doi.org/10.1109/TNNLS.2024.3437741).

**Defaults: IEEE 30 · DC physics · numerical DC-OPF.** IEEE 14 and AC remain
selectable. The [Germany-30 benchmark](docs/germany30.md) is a DC-only reduced
transmission model based on open PyPSA-Eur data. PF evaluates prescribed
decisions; OPF generates references.
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
The [seed-2027 IEEE/German suite](docs/full-experiment-seed2027.md) completed
all 114 stages and 84 evaluations using eight workers in 32m 29s, with separate
coverage, accuracy and feasibility reporting for each grid.

## Full-paper-aligned training pilot

The [full-text audit](docs/paper-protocol.md) confirms a **1,000-epoch M1 baseline
followed by 200 warm-started epochs**, not the independent 100-epoch trainings
used in the historical suites. Preview a pilot on an existing dataset:

```sh
powersim pilot DATASET artifacts/gao-pilot --plan
# Explicitly launch training/validation only:
powersim pilot DATASET artifacts/gao-pilot
```

The pilot shares one M1 initializer across M1-continued/M2/M4/M5, uses learning
rate 0.01, records provenance, keeps best/final checkpoints and supports
hash-checked `--resume`. It does not regenerate scenarios or evaluate held-out
data. Add `--learning-rates 0.01,0.001` for a declared stability control.
Generic training defaults and existing model/decoder files remain compatible.
The completed [IEEE-30 pilot](docs/ieee30-pilot-seed2027.md) follows this schedule
but **does not reproduce the paper's accuracy or method ranking**; its report
separates fitting failure, data-definition uncertainties and decoder coverage.
The subsequent [public-source audit](docs/gao-source-audit.md) found concrete
numerical differences **and conflicts between the examples and final paper**.
It preserves both specifications rather than silently mixing their settings.

## Source-informed full reconstruction suite

The user-authorized [reconstruction protocol](docs/gao-reconstruction-suite.md)
proceeds without the missing full project. It keeps **Germany excluded**, uses
explicit 185/230/300 MW capacity and per-unit/ReLU hypotheses, and separately
reports source-informed, journal-equation and literal-example numerical variants.

```sh
python -m powersim.reproduction artifacts/gao-reconstruction --plan
python -m powersim.reproduction artifacts/gao-reconstruction
# After an interrupted driver and all its children have stopped:
python -m powersim.reproduction artifacts/gao-reconstruction --resume
# After completion: verify hashes and generate descriptive JSON/Markdown reports.
python -m powersim.reproduction_analysis artifacts/gao-reconstruction
```

This opt-in workload freezes committed source: **18 trainings, 18 decoders,
165 evaluations**, including new IID/OOD test streams, decoder/fallback
comparisons and 1 MW FGSM with separately re-solved physical labels.
`--smoke` runs the same workflow with tiny budgets for plumbing tests, not
accuracy tuning. Existing datasets, checkpoints and generic defaults are kept.
`M3-example` is explicitly the Gao branch-penalty comparator, not a claim of
complete external DeepOPF reproduction.

## What is implemented

- Validated DC/AC configuration, immutable nodal policy inputs, actual polynomial
  costs, stable ordering, structural/configuration/scenario identities.
- Numerical OPF references including external-grid generation; separate PF
  audits with raw balance error, explicit slack adjustment, source and
  line/**transformer** constraints. DC voltage/Q checks are unavailable, not zero.
- Seeded load/wind/PV scenarios, documented conversion curves, failed-solve
  records, separate train/validation/IID/OOD splits, training-only normalization.
- Configurable three-hidden-layer baseline (default 200 neurons/layer) and
  individually configurable L0–L5 Jacobian losses; KKT reference sensitivities,
  checked weights-only warm starts and a shared-M1 paper-schedule pilot.
- Frozen-OPF decoder training, validation-only calibration, learned/OPF hybrid
  routing with explicit fallback failure and separate production rejection logs.
- Measured accuracy, physical feasibility, feasible-only objective gaps,
  individual-source and joint-scenario accuracy, accepted-subset coverage,
  reconstruction selection curves, CPU latency and separate batch throughput.

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

## Use multiple cores or inspect one solution

```sh
powersim dataset artifacts/germany-data --benchmark germany30 --workers 4
powersim evaluate DATASET MODEL.pt artifacts/parallel-eval --workers 4 --decoder DECODER.pt
powersim plot-solution artifacts/germany-solution.html --benchmark germany30
```

`dataset` and `evaluate` default to one worker. `--workers 4` uses independent
processes, with deterministic scenario ordering. A decoder evaluation already
reports raw-model **and** hybrid results, so it need not be preceded by a second
raw evaluation unless a separate model-only latency comparison is wanted.
Worker throughput includes audits, startup and scheduling; it is not
single-scenario latency. See [runtime measurements](docs/performance.md).

The dashboard's **Solution Flow** graph shows bus injections and demand, the
slack, flow arrows, branch loading, and transformers. Learned and fallback
views are distinct. `plot-solution` makes an offline interactive HTML report
or replays a saved dataset reference without re-solving it:

```sh
powersim plot-solution artifacts/sample.html --dataset artifacts/germany-data --split train --sample 0
```

## Protocol, configuration and limitations

See [docs/research.md](docs/research.md) for loss equations, normalization,
sensitivity regularity, metric denominators, CLI configuration, renewable
penetration definitions and remaining differences from the paper. Full-text
M1/M2/M4/M5 objectives are now verified; M3 is DeepOPF, not an implemented
leave-one-out ablation. See the [paper protocol](docs/paper-protocol.md).
Conversion curves, validation size, activation and training settings are explicit
assumptions. The stress preset scales branch limits to 90% of bundled limits;
manifests report actual active sets and label failures rather than assuming
congestion. No repair/projection or automatic retraining is implemented.

Custom heuristics remain in `controllers/my_dispatch.py`, receiving frozen DTOs,
never a mutable pandapower network. The supported benchmarks are bundled
balanced quasi-static models with polynomial conventional-source costs; the
research adapter is not a production grid operating tool.

## Geographic view

The existing optional [geospatial layer](docs/geospatial.md) remains separate.
`examples/geography/synthetic_feeder.geojson` is illustrative, not IEEE network
geography. DC has no voltage-magnitude prediction; the map shows unavailable
values rather than manufacturing AC measurements.
