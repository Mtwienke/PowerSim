# CPU scenario parallelism

Independent PF/OPF cases are the target, not PyTorch training: evaluations
accounted for 96.8% of timed stages in the previous full IEEE-30 suite.
`dataset --workers N` and `evaluate --workers N` use spawned processes with
one numerical-library thread per child and one PyTorch thread per evaluation
worker. The default `--workers 1` avoids startup cost and enables isolated
single-scenario timing. For comparable serial BLAS configuration, set the
thread limits in the shell before startup (example below).

Each worker prepares its own network / one load row per bus once, and replaces
nodal active load and all prescribed active setpoints on each case. The PF and
fallback OPF use separate scratch networks. Pandapower still runs the full PF
or OPF as appropriate; no optimizer is substituted for a raw prediction.
Results are emitted in original order and submitted work is bounded. RNG
draws come from the parent process. No solver convergence
criteria or label validation rules were loosened.

A hybrid evaluation audits raw separately and reuses that audit for unchanged
accepted results. **Use one decoder-enabled evaluation to report accuracy and
physics for both variants.** Run another raw-only evaluation only if a separate
isolated non-hybrid measurement is needed.

## Measured on Apple M2 Pro

Python 3.13.5, pandapower 3.5.6, PyTorch 2.15.0, without Numba. One run per
worker count (not a multi-run benchmark); 880 IEEE-30 dataset attempts and
the same 1,886 reference-valid highest-renewable OOD cases from the archived
full suite for every evaluation run. The archived baseline model and decoder
were reused. Evaluation wall time includes stage worker startup, PF audits,
fallback OPF, ordered collection and prediction JSONL writing. Dataset wall
time includes generation and NPZ writing. Neither column measures isolated
inference latency.

| Workers | Dataset wall time | Dataset speedup | Hybrid evaluation wall time | Evaluation speedup |
|---:|---:|---:|---:|---:|
| 1 | 16.32 s | 1.00× | 24.37 s | 1.00× |
| 2 | 9.99 s | 1.63× | 15.15 s | 1.61× |
| 4 | 6.36 s | 2.57× | 9.32 s | 2.61× |
| 8 | 4.85 s | 3.36× | 6.85 s | 3.56× |

Dataset and prediction NPZ arrays were exactly equal across worker counts,
as were split metadata, evaluation accuracy, feasibility and routing metrics.
Those evaluation metrics also exactly match the historical run of the same
model/split. The historical evaluation loop took 635.76 s; that run was in
a different session and is not a controlled old-versus-new timing comparison.
No new full experiment suite was launched.

On other machines start with 2–4 workers and measure; per-worker model memory,
process startup and CPU load can outweigh the benefit for small datasets.
Parallel reports leave `single_scenario_*` unavailable. Per-worker decision
time under contention, end-to-end evaluation throughput and isolated
OPF-model-only batch throughput are reported separately.

## Reproduce

Use fresh output directories. For the dataset numbers pass a JSON override
with `{"train_cases":240,"validation_cases":40,"test_cases":200}`. For the
evaluation use your chosen dataset, model and paired decoder, and keep them
unchanged when varying worker counts.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
powersim dataset artifacts/bench-data-w4 --config dataset-config.json --workers 4
powersim evaluate DATASET MODEL.pt artifacts/bench-eval-w4 \
  --decoder DECODER.pt --split ood_test_2 --workers 4
```

Local measurements are in `artifacts/performance-check-20261119/`. Worker
count is execution metadata and does not enter scenario or model identity.
