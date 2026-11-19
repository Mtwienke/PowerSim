# Seed 2027: IEEE-30 + Germany-30, eight workers

**Completed 2026-11-19: 114/114 stages, 84/84 evaluations in 32 min 29 s.**
All **324 required stage output hashes** were verified afterward; no mismatches.
No stage failed. All PF audits and fallbacks on reference-valid evaluation
samples returned successfully; this does not imply the learned vectors were
feasible.

**Main finding:** baseline has lower pooled MAE/RMSE than full L0–L5 on both
benchmarks. IEEE-30 has high 1 MW accuracy but almost no strictly feasible raw
outputs. On Germany-30, **none of the seven raw networks passes the complete
1 MW source-error criterion or the configured feasibility checks**. Its
successful hybrid cases are entirely conventional OPF fallbacks.

## Reproducibility and scope

- Run root: `artifacts/full-suite-seed2027-20261119T021858Z/`.
- Frozen code: `599ffd2917b38168688adf19334a3c1336f65c8a`.
- Benchmarks: IEEE-30 (`case30`) and [Germany-30](germany30.md), DC physics.
- Seed **2027** for generation, neural initialization/training, and decoder.
  It changes installation placement as well as loads/weather and ML randomness.
- Per benchmark: 10,000 train, 1,000 validation, 2,000 test **attempts** per
  renewable regime. Total **34,000 attempts**, no hidden replacement of failures.
- Seven methods per grid: L0 baseline, full L0–L5, and five leave-one-out
  variants (without L1, L2, L3, L4, or L5). Enabled weights remain 1.
- Networks: 200–200–200, tanh, Adam 0.001, batch 64, 100 epochs; best validation
  generation-MSE checkpoint, not a test-selected checkpoint.
- Decoders: frozen OPF network, hidden width 128, batch 64, 100 epochs,
  validation-only 95th-percentile normalized reconstruction-MSE calibration.
- **Eight processes** for dataset and evaluation, one numerical/ML thread
  per worker. Sequential models/stages. All three test regimes were evaluated
  with and without decoder fallback.
- macOS arm64 / Apple M2 Pro, Python 3.13.5, pandapower 3.5.6, PyTorch 2.15.0,
  NumPy 2.4.6, SciPy 1.18.1, pandas 2.3.3, no Numba.

This uses the documented 0.9 branch-rating scale and assumed nameplate
penetration interpretation. German source topology is open-data-derived, but
demand, dispatch capacity, cost curves and renewable operating inputs are
illustrative, **not historical German grid measurements**. The original paper's
M1–M5 combinations remain unverified. No test-dependent tuning or follow-up
retraining was performed.

## Reference coverage

| Grid | Split | Attempts | Valid labels | Failed | Failure rate |
|---|---|---:|---:|---:|---:|
| IEEE-30 | Train | 10,000 | 9,425 | 575 | 5.75% |
| IEEE-30 | Validation | 1,000 | 929 | 71 | 7.10% |
| IEEE-30 | IID, 35.58% | 2,000 | 1,888 | 112 | 5.60% |
| IEEE-30 | OOD, 40.71% | 2,000 | 1,881 | 119 | 5.95% |
| IEEE-30 | OOD, 47.24% | 2,000 | 1,885 | 115 | 5.75% |
| Germany-30 | Train | 10,000 | 8,944 | 1,056 | 10.56% |
| Germany-30 | Validation | 1,000 | 899 | 101 | 10.10% |
| Germany-30 | IID, 35.58% | 2,000 | 1,794 | 206 | 10.30% |
| Germany-30 | OOD, 40.71% | 2,000 | 1,683 | 317 | 15.85% |
| Germany-30 | OOD, 47.24% | 2,000 | 1,431 | 569 | **28.45%** |

30,759 valid labels were retained from 34,000 attempts. Failed-reference
records remain in JSONL; nonconvergence is **not proof of infeasibility**.
The particularly high German OOD failure rate needs independent numerical and
feasibility diagnosis. Error metrics below are conditional on retained
reference-valid samples, not all attempted conditions.

Training cases with active branches: 1,694/9,425 (IEEE-30), 8,269/8,944
(Germany-30). Active source-bound rows comprise 0.073% and 33.703%,
respectively. All retained sensitivities passed the implemented availability
checks; those checks alone do not establish accuracy near active-set transitions.

## Pooled results across three test regimes

Pool each grid separately: **5,654** IEEE-30 and **4,908** Germany-30
reference-valid cases. MAE includes every conventional source including slack;
RMSE is pooled from squared errors with scenario-count weighting.
Success means **all source errors <= 1 MW in a scenario**, not mean error <=
1 MW. Feasibility separately enforces raw balance (0.0001 MW), source/branch
bounds (0.001 MW), output coverage and explicit slack discrepancy checks.

| Grid | Output | MAE MW | RMSE MW | Max error MW | 1 MW success | Feasible |
|---|---|---:|---:|---:|---:|---:|
| IEEE-30 | Baseline raw | 0.1237 | 0.3557 | 19.6111 | 95.048% | 0.018% |
| IEEE-30 | Baseline hybrid | 0.1084 | 0.3077 | 14.0143 | 95.914% | 7.181% |
| IEEE-30 | Full guided raw | 0.1266 | 0.3923 | 21.1757 | 93.774% | 0.035% |
| IEEE-30 | Full guided hybrid | 0.1122 | 0.3493 | 15.2770 | 94.429% | 8.030% |
| Germany-30 | Baseline raw | 4.4276 | 8.9910 | 499.8028 | 0% | 0% |
| Germany-30 | Baseline hybrid | 3.7576 | 7.1259 | 167.7861 | 5.990% | 5.990% |
| Germany-30 | Full guided raw | 4.6383 | 9.0117 | 582.2882 | 0% | 0% |
| Germany-30 | Full guided hybrid | 3.9899 | 7.2092 | 149.1484 | 5.807% | 5.807% |

Germany's MW system scale and source count differ from IEEE-30; their absolute
errors should not be interpreted as a like-for-like cross-grid ranking.
Nevertheless its accepted outputs clearly fail the requested absolute 1 MW
criterion. The raw baseline's mean residual is 39–42 MW per German test
regime, despite the decoder accepting approximately 93–95% of samples.
Acceptance is not a balance, feasibility or accuracy guarantee.

## Ablations (pooled MAE)

Descriptive loss names, not verified paper M1–M5 labels:

| Method | IEEE raw MW | IEEE hybrid MW | German raw MW | German hybrid MW |
|---|---:|---:|---:|---:|
| Baseline L0 | 0.1237 | 0.1084 | 4.4276 | 3.7576 |
| Full L0–L5 | 0.1266 | 0.1122 | 4.6383 | 3.9899 |
| Without L1 | 0.1151 | 0.1022 | 4.0241 | 3.5297 |
| Without L2 | 0.1421 | 0.1244 | 4.0136 | 3.5325 |
| Without L3 | 0.1125 | 0.1006 | 4.2170 | 3.6384 |
| Without L4 | 0.1403 | 0.1255 | 3.6866 | 3.1468 |
| Without L5 | 0.1253 | 0.1081 | 3.8678 | 3.3087 |

These are descriptive held-out results, not a basis for silently selecting a new
loss recipe or retuning to the same tests. All German raw ablations still
have 0% 1 MW success and 0% strict feasibility.

## Fallback and coverage caveats

Pooled fallback rates were 7.163% / 7.994% for IEEE baseline/full guided,
and 5.990% / 5.807% for German baseline/full guided.
German hybrid successes and feasible outputs are exactly its fallback outputs
(294 baseline and 285 full guided over 4,908 retained cases). None of the
accepted learned cases passes either complete criterion.

In the highest German regime, accepted-subset MAE is 4.074 MW (baseline)
and 4.299 MW (full guided), with 0% complete 1 MW success. Out of all
2,000 **attempted** cases, conservative hybrid 1 MW success is only 5.10%
and 5.15%, respectively; excluded reference failures count as unsuccessful
in that additional denominator.

There were no PF/fallback failures on the evaluated reference-valid subset.
This does **not** assess fallback reliability on failed-reference samples.
German feasible-only hybrid objective gaps are 0% because feasible outputs
come from OPF; raw objective gaps are unavailable, not zero. DC voltage/Q
metrics remain unavailable. Constraint-violation counts are events rather
than necessarily distinct scenarios.

## Runtime

Start **02:20:33 UTC**, finish **02:53:03 UTC**, 2026-11-19:
**32 min 29 s wall time**. Recorded command timers total 32 min 26 s.

| Grid | Dataset generation | Seven trainings | Seven decoders | 42 evaluations |
|---|---:|---:|---:|---:|
| IEEE-30 | 44.05 s | 354.49 s | 45.21 s | 333.64 s |
| Germany-30 | 67.25 s | 714.11 s | 41.35 s | 345.87 s |

Evaluation reports measure parallel throughput including PF, startup,
scheduling and prediction JSONL writing. Single-scenario isolated-latency
fields are unavailable with eight workers. Training now accounts for more
time than evaluation. The previous suite's 7 h 43 min run is not a controlled
speedup denominator: code, seed and scope changed in addition to worker count.
See [controlled worker-count measurements](performance.md).

## Artifacts and next work

The run directory contains `suite_manifest.json`, `status.json`, `runner.log`,
source archive, per-stage logs and checkpoints, `summary.json` (84 reports),
`results.csv` (84 rows), and derived `pooled-results.json`.
The earlier seed-2026 run was left untouched. Preflight passed all 61 tests
and `pip check`; no simulation code was changed for this run.

Recommended scientific follow-up: independently classify the German
failed-reference cases, inspect training/validation sensitivity and loss scales,
and validate any stronger acceptance/repair mechanism separately from raw
predictions. Any tuning needs validation-only selection and a newly reserved
evaluation protocol. These models should not be treated as operationally safe
dispatch methods, and this run does not reproduce the publication's claimed
trustworthiness.
