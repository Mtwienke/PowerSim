# Full IEEE-30 DC suite — 2026-11-18

**Completed: 57/57 stages, 42/42 evaluations.** All 157 required output files
were subsequently checked against the runner's recorded SHA256 hashes; no
mismatches. No stage, PF audit or reference-valid-case fallback failed.

This is a **single-seed, paper-inspired computational experiment**, not an exact
reproduction or a demonstration of the paper's robustness claims. In particular,
**the full L0–L5 network did not improve average accuracy over the baseline, and
its raw predictions did not meet the configured feasibility checks.**

## Provenance and settings

- Run: `artifacts/full-suite-20261118T044204Z/` (ignored experiment artifacts).
- Archived source: commit `c98947a3724d6dcd5974889d4605c233e324d35c`.
- IEEE 30, DC, bundled polynomial costs, branch limits scaled to 90%.
- Data/training/decoder seed: 2026; separate training/validation/IID/OOD streams.
- Train attempts: 10,000; validation attempts: 1,000; test attempts: 2,000 per
  renewable regime (35.58%, 40.71%, 47.24% assumed nameplate penetration).
- Seven methods: L0 baseline, full L0–L5, and full loss without each of L1–L5.
- OPF networks: three 200-neuron hidden layers, tanh, linear output, Adam 0.001,
  100 epochs, batch 64. Equal enabled-loss weights are an explicit assumption.
  Best generation-MSE checkpoint selected using validation labels only.
- Decoder: frozen OPF network, hidden width 128, 100 epochs, batch 64,
  normalized-input reconstruction MSE, validation-only 95th-percentile threshold.
- CPU, sequential stages, one numerical-library thread; no Numba.
- Python 3.13.5, pandapower 3.5.6, PyTorch 2.15.0, NumPy 2.4.6, SciPy 1.18.1,
  pandas 2.3.3; macOS arm64. See the committed ML constraints.

The runner used an archived source tree and recorded configuration, versions,
commands, source/output hashes and individual attempts. No test-dependent
retraining, threshold adjustment or hyperparameter search was performed.

## Data coverage and constraints

| Split | Attempts | Valid labels | Failed solves | Cases with active branches |
|---|---:|---:|---:|---:|
| Training | 10,000 | 9,296 | 704 | 1,690 |
| Validation | 1,000 | 942 | 58 | 178 |
| IID, 35.58% | 2,000 | 1,871 | 129 | 344 |
| OOD, 40.71% | 2,000 | 1,875 | 125 | 369 |
| OOD, 47.24% | 2,000 | 1,886 | 114 | 396 |
| **Total** | **17,000** | **15,870** | **1,130** | |

Failures (6.65%) remain in JSONL and were not turned into labels or silently
resampled. Solver nonconvergence does not establish mathematical infeasibility;
these cases need independent feasibility/numerical diagnosis.

All retained local KKT sensitivities were marked available by the configured
regularity checks. Training had 17 active source-bound rows in 16 cases, so L3
was exercised but very sparsely. Branch activity was materially more common.

## Pooled accuracy and feasibility

The following metrics pool the **5,632 reference-valid test scenarios** across
the three regimes, weighted by scenario count. RMSE is pooled from squared
errors, not averaged from per-regime RMSEs. Every source, including slack, is
included.

Success at 1 MW means complete output with **maximum absolute error across all
sources <= 1 MW**. It is not a feasibility criterion.

| Output | MAE MW | RMSE MW | Max error MW | 1 MW success | Feasible |
|---|---:|---:|---:|---:|---:|
| Baseline raw | 0.1114 | 0.3120 | 14.7209 | 96.005% | 0.036% |
| Baseline hybrid | 0.0979 | 0.2834 | 14.7209 | 96.680% | 7.386% |
| Full guided raw | 0.2108 | 0.4869 | 13.7908 | 88.920% | 0% |
| Full guided hybrid | 0.1880 | 0.4341 | 13.7908 | 90.163% | 6.357% |

Baseline has lower MAE/RMSE and higher threshold success here, although the full
guided network's worst source error is slightly smaller. One seed cannot
establish statistical superiority or a generalization theorem.

Full guided hybrid feasibility equals its fallback coverage: **none of its
accepted learned predictions passed the complete configured checks**. Baseline
had only two feasible raw predictions across all test regimes. Thus these
models are not suitable as unguarded operational solutions.

## Highest-renewable regime and ablations

47.24% nameplate penetration; 1,886 reference-valid scenarios:

| Method | Raw MAE MW | Raw 1 MW success | Hybrid MAE MW | Hybrid 1 MW success | Fallback |
|---|---:|---:|---:|---:|---:|
| Baseline L0 | 0.1247 | 95.334% | 0.1050 | 96.235% | 10.021% |
| Full L0–L5 | 0.2311 | 86.797% | 0.1992 | 89.024% | 9.067% |
| Without L1 | 0.1363 | 95.546% | 0.1177 | 96.341% | 10.657% |
| Without L2 | 0.2336 | 86.744% | 0.2011 | 88.918% | 9.597% |
| Without L3 | 0.2427 | 88.547% | 0.1969 | 91.463% | 10.233% |
| Without L4 | 0.2332 | 84.624% | 0.2100 | 85.790% | 9.544% |
| Without L5 | 0.2817 | 81.336% | 0.2302 | 85.896% | 9.703% |

Removing L1 brought average accuracy closer to baseline in this configuration.
That is a diagnostic observation, not a test-selected new training recipe.
For context, the full model's final-epoch training terms included L1 = 20.31 and
L0 = 0.250 (different units, both given unit weights). Future investigations
should audit sensitivity magnitudes and loss scaling using training/validation
data, not optimize these choices against the already inspected test results.

## Decoder, violations and objective gaps

In the highest-renewable regime:

| Metric | Baseline raw | Baseline hybrid | Full guided raw | Full guided hybrid |
|---|---:|---:|---:|---:|
| Mean absolute balance residual MW | 0.2581 | 0.1943 | 0.5963 | 0.4698 |
| Branch violation events | 311 | 289 | 398 | 380 |
| Maximum branch overload MW | 0.9343 | 0.7825 | 2.1852 | 2.1852 |
| Source-bound violation events | 1 | 1 | 0 | 0 |
| Feasible scenarios | 1 | 190 | 0 | 171 |

Branch counts are constraint-violation events, not necessarily distinct
scenarios. DC voltage and reactive-power metrics remain unavailable.
Feasibility uses the documented raw-balance tolerance of 0.0001 MW and
P/branch tolerance of 0.001 MW; the looser 1 MW accuracy threshold cannot imply
those constraints.

Highest-regime decoder acceptance was 89.979% for baseline and 90.933% for full
guided. Accepted-subset MAE/success were 0.1167 MW / 95.816% and
0.2190 MW / 87.930%, respectively. Reconstruction selection removed some hard
cases but did not certify accepted predictions.

No fallback failed **on the reference-valid evaluation subset**. This does not
measure fallback reliability on the 1,130 excluded failed-reference scenarios.
Counting all 2,000 requested highest-regime cases conservatively, hybrid success
was 90.75% for baseline and 83.95% for full guided.

Objective gaps are reported only for feasible predictions. The full guided raw
gap is unavailable; its hybrid gap is 0% over the 171 successful OPF fallbacks,
not a claim that its learned outputs are optimal. Baseline hybrid mean gap is
approximately 0.00000063% over 190 feasible outputs, almost all OPF fallbacks.

## Measured runtime

Started **04:46:53 UTC**, finished **12:30:19 UTC** on 2026-11-18:
**7 h 43 min 25 s wall time**. Recorded subprocess stage timers sum to
**6 h 55 min 42 s**. The remaining wall-clock difference is not attributed to
algorithmic computation; suspension/clock gaps or other unmeasured intervals
were not independently diagnosed.

Pooled mean single-scenario decision latency:

| Method | Raw | Hybrid, including decoder/fallback |
|---|---:|---:|
| Baseline | 0.112 ms | 21.210 ms |
| Full guided | 0.124 ms | 18.247 ms |

These decision timings exclude model loading and independent PF audits.
Highest-regime hybrid p95 latency was 278.5 ms (baseline) and 275.0 ms (full).
The full guided network's recorded 100-epoch training stage took 63.0 s versus
11.0 s for baseline. Most suite time was in repeated physical evaluation and
network construction. No FLOP-derived runtime or commercial-solver speedup is
claimed; model-only batch throughput is separate in the raw reports.

## Outputs and next scientific checks

Within the run directory:

- `status.json`: completed stage ledger and checksums.
- `suite_manifest.json`, `source.tar`, `source/`: configuration and frozen code.
- `summary.json`: all 42 complete evaluation reports.
- `results.csv`: flat metrics for all 42 evaluations.
- `stages/`: complete data, checkpoint, decoder and prediction artifacts.
- `logs/`, `runner.log`: individual-stage output and execution timestamps.

Next work should independently classify reference failures, verify extreme
training/validation sensitivities, investigate validation-selected loss scaling,
and repeat with independent seeds and newly reserved evaluation data.
Any feasibility repair or stronger routing gate would be a separate method
whose raw and repaired outputs must remain distinct. No such follow-up
experiment or model change was performed automatically.

See [research protocol](research.md) for the configurable assumptions and
remaining differences from the publication.
