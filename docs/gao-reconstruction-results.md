# IEEE-30 reconstruction: completed full-suite results

**Completed 2026-11-19. The source-informed reconstruction substantially improves
on the earlier pilot, but does not reproduce the paper's M5 advantage or provide
feasible raw dispatch.**

Primary M5 reaches **89.375%** highest-regime individual-source accuracy, versus
**31.01%** in the [earlier pilot](ieee30-pilot-seed2027.md) and **97.94%** in the
paper. Primary M1 reaches
**90.167%**, so M5 remains **0.792 percentage points below M1**, rather than
showing the paper's **28.63-point gain**.

The [predeclared protocol](gao-reconstruction-suite.md) and
[pinned-source audit](gao-source-audit.md) distinguish evidence from hypotheses.
The primary variant remains `source-informed`; neither controls nor saved
checkpoints are promoted after seeing results. **Germany was excluded, no
authors were contacted, and no post-result retraining or dispatch repair
occurred.** Historical datasets, checkpoints and negative reports are retained.

## Execution and provenance

- Run root: `artifacts/gao-reconstruction-full-20261119T133302Z/`.
- Frozen source: **`e83e0852af414b4b3aa4f2d5357c218d155f8d0a`**.
- Pinned public examples: `e775ce3a24af0408aed9c02236abd862707cbae0`.
- Start **13:33:59 UTC**, finish **14:43:03 UTC**, 2026-11-19;
  **69 min 03.75 s** wall time.
- **239 stages**, all on their first stage attempt: one dataset, **18 model
  trainings, 18 decoders**, one calibration, **36 attack datasets** and
  **165 evaluations**. Scenario-level reference failures are separate below.
- All **829 required output hashes**, the frozen source files and source archive
  passed post-completion verification. All continuations' parent hashes match
  their own variant's final epoch-1000 M1.
- No evaluation PF audits or hybrid fallbacks failed. A converged PF audit does
  **not** imply that the original learned proposal satisfies constraints.
- Seed **2027** for placement/training/validation/ML; reserved test seed
  **20281121**. Three 200-unit ReLU layers, 100 MW per-unit scaling, unscaled
  branch ratings and successful-label collection follow the frozen protocol.
- Environment: Python **3.13.5**, PyTorch **2.15.0**, NumPy **2.4.6**,
  SciPy **1.18.1**, pandas **2.3.3**, pandapower **3.5.6**; CPU training and
  eight independent reference/evaluation workers.

The run contains `suite_manifest.json`, `status.json`, `training_summary.json`,
`summary.json`, `results.csv`, stage logs, histories, checkpoints, attack vectors
and compressed per-case physical audits. The frozen analyzer produced
`analysis/analysis.json` and `analysis/report.md` without updating models.
Descriptive target-shift and renewable statistics below are direct reductions
of saved arrays/JSONL, not additional training or solver runs.

## Data and reference coverage

| Split | Attempts consumed | Valid labels | Reference-active-branch cases |
|---|---:|---:|---:|
| Training | 10,099 | 10,000 | 1,116 |
| Validation | 1,015 | 1,000 | 124 |
| IID | 2,016 | 2,000 | 224 |
| OOD1 | 2,037 | 2,000 | 327 |
| OOD2 | 2,057 | 2,000 | 543 |

Every consumed failed reference attempt is persisted. Reported prediction
accuracy is conditional on valid references; resampling to a valid-case target
changes that conditional distribution. These solver failures are not proofs of
mathematical infeasibility. Counting excluded attempts as unsuccessful instead,
primary M5 source accuracy is **97.718 / 94.502 / 86.898%** on IID/OOD1/OOD2.

All **10,000** training Jacobians are available. The public-example L1 predicate
retains **9,892**, masking **108** only from sensitivity learning. All 10,000
supervised L0 cases remain; no Jacobian labels were clipped.

The capacity hypothesis is `R / (335 MW + R)`, with **185/230/300 MW**
nameplates. Sites are `bus:23` and `bus:19` for wind, `bus:20` and `bus:6` for
PV, each allocated one quarter of installed capacity. These are stored,
seeded choices, not recovered author settings. Realized generation is not the
printed installed-capacity penetration:

| Valid test split | Printed regime | Mean renewable MW | Mean renewable / gross load | Realized share p05 / median / p95 |
|---|---:|---:|---:|---:|
| IID | 35.58% | 43.384 | 23.045% | 10.629 / 22.445 / 37.671% |
| OOD1 | 40.71% | 53.974 | 28.646% | 13.716 / 27.586 / 46.457% |
| OOD2 | 47.24% | 69.176 | 36.684% | 16.508 / 36.051 / 58.484% |

The missing generator leaves these operating distributions unverified.
Matching rounded regime labels cannot establish matching test difficulty.

## Primary fixed-final raw accuracy

Strict **individual-source absolute error < 1 MW**, over six source outputs
including slack per reference-valid scenario. This is the inferred paper
aggregation, not a feasibility or joint-scenario metric.

| Method | IID 35.58% | OOD1 40.71% | OOD2 47.24% |
|---|---:|---:|---:|
| M1 | 99.233% | 97.500% | 90.167% |
| M1-continued | 99.058% | 97.275% | 88.450% |
| M2 | 99.225% | 97.892% | 91.958% |
| M3-example | 98.492% | 96.900% | 88.658% |
| M4 | 99.250% | 97.850% | 92.508% |
| M5 | 98.500% | 96.250% | 89.375% |

`M3-example` implements the visible supervised plus normalized branch-flow
penalty objective. It is **not** a complete external DeepOPF reproduction and
must not be treated as a like-for-like paper M3 result.

### Comparison with the paper and earlier pilot

Paper: Gao et al., DOI
[10.1109/TNNLS.2024.3437741](https://doi.org/10.1109/TNNLS.2024.3437741),
p. 8055, Table II.

| Method | Paper IID / OOD1 / OOD2 | New primary OOD2 minus paper | Earlier pilot OOD2 |
|---|---:|---:|---:|
| M1 | 99.94 / 89.59 / 69.31% | +20.857 pp | 82.46% |
| M2 | 99.89 / 96.18 / 88.95% | +3.008 pp | 44.16% |
| M4 | 99.96 / 99.15 / 92.66% | -0.152 pp | 38.61% |
| M5 | 99.92 / 99.16 / 97.94% | -8.565 pp | 31.01% |

- M5 gains about **58.37 pp** over the failed pilot, but this bundles changes
  to data, activation, units, losses, masks, optimizer and sampling. It is not
  an isolated causal ablation or a same-test-set improvement estimate.
- Primary M4 improves on M1 by **2.342 pp** at OOD2 and approaches the paper's
  absolute M4 accuracy, but M5 is **3.133 pp below M4**.
- Primary M1's IID-to-OOD2 drop is **9.067 pp**, versus **30.63 pp** in the
  paper. Our baseline's much easier generalization is itself a reproduction
  mismatch; high absolute percentages alone are not evidence of fidelity.
- M1-continued loses accuracy relative to M1. M5 does exceed this extra-epoch
  control at OOD2, but not M1, M2 or M4. The published method ranking is absent.

### All predeclared numerical variants

| Variant | Method | IID | OOD1 | OOD2 |
|---|---|---:|---:|---:|
| journal-equations | M1 | 97.650% | 94.308% | 83.150% |
| journal-equations | M1-continued | 97.842% | 93.958% | 81.225% |
| journal-equations | M2 | 95.925% | 92.092% | 81.467% |
| journal-equations | M3-example | 95.725% | 90.725% | 77.917% |
| journal-equations | M4 | 92.542% | 88.208% | 78.292% |
| journal-equations | M5 | 96.483% | 92.292% | 81.308% |
| public-example | M1 | 99.233% | 97.500% | 90.167% |
| public-example | M1-continued | 99.058% | 97.275% | 88.450% |
| public-example | M2 | 99.225% | 97.892% | 91.958% |
| public-example | M3-example | 98.492% | 96.900% | 88.658% |
| public-example | M4 | 98.458% | 97.325% | 92.775% |
| public-example | M5 | 98.858% | 96.633% | 89.483% |

Source-informed and public-example M1/M1-continued/M2/M3-example have identical
raw results, as expected from identical optimized objectives and seeds. They
are **not independent replications**. Neither control reproduces M5's claimed
advantage over M1. No variant is substituted for the primary.

## Fitting and checkpoint sensitivity

Primary fixed-final fitting and the predeclared saved-best-validation arm:

| Method | Final training accuracy | Final validation accuracy | Best-validation epoch | Best-validation IID / OOD1 / OOD2 |
|---|---:|---:|---:|---:|
| M1 | 99.470% | 99.167% | 666 | 99.367 / 98.225 / 91.067% |
| M1-continued | 99.288% | 99.233% | 10 | 99.383 / 97.992 / 89.875% |
| M2 | 99.010% | 99.133% | 1 | 99.367 / 97.875 / 90.158% |
| M3-example | 98.807% | 98.533% | 106 | 99.550 / 98.050 / 89.983% |
| M4 | 98.995% | 99.033% | 2 | 99.317 / 97.750 / 89.567% |
| M5 | 98.095% | 98.050% | 1 | 98.333 / 95.958 / 86.425% |

Best means minimum validation MSE during the same training, not maximum test
accuracy. Continuation epochs are counted from their own start; every parent
remains the final epoch-1000 M1, not the saved best M1.

The earlier pilot's severe fitting collapse is greatly reduced, but M5 still
worsens training and validation fit relative to M1/M4. Saved-best M5 is worse
than final M5 at OOD2, so switching to that predeclared checkpoint does not
resolve the discrepancy. Last-epoch primary M5 minibatch losses include
L0 **0.001436** and L5 **0.008808**; these differently defined objective values
are not gradient/update ratios or an isolated explanation of the accuracy loss.

## Accuracy is not physical feasibility

Pooled across **6,000** reference-valid primary test scenarios:

| Method | Joint all-source <1 MW | MAE MW | RMSE MW | Mean absolute raw balance MW | Strict feasible |
|---|---:|---:|---:|---:|---:|
| M1 | 92.283% | 0.185214 | 0.806301 | 0.437564 | 0.000% |
| M1-continued | 90.167% | 0.240762 | 0.853011 | 0.772173 | 0.017% |
| M2 | 93.417% | 0.185188 | 1.023229 | 0.480957 | 0.083% |
| M3-example | 87.917% | 0.227079 | 0.860413 | 0.632918 | 0.067% |
| M4 | 92.217% | 0.163805 | 0.908254 | 0.122763 | 0.200% |
| M5 | 91.783% | 0.276290 | 1.331224 | 0.192251 | 0.000% |

Strict checks retain **0.0001 MW** raw balance/slack-adjustment tolerance and
**0.001 MW** source/branch-bound tolerance. Balance-pattern derivative learning
does not impose an exact absolute-balance intercept, and approximate Jacobian
matching does not guarantee constraints. No predictions are projected or
silently repaired. Feasible-only objective gaps for primary M1/M5 are
unavailable, not zero.

At OOD2, the separate branch metrics are:

| Method | Strict branch-scenario events | Scenarios with a strict branch violation | Source-like events beyond RATE_A + 1 MW |
|---|---:|---:|---:|
| M1 | 476 | 437 | 99 |
| M2 | 201 | 178 | 38 |
| M3-example | 458 | 434 | 89 |
| M4 | 248 | 231 | 56 |
| M5 | 438 | 378 | 220 |

Branch flows are audited after explicit PF slack rebalance, while strict
feasibility additionally checks the original proposal's imbalance. The
source-like column counts **branch × scenario events**, not distinct branches.
M5's mean excess above the enlarged `RATE_A + 1 MW` bound is **1.542 MW**.
The public evaluator's M5 comment reports **three** events; even under that
reporting convention our result is not close. Data differences still prevent
claiming a controlled reproduction of Fig. 6.

## Decoder selection and numerical fallback

Primary M5's **predeclared fixed 6 MW physical-RMSE gate**:

| Split | Neural coverage | Accepted source accuracy | Completed hybrid source accuracy | Hybrid strict feasibility |
|---|---:|---:|---:|---:|
| IID | 98.70% | 98.539% | 98.558% | 1.30% |
| OOD1 | 96.40% | 97.078% | 97.183% | 3.60% |
| OOD2 | 79.35% | 95.768% | 96.642% | 20.65% |

The separately declared validation-q95 RMSE threshold is **4.649828910827637 MW**:

| Split | Neural coverage | Accepted source accuracy | Completed hybrid source accuracy | Hybrid strict feasibility |
|---|---:|---:|---:|---:|
| IID | 95.65% | 98.554% | 98.617% | 4.35% |
| OOD1 | 88.70% | 97.454% | 97.742% | 11.30% |
| OOD2 | 63.50% | 97.625% | 98.492% | 36.50% |

**Every feasible primary-M5 hybrid output comes from numerical fallback.**
Neither gate accepts a strictly feasible M5 output, and neither is a safety
certificate. The hybrid's higher accuracy is not raw M5 accuracy or evidence
that the neural method alone matches paper M5. Table V has no IEEE-30 decoder
row, so these are not direct reproductions of a published Table V entry.

The unresolved gate norm is material. With the **same final primary-M5
decoder at OOD2**, the predeclared score diagnostics give:

| Physical reconstruction score | Coverage at 6 MW | Accepted accuracy at 6 MW | Coverage at 7 MW | Accepted accuracy at 7 MW |
|---|---:|---:|---:|---:|
| Mean absolute | 99.80% | 89.521% | 99.95% | 89.411% |
| Root mean square | 79.35% | 95.768% | 87.60% | 93.912% |
| Summed absolute | 0.00% | Unavailable | 0.00% | Unavailable |
| Maximum absolute | 5.35% | 97.040% | 8.75% | 97.619% |

Summed absolute error is 30 times MAE for 30 inputs; the same number is not the
same operating point under these norms. No norm or threshold was selected
from these tests. Full variant/model gate results remain in the generated
analysis and `results.csv`.

## FGSM and the reference-target convention

White-box self-FGSM uses the model's declared supervised loss, with
**L-infinity 1 MW across all 30 input coordinates**, no nonnegative clipping
and no attack training. Each model may induce different attacks. Saved deltas
obey the budget; a 1 MW per-coordinate budget is **not** a 1 MW total-demand
budget.

Primary representative results:

| Method | Attack source | Accuracy against original dispatch, all 2,000 attacks | Accuracy against newly solved OPF dispatch | Valid new OPFs |
|---|---|---:|---:|---:|
| M1 | IID | 1.658% | 98.708% | 2,000 / 2,000 |
| M1 | OOD1 | 2.800% | 96.615% | 1,999 / 2,000 |
| M5 | IID | 0.533% | 98.330% | 1,996 / 2,000 |
| M5 | OOD1 | 0.900% | 96.511% | 1,997 / 2,000 |

For primary M5, **91.0% / 87.8%** of IID/OOD1 attacks move every coordinate
in the same direction. The median absolute total-net-load change is **30 MW**
in both sets. On valid re-solved cases, the mean absolute change in the
six-generator target is **5.156 / 5.224 MW per output**, and **99.424 / 99.082%**
of target outputs change by at least 1 MW. These statistics come directly from
`attack_all_delta_mw`, `attack*_original_y` and the newly solved `attack*_y`.

Thus the very low frozen-target scores do not by themselves establish failure
to predict the optimum for the changed physical input. Conversely, high
re-solved-target scores do not demonstrate robustness to corrupted
measurements when the original physical load is unchanged. Both estimands are
reported; the paper's target convention is not recovered. Its attack rows
cannot be claimed reproduced from whichever convention looks closer.
Fresh-reference failures remain visible, with no replacement of attacks.
All 36 attack evaluations are preserved in the generated report.

## Runtime and remaining limits

Stage wall-time totals: dataset **56.32 s**, model training **1,872.72 s**,
decoders **568.49 s**, calibration **1.86 s**, attack generation **345.35 s**,
evaluations **1,289.11 s**. The overall wall time additionally includes
orchestration and verification. Contention varied; this is not a controlled
hardware speed comparison with the paper or prior pilot. Worker decision time
is distinct from batched inference and audited end-to-end throughput.

This completes the requested approximate full-suite rerun, not an exact
reproduction. Principal remaining gaps are the missing author data generator,
penetration/placement/conversion/dependence choices, paper/example numerical
conflicts, framework parity, gate/attack conventions, full external DeepOPF
and other paper systems. Only one training seed was run; 2,000 scenarios per
regime and duplicated objectives do not quantify training-seed uncertainty.
The new evidence does not justify changing any frozen settings after the fact.

## Verification and artifact anchors

Preflight passed **141 regression tests**, six subsequent runner/report checks
and `pip check`. Historical final M1/M2/M4/M5 batched predictions remained
bitwise identical and all **135 earlier pilot output hashes** unchanged.
The final post-run regression check passed **143 tests**, with 14 existing
pandapower transformer-data deprecation warnings.

The following SHA256 anchors identify the local completed outputs; these are
ignored run artifacts, not large files committed with this report:

| Relative to run root | SHA256 |
|---|---|
| `suite_manifest.json` | `3409363775e681b39f0da377fafcbb8f8bb88ecbb29b37a39ea9bf439b49968f` |
| `results.csv` | `cd46950b9e91fa6a47b99b1e2db6c8f0d41815d2450470b29e88c3f0110b9661` |
| `analysis/analysis.json` | `b8dae2ad2cdb75296dd3e0e395cff7e8109ee35e883619ba9293c866a65ae690` |
| `analysis/report.md` | `ce2ef91d92ededfefaabc89b619bd1321cf325b3de7e8690c0a53be72cd03a79` |

To verify and independently regenerate descriptive analysis using the original
source, choose a **new** output directory (the analyzer refuses overwrites):

```sh
RUN=artifacts/gao-reconstruction-full-20261119T133302Z
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  PYTHONPATH="$RUN/source" .venv/bin/python -B -m powersim.reproduction_analysis \
  "$RUN" --output "$RUN/analysis-recheck"
```

This command verifies saved outputs and frozen source; it does not rerun
training, choose a checkpoint or replace any original result.
