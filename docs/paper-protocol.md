# Full-text audit and shared-M1 pilot

Target: Gao et al., *A Trustable Data-Driven Optimal Power Flow Computational
Method With Robust Generalization Ability*, IEEE TNNLS 36(5), 8049–8059,
DOI [10.1109/TNNLS.2024.3437741](https://doi.org/10.1109/TNNLS.2024.3437741).
The user supplied the full article on 2026-11-19. Page numbers below are the
printed article pages. The licensed PDF is not redistributed in this repository.

## Confirmed protocol

Table I and Section V-A/B, p. 8055:

| Method | Objective / algorithm |
|---|---|
| M0 | Conventional numerical interior-point OPF |
| M1 | L0 supervised regression |
| M2 | L0 + L1 sensitivity learning |
| M3 | DeepOPF, **not** a leave-one-out loss ablation |
| M4 | L0–L4 |
| M5 | L0–L5 |

For IEEE-30: three 200-neuron hidden layers, **1,000 epochs of M1 pretraining**,
then **200 additional epochs for M2–M5 initialized from trained M1 weights**.
The OPF Adam learning rate is **0.01**, and all enabled loss weights are **1**.
The old full suites instead used independent initialization, 100 epochs and
learning rate 0.001. Their files and measured results have not been rewritten.

The Weibull parameters in the current generator, scale **5.089** and shape
**2.016**, agree with Section V-A. The reversed labels in the related patent
must not be substituted for the journal's parameters.

Section IV-A and V-C, pp. 8054/8057, specify a decoder with **two hidden layers**,
penultimate-hidden-layer input, learning rate **0.001**, and **1,000 epochs**.
Fig. 10 uses a 7 MW threshold; Table V uses 6 MW. The reconstruction norm/scaling,
decoder widths, activation, batch size, normalization, perturbation magnitude,
renewable placement and precise penetration definition remain insufficiently
specified for exact numerical reproduction. Current choices remain declared
assumptions. The paper's introductory example describes 20% load fluctuation;
this is consistent with, but does not fully specify, our relative-load sampler.

## Penetration-denominator hypothesis (not author-verified)

During the IEEE-30 pilot, an independent arithmetic check found a specific
candidate for the paper's unspecified penetration definition. Our IEEE-30
conventional source capacities sum to **335 MW**. If penetration instead means
`renewable nameplate / (conventional nameplate + renewable nameplate)`, these
round renewable capacities reproduce every Table II percentage:

| Renewable MW | `R / (335 + R)` | Rounded paper rate | Our existing renewable MW |
|---:|---:|---:|---:|
| 185 | 35.576923% | 35.58% | 67.31736 |
| 230 | 40.707965% | 40.71% | 77.02332 |
| 300 | 47.244094% | 47.24% | 89.37808 |

Our dataset uses `renewable nameplate / nominal gross demand`, with demand
189.2 MW. The candidate total-capacity definition would imply both substantially
larger installations and a stronger capacity increase: **62.2%** from 185 to
300 MW, versus **32.8%** from 67.31736 to 89.37808 MW. Mean realized renewable
generation in our reference-valid tests is only 8.53%, 9.70%, and 11.18% of
sample demand.

This exact rounded match is evidence for a testable hypothesis, **not proof of
the authors' sampling code**. No capacity, dataset, training or threshold was
changed on its basis in the running pilot. Author clarification or a separately
declared sensitivity experiment is needed before treating this as the paper's
definition. Matching the printed percentages alone does not establish matching
operating distributions.

## IEEE-30 sensitivity-tail diagnostic

A separate read-only check during the new pilot examined the existing
seed-2027 IEEE-30 training labels. Two of the 9,425 valid training cases account
for **96.1854% of the reference Jacobians' total squared magnitude**. Their
largest absolute derivative is **1,555.846 MW/MW**. Both have active branches
`line:9`, `line:28`, `line:29`, `line:34`, plus a source at its upper bound.
The KKT condition number is approximately 690,805, below the implementation's
singularity rejection threshold but much larger than typical training cases.

The extreme derivatives are not merely an assumed label bug. At training
attempt 7,163 (filtered row 6,749), independent central finite differences of
fresh DC-OPF solves at bus 7 preserve the active branch set. Perturbations of
0.001 and 0.0001 MW reproduce the stored derivative to maximum relative
errors of approximately **1.21e-8** and **8.86e-11**, respectively. The baseline
re-solve reproduces saved generation exactly.

This identifies a possible sensitivity-learning difficulty, not a reason to
delete those cases or silently clip L1. Reference squared magnitude is not
itself the trained loss or its gradient contribution; those require separate
checkpoint diagnostics. The current run keeps all labels and unit weights
unchanged. Any robustness treatment must be reported as a separate ablation,
and the paper's unspecified sampling/branch-limit settings still matter.

## Accuracy is not feasibility, and selection is not hybrid completion

Equation (26) uses **strict error < 1 MW**, but leaves vector reduction unclear.
There is strong evidence that tables aggregate individual generator outputs:
Table V's IEEE-118 case at 44.26% penetration discards 1,998/2,000 scenarios
and reports 95.37% accuracy. The remaining two scenarios have 108 source outputs;
**103/108 = 95.37037%**. Joint scenario accuracy for two cases cannot be 95.37%.
Generator-wise aggregation is therefore explicitly labelled an inference,
not quoted as a fully specified author metric.

New reports expose:

- `generator_accuracy_lt_1mw`: strict individual-output accuracy.
- `joint_accuracy_lt_1mw`: strict complete-scenario accuracy.
- Counts, denominators and per-generator accuracy, including slack.
- Legacy `success_rate_1mw` and explicit alias `joint_success_rate_1mw`:
  **all** source errors **<= 1 MW**, preserving existing reports/API semantics.
- Conditional `accepted_accuracy` and `accepted_physics`, separate from
  `hybrid_accuracy`, `fallback_accuracy` and whole-hybrid physical audits.
- Neural coverage and reference coverage with actual scenario counts.
- Conservative all-requested accuracy: excluded references and failed outputs
  remain unsuccessful in that denominator.

NaN/infinite predictions remain failures, not dropped accuracy denominators.
An empty accepted subset has **null**, not 100%, conditional accuracy.
MAE/RMSE retain the historical complete-finite-row aggregation.
Individual generator outputs in one scenario are correlated; their count is
not a count of independent confidence-interval trials.

The paper's Table V high accepted accuracy can accompany very low coverage:
0.9% at IEEE-57 / 50.31%, 0.1% at IEEE-118 / 44.26%, and 11% at IEEE-300 /
26.05%. Those conditional percentages are not the accuracy or speed of a
completed workload including OPF fallbacks. Table IV times raw M5 inference;
we continue measuring decoder/fallback latency separately. The reported
**30.19%** improvement reproduces as **30.19 percentage points**, averaging
M5 minus M2/M3 over the eight OOD conditions in Table III.

Fig. 6 reports remaining branch violations for M5. Its high prediction accuracy
is not a strict feasibility guarantee. A constant output-head bias leaves
Jacobian patterns and decoder features unchanged while potentially violating
balance/bounds. Strict PF audits are retained, with separate branch-event,
affected-scenario and distinct-branch counts.

## Run the pilot explicitly

No dataset generation, training or held-out evaluation starts automatically.
The pilot uses one existing DC dataset, leaving its operating assumptions
unchanged. For example:

```sh
DATASET=artifacts/full-suite-seed2027-20261119T021858Z/stages/germany30--dataset/attempt-001

# Read-only: validates inputs and prints the plan; does not create OUTPUT.
powersim pilot "$DATASET" artifacts/gao-pilot-germany2027 --plan

# Explicit long training: five stages, training/validation reporting only.
powersim pilot "$DATASET" artifacts/gao-pilot-germany2027

# Optional paired learning-rate control; use a different output directory.
powersim pilot "$DATASET" artifacts/gao-pilot-germany-controls --learning-rates 0.01,0.001
```

Repeat with the existing `case30--dataset/attempt-001` dataset for IEEE-30.
Transferring this schedule to Germany is an experiment, not a German result
reported by Gao et al.

At each learning rate:

1. Train M1 for 1,000 epochs.
2. Initialize **M1-continued, M2, M4 and M5 independently from that same M1**.
3. Train each for 200 epochs. M1-continued controls for the added training budget.
4. Reset Adam moments for every continuation, consistently across all methods.
5. Report selected-checkpoint train/validation accuracy and validation history.

The pilot defaults to final-epoch selection (`--checkpoint-selection last`).
The author checkpoint-selection rule is unspecified; this default is an explicit
assumption. `--checkpoint-selection best_validation` is an opt-in alternative.
All training saves `last.pt`, `best.pt`, and selected `model.pt`; both alternatives
are retained, never selected using test accuracy.

`pilot_manifest.json` freezes settings, source/package identities, dataset file
hashes and assumptions. `status.json` records stage progress; `summary.json`
reports only train/validation performance. Completed stages are hash-verified
on explicit `--resume`:

```sh
powersim pilot "$DATASET" artifacts/gao-pilot-germany2027 --resume
```

An interrupted stage restarts in a **new attempt directory**; earlier attempts
are preserved. This is stage-level, not optimizer/epoch-level resumption. Use
the same configuration on resume. Source, package or input changes require a new
output. A `.pilot.lock` prevents concurrent drivers; after a hard kill, check
that its PID is no longer running before removing only that stale lock.

The generic training command also supports checked weights-only initialization:

```sh
powersim train DATASET M2_OUTPUT --method M2 --epochs 200 --learning-rate 0.01 \
  --initialize-from M1_OUTPUT/model.pt --checkpoint-selection last
```

Warm starts verify dataset/model IDs, source/bus ordering, architecture,
normalization, scenario/input IDs and (for new checkpoints) train/validation
data and state hashes. Legacy checkpoints remain usable but explicitly report
that their missing training-data hash could not be verified. Shuffle and
perturbation RNG streams are separate, so enabling L5 does not change later
epoch batch permutations.

## Decoder and evaluation are separate follow-ups

Once IID fitting is adequate, train a decoder explicitly. The example widths
below are **assumptions**, not widths disclosed in the paper:

```sh
powersim decoder DATASET MODEL.pt DECODER_OUTPUT --epochs 1000 --hidden 128,128 \
  --learning-rate 0.001 --error-units mae_mw
powersim evaluate DATASET MODEL.pt EVALUATION_OUTPUT --decoder DECODER_OUTPUT/decoder.pt \
  --split iid_test --workers 8
```

Default calibration remains the validation-only 95th percentile, not a target
accuracy guarantee. Scores now have explicit norms/units:

| `--error-units` | Reconstruction score |
|---|---|
| `normalized` | Normalized-input MSE (legacy default) |
| `mw2` | Physical-input MSE, MW² (legacy) |
| `mae_mw` | Physical-input MAE, MW |
| `rmse_mw` | Physical-input RMSE, MW |

`--threshold 6` explicitly requests a fixed threshold in those score units;
it does **not** establish the paper's unspecified norm or certify dispatch
accuracy. Fixed and quantile thresholds are recorded distinctly. One-hidden-
layer legacy decoder files remain loadable.

Evaluation reports descriptive accuracy/physical-risk-versus-coverage rows
at validation-derived thresholds and the actual operating threshold, including
a reject-all row. It does not choose a threshold using held-out performance.
Old decoders without validation threshold grids report only their operating
threshold and reject-all. Confidence bounds and label/feasibility-targeted gate
calibration are not implemented by this reconstruction-quantile comparison.

## Sequenced next experiments (not silently enabled)

1. Establish IID fitting under the shared-M1 schedule on unchanged data.
2. If fitting remains inadequate, isolate prediction-loss units and measure
   gradient contributions before changing loss weights. Preserve physical
   Jacobian chain rules; do not mistake a scale-extension for exact reproduction.
3. Keep the current German stress dataset; evaluate hosting-capacity-aware
   allocation or explicit curtailment as a **separate** optimization problem.
4. Compare gates by conditional accuracy, coverage, physical violations and
   measured end-to-end fallback cost, with adequate independent accepted cases.
5. Use multiple seeds and new reserved tests for confirmatory claims. Add the
   paper's 1 MW FGSM attacks and genuine DeepOPF/M3 only as explicit experiments.
6. Any online adaptation needs a separate chronological stream; never train on
   the held-out records used to evaluate the existing models.

No curtailment, network repair, loss rescaling, FGSM, automatic online learning
or regenerated benchmark is introduced by the pilot.
