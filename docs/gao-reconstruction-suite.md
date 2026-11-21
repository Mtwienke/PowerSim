# Source-informed IEEE-30 reconstruction suite

**Predeclared 2026-11-19, before training or inspection of the new test streams.**
User authorization: proceed without the expired full-project share or author
contact, apply the strongest source/mathematical hypotheses, then rerun a full
suite. **IEEE-30 DC only; Germany remains excluded.**

This is an approximate reconstruction, not recovered author settings. The
[pinned-source audit](gao-source-audit.md) remains the evidence record. Its
earlier recommendation to wait for missing source is superseded by the user's
explicit decision to proceed with labelled hypotheses. No author is contacted.

**Completion update (after execution):** all **239 stages / 165 evaluations**
completed with **829 required output hashes verified**, frozen at `e83e085`.
The [measured full-suite report](gao-reconstruction-results.md) records the
improvement over the earlier pilot and the remaining M5-ranking/feasibility
failure. The predeclared numerical protocol below was not retuned.

## New data, old artifacts preserved

- Keep the bundled IEEE-30 topology, source costs/bounds and **unscaled branch
  ratings (1.0)**, matching the examples' ordinary `case30()` convention.
- Adopt the strong installed-capacity hypothesis
  `penetration = renewable / (335 MW conventional + renewable)`.
  Explicit **185 / 230 / 300 MW** nameplates round to **35.58 / 40.71 / 47.24%**.
  Store both printed regimes and actual fractions; do not call them verified.
- Retain two wind and two PV sites, seeded placement, equal technology/site
  capacity allocation, independent relative 20% load draws, the journal's
  Weibull/Beta parameters, cubic wind curve and PV performance ratio 0.85.
  These unresolved choices are held fixed rather than fit to published numbers.
- Seed **2027** for placement, training/validation scenarios and ML.
  Separate **20281121** test streams create new IID/OOD conditions. Test records
  must not enter fitting, model selection, decoder training or calibration.
- Collect **10,000 valid training, 1,000 valid validation, 2,000 valid cases per
  test regime**. Replace failed attempts until that target, with a declared
  finite attempt cap; persist every consumed failed attempt and report coverage.
  This is a labelled successful-sample hypothesis, not the historical
  fixed-attempt protocol or proof of mathematical infeasibility.
- Retain training-only standardization statistics for compatibility, but the new
  networks use **zero-centered 100 MW per-unit input/output scaling**.
- No reference-label clipping, curtailment, branch repair or hidden dispatch
  projection. An L1 sensitivity mask never deletes an L0 training case.

## Three numerical specifications, all reported

Common: three 200-neuron ReLU hidden layers; linear all-generator head;
Glorot-uniform weights, zero biases; batch 50; float32 CPU; Keras-v2-style Adam
equations with epsilon-hat 1e-7 and inverse-time decay 2e-6; fresh moments for
each continuation. PyTorch execution is not claimed bitwise identical to TF.

All use the example's L1 predicate `sum(J) != 0 && signed_max(J) < 10`, applied
only in addition to reference-sensitivity availability. Masked residuals are
averaged over the full selected tensor, not renormalized by active entries.
Exploration uses independent Gaussian std **1 per-unit = 100 MW** across input
coordinates; no active-set labels are extrapolated to those points.

| Variant | Role | Rate | Prediction loss | Pattern residuals | Active constraints |
|---|---|---:|---|---|---|
| `source-informed` | Primary approximate reconstruction | 0.001 | Per-unit MAE | MAE, except squared L1 | Include generator-bound pattern and all active branches |
| `journal-equations` | Predeclared equation/rate control | 0.01 | Per-unit MSE | Squared | Include generator-bound pattern and all active branches |
| `public-example` | Source-objective diagnostic | 0.001 | Per-unit MAE | MAE, except squared L1 | Omit L3 and restrict L4 to source branch row 20 |

The primary differs from the visible example by generalizing its local
constraint patterns according to the journal mathematics. Its absolute-loss
norms are **not** the journal's squared L0–L5. The journal-equation control still
uses inferred numerical settings and label masking. Never select whichever
variant happens to match the paper best and relabel it “the reproduction.”

Within each variant:

1. M1: supervised objective, **1,000 epochs**, final checkpoint.
2. M1-continued: another **200 supervised epochs**, budget control.
3. M2: supervised plus sensitivity learning, **200 epochs**.
4. M4: supervised, sensitivity and applicable local inherent patterns,
   **200 epochs**.
5. M5: M4 plus unlabelled balance-pattern exploration, **200 epochs**.
6. `M3-example`: the public comparator's supervised loss plus
   `mean(relu((branch_flow / branch_limit)^2 - 1))`, **200 epochs**.
   This is an actual branch-flow-penalty comparator, not a leave-one-out.
   It is **not** relabelled as the complete security-constrained DeepOPF
   predict/reconstruct/projection algorithm, which remains an external gap.

Every continuation starts independently from its variant's same **final M1**.
All enabled weights are 1; source-excluded terms are explicitly zero. Keep
final and best-validation checkpoints. Final is primary; the primary variant's
best-validation raw evaluations are a predeclared secondary arm. No M1-parent
selection based on held-out performance.

## Decoders and gate hypotheses

Train a frozen-network decoder for every model, **1,000 epochs**, batch **500**,
ReLU width **200**, per-unit MSE, Adam **0.001**, inverse-time decay **2e-6**,
Glorot/zero initialization, and fixed-final selection.

- Primary: last hidden feature (source evidence), **two** decoder hidden layers
  (journal depth).
- Journal-equation control: second/penultimate hidden feature, **two** layers.
- Public-example diagnostic: last hidden feature, **three** layers.

Primary operating gate: **physical reconstruction RMSE <= 6 MW**. RMSE is a
dimensionally consistent hypothesis for the MSE-trained decoder, **not a
recovered author norm**. Also record fixed 6/7 MW MAE, RMSE, summed-absolute and
maximum-absolute accepted-accuracy/coverage diagnostics without selecting a
threshold from tests. A separate primary-M5 hybrid uses validation-only q95
RMSE, clearly labelled secondary.

Acceptance is not feasibility. Keep strict raw and hybrid PF checks unchanged.
Also report the source-like **branch-scenario event count beyond RATE_A + 1 MW
after slack rebalance**, separately from strict physical feasibility.

## Full evaluation workload

- All **18 model trainings and 18 decoder trainings** (six methods × three
  numerical specifications); no German scenario.
- Fixed-final raw and fixed-6-MW-RMSE hybrid on all three regimes: **108 passes**.
- Primary variant's saved best-validation raw checkpoints: **18 passes**.
- Primary M5 validation-q95-RMSE hybrid: **three passes**.
- White-box **1 MW FGSM** on IID and first OOD inputs for all fixed-final models:
  **36 passes**. Different models may create different attacks.
- Total **165 evaluation passes**; dataset/reference generation and calibration
  are additional stages. Eight independent PF/OPF workers; sequential trainings.

FGSM uses the model's declared supervised loss against the original reference.
Store attack vectors/model hashes, enforce L-infinity <= 1 MW, and do not clip
net-load inputs to nonnegative values. Re-solve OPF at perturbed inputs rather
than silently treating old optima as new physical labels. Report **both**
fixed-original-target adversarial accuracy over all attacked cases and
perturbed-reference accuracy/physics conditional on new valid solves.
No attack data enters training or decoder calibration.

Report strict individual-generator <1 MW accuracy (paper aggregation inference),
joint accuracy, MAE/RMSE, raw balance/bounds/flows, branch-event counts,
accepted accuracy and coverage, fallback failures and measured costs. Excluded
reference attempts remain visible in additional conservative denominators.
Do not equate improved prediction accuracy with strict physical feasibility.

## Integrity, stopping and analysis

**Pre-full-run seed amendment:** the initial frozen plumbing driver used
20281119 for its tiny test streams. To avoid even prefix overlap with those
plumbing cases, the full suite reserves **20281121**, before full training or
evaluation. No smoke accuracy is used to choose this seed or any numerical
setting. Subsequent `--smoke` defaults use separate 20261119/20261120 seeds.
This is an isolation change, not a search over favorable test outcomes.

Commit each substantive implementation/protocol step. Freeze committed source,
package versions, settings and hashes before the full run. Use new stage/attempt
directories, resumable hash checks and one live driver lock; never overwrite old
datasets/models. Compress new per-case evaluation JSONL to bound disk use.

Stop on nonfinite losses, invalid provenance, attempt-cap exhaustion, insufficient
disk or stage failure; preserve diagnostics. Do not silently lower epochs,
discard sensitivities, change rate/losses or choose a better-looking test result.
Unit/integration smoke tests may verify plumbing before launch; they are not
hyperparameter searches. All three specifications are evaluated regardless of
whether their fitting or method rankings improve.

After completion, verify hashes and compare with the paper and earlier pilot.
Separate the bundled effects of changed data/numerics from causal ablations.
Single-seed agreement would still leave conversion/placement, framework,
DeepOPF and other-system reproduction gaps.

## Execution record

Implementation includes opt-in numerical profiles, bounded successful-sample
collection, physical decoder norms, source-like branch-event reporting, separate
fixed-target/re-solved-target FGSM metrics and compressed prediction details.
Historical defaults and saved models remain compatible.

```sh
python -m powersim.reproduction artifacts/gao-reconstruction --plan
python -m powersim.reproduction artifacts/gao-reconstruction
# After the full driver has completed:
python -m powersim.reproduction_analysis artifacts/gao-reconstruction
```

The runner freezes committed source and records every stage's input/output
hashes. `--resume` verifies completed outputs and starts interrupted stages in
new attempt directories. Do not steal `.suite.lock`; verify the old driver and
children have stopped first. Source/package/configuration changes require a new
run, or resumption through the original frozen source/environment.

Preflight and a tiny `--smoke` workflow preceded the full run. The launch
record below was committed before execution; measured results are recorded
separately in the [completed report](gao-reconstruction-results.md), not
inferred from smoke accuracy.

Preflight at launch:

- Full regression suite: **141 passed**; subsequent runner/report checks:
  **six passed**. `pip check` is clean.
- Historical final M1/M2/M4/M5 batched predictions are **bitwise identical**
  under the extended loader.
- All **135 earlier pilot output hashes** remain unchanged.
- The first actual tiny workflow is frozen at **`6252b8f`** under
  `artifacts/gao-reconstruction-smoke-20261119/`. Its purpose is orchestration
  validation only; no smoke accuracy informs full-run settings.

The real smoke workflow **completed all 239 stages / 165 evaluations**, with
**829 required output hashes verified**. The read-only JSON/Markdown analysis
path also passed against those real artifacts. Startup contention with the
regression tests makes its runtime unsuitable for a full-run estimate.

Full-budget run root reserved after these checks (prelaunch record):
`artifacts/archive/gao-reconstruction-full-20261119T133302Z/`.
The driver recorded its exact frozen commit and full-budget configuration in
`suite_manifest.json`. Prelaunch free disk: **20.1 GiB**, with a 5 GiB stop reserve.
No numerical setting or threshold was selected from smoke accuracy.

The full run completed **2026-11-19, 13:33:59–14:43:03 UTC** in **69m 03.75s**:
18 model trainings, 18 decoders, 36 attack datasets and 165 evaluations.
The frozen post-completion analyzer reverified all 829 required output hashes
and the archived source. Primary M5 OOD2 accuracy is **89.375%**, versus
**90.167%** for M1 and **97.94%** for paper M5. The full improvement and
negative findings, control variants, gates and target conventions are retained
in the linked report. No settings or checkpoints were replaced afterward.
