# Report metric definitions and denominators

Unless stated otherwise, CSV rates are fractions in `[0,1]`, not percentages.
`*_difference_pp` is `100 × (method − baseline)` and `*_relative_change_pct` is
`100 × (method − baseline) / baseline` (undefined for zero baseline). Blank
means not applicable, not measured, or an empty denominator; it is not zero.

## Keys, cohorts and error

- `experiment_id` identifies the run/profile/method/checkpoint/scenario/protocol
  and dataset checksum. `dataset_id` is the SHA-256 of the frozen label NPZ.
  `method` retains the archival stage token; `run` and `profile` qualify it.
  In particular, `M2`, `M4` and `M5` in the pilot/reconstruction are local
  analogues, not the authors' implementations, and `public-example` omits L3.
  Within that dataset, pair samples by `sample_id`, not filtered position.
  `source_sample_id` connects an FGSM attempt to its clean counterpart. Per-array
  row and column IDs, errors, dispatch, flows and bound excesses are in
  `matrices/<experiment_id>.npz` in MW. `reference_attempts.csv` contains **all**
  recorded held-out attempts including no-label cases and their statuses.
- `sample_metrics.csv` shows evaluated predictions, not fabricated predictions
  for failed reference attempts. The fixed-original-target FGSM row includes
  all attacks even when *fresh* `reference_valid` is false; its target is the
  clean optimum, and it deliberately has no physical or cost comparison.
  Attack-vs-clean differences in `comparisons.csv` are matched on source
  scenario, including for the subset with valid fresh OPF dispatch.
- `error_ij = abs(pred_ij - ref_ij)` in MW, including the external-grid slack.
  The data and saved predictions are already de-normalized to MW; no further
  base-MVA multiplication is needed. A model's normalization scale and the
  electrical network's `sn_mva` are separate.
- For each of 0.5, 1 and 2 MW, `element_lt_*` = hit elements / (`N × G`);
  `max_lt_*` = scenarios with **every** generator error below threshold / `N`;
  `mean_lt_*` = scenarios with mean absolute generation error below threshold
  / `N`. `N` is the reference-valid scenario count (or all original targets
  for fixed-target FGSM). Nonfinite outputs are unsuccessful. Corresponding
  `_all_attempts` uses the attempted cohort including reference failures.
  Historical `legacy_joint_le_1mw` is inclusive; the paper's strict 1 MW
  reduction is plausibly element-wise but not explicitly specified. One
  clue is Gao Table V: the IEEE-118 row with two accepted cases has 95.37%
  accuracy (compatible with 103 of 108 source outputs, not two whole cases).
- MAE, RMSE, median, p95, p99 and maximum pool generator errors over **complete
  finite rows**, matching the historical MAE/RMSE scope. Row-wise MAE and max
  are preserved separately. No deterministic case or generator is counted as
  an independent training run. The historical two-seed summaries use sample
  SD and an unbounded Student-t interval (`df=1`), exploratory only: the
  generator placement, training data and test draws also change by seed.

**Hand check:** predictions `[11, 20]`, `[9.5, 20.5]` against two `[10, 20]`
references give errors `[1, 0]`, `[0.5, 0.5]`. At strict 1 MW, element accuracy
is 3/4, joint-max accuracy 1/2, and joint-mean accuracy 2/2. MAE is 0.5 MW and
RMSE is `sqrt(1.5/4) ≈ 0.61237 MW`. The legacy inclusive-max rate is 2/2.

## DCOPF feasibility and cost

- Balance uses the **proposed** conventional dispatch: `sum(g) - sum(x) -
  shunt_mw`, with `x` the fixed-injection-adjusted net demand. Renewable output
  is not subtracted again. We report signed per-sample residuals and mean,
  p95 and maximum absolute residuals. `balance_exceeds_*` uses complete physics
  rows. 0.01/0.1/1 MW are descriptive thresholds, not engineering acceptance.
- Flow is the saved project map `g @ M_G.T - x @ M_D.T + constant`, including
  tap/phase/shunt conventions; rebuilt from the frozen network and verified
  against saved **numerical** dispatch flows in every evaluated split. When
  unbalanced, DC flows diagnose a compensating injection at the reference
  bus; by themselves they are **not** flows of a valid raw dispatch. Raw
  output, raw generator limits, and raw balance are never replaced by that
  diagnostic injection. This report checks proposed bounds; the older PF
  audit additionally checked slack-adjusted source bounds.
- Lower/upper generation excesses are `max(p_min - p, 0)` and
  `max(p - p_max, 0)`; line excess is `max(f - f_max, f_min - f, 0)` with
  `f_min=-RATE_A`, `f_max=RATE_A`. Unrated lines are unbounded. Limits come
  from the stored reference and include its rating scale. A violating event
  is one asset × sample with excess >0.001 MW. Counts distinguish affected
  cases, events and distinct assets. Magnitudes are raw excess among
  violating events, not excess minus numerical tolerance. The separately
  labelled public-example diagnostic counts events beyond `RATE_A + 1 MW`
  and measures excess over that enlarged limit.
- Strict feasibility means complete finite dispatch, absolute balance
  ≤0.0001 MW, all source and branch excesses ≤0.001 MW. Pass/combined rates
  use the scored scenario cohort; missing outputs do not pass. Violation
  rates use physics-evaluated outputs only. This is numerical feasibility
  **relative to modeled DC constraints**, not a grid-operation guarantee.
- Cost is the frozen network's active polynomial `sum(c0 + c1*p + c2*p²)`
  (EUR/h). Signed differences and ratios `(pred-ref)/abs(ref)` are split by
  feasible and infeasible predictions. Relative differences are unavailable
  if `abs(ref) ≤ 1e−9 EUR/h`. A lower infeasible cost is not an optimality
  improvement. Reference solves require convergence and validated
  constraints/finite outputs; no independently certified optimum is claimed.
  M0 rows describe reference coverage, not circular self-prediction accuracy.

## Decoder and fallback

- `accepted = finite score and complete raw output and score ≤ threshold`.
  Accepted/rejected metrics describe **raw** prediction; completed hybrid
  metrics describe the chosen output, from a fresh numerical solve on
  rejection. A failed fallback does not become the stored optimum. Counts
  and calls per scored sample are explicit. `score_type` is the reconstruction
  reduction; `score_units` is the resulting unit (e.g., MW or normalized²).
- Positive class is **unreliable**, predicted positive means **reject**.
  Accuracy-positive iff raw joint-max criterion `<1 MW` fails (including
  missing output) among reference-valid cases. Feasibility-positive iff
  measured raw physics fails a constraint; outputs without a physical
  audit are unknown for this classification and excluded, not labelled
  feasible. `TP=reject+unreliable`, `FN=accept+unreliable`, `FP=reject+reliable`,
  `TN=accept+reliable`. False acceptance `FN/(TP+FN)`; false rejection
  `FP/(FP+TN)`; precision `TP/(TP+FP)`; recall `TP/(TP+FN)`. Prevalence
  `(TP+FN)/(TP+FN+FP+TN)`. Accept/reject rates use all scored scenarios;
  subset accuracy and feasibility use only that subset.
- AUROC is trapezoidal ROC area and AUPRC is non-interpolated average
  precision, grouping tied scores. Score metrics use known labels with
  finite scores; no discrimination statistic is reported for a one-class
  cohort. Scoring is for unreliable prediction, **not** for recognizing OOD.
- Actual operating gates come from frozen validation calibration or declared
  fixed thresholds. `threshold_curves.csv` includes reject-all, validation
  quantiles and explicit 6/7 MW diagnostics in MAE/RMSE units **only**. Legacy
  grids are re-inferred on decoder-bound validation inputs only. Curves do
  not select a final threshold using test outcomes. Estimated sweep times
  use archived summed non-fallback worker decision time plus rejected count
  times the observed mean fallback duration. These are estimates, not
  elapsed wall time or replayed fallback outcomes.

## Runtime

Controlled timing uses the same ordered reference-valid inputs for numerical,
neural and hybrid paths; no expensive startup, audit, or file writing in the
per-case path. CPU with one Torch thread, warm-ups and repeated wall
measurements; no GPU synchronization applies. Pure neural latency is batch
1; throughput is also measured at batch 256. Decoder score time includes
feature extraction and reconstruction; the current hybrid implementation
recomputes features. `*_including_load` adds independently measured NPZ read
and contiguous input preparation to the matching decision workload; excludes
one-off model and network initialization. Report speedup as numerical wall
time divided by method wall time within the same workload and repeat. Batch
throughput and summed worker time do not imply per-sample wall-clock latency.
Archived drawing/training/parallel stage times are useful provenance, not
speedup denominators. Runtime on valid inputs excludes failure-heavy
unlabelled workloads and does not establish paper hardware parity.
