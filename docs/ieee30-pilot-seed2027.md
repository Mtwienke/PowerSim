# IEEE-30 shared-M1 pilot: seed 2027

**Completed 2026-11-19 in 11 min 18 s: five training stages, one decoder,
one threshold-calibration stage and 36 evaluation passes.**
Requested scope: IEEE-30 only; Germany-30 is excluded.

**Later source evidence:** the [pinned public-example audit](gao-source-audit.md)
supersedes statements below that author activation, batching and all numerical
conventions are wholly unknown. The examples establish several concrete
differences, but also conflict with the final journal. They were discovered
**after this frozen run**; its methods, settings, results and negative outcome
have not been changed.

**Conclusion: the published accuracy and method ranking were not reproduced.**
The new implementation follows the verified loss definitions and shared-M1
training schedule much more closely, but this is not yet a faithful numerical
reproduction. At the highest regime, fixed-final M5 source-wise accuracy is
**31.01%**, versus **97.94%** in the paper; our M1 reaches **82.46%**.
Poor fitting already occurs on training and IID validation data.

## Frozen comparison protocol

- Existing IEEE-30 dataset:
  `artifacts/full-suite-seed2027-20261119T021858Z/stages/case30--dataset/attempt-001/`.
  All eight recorded dataset output hashes verified before launch.
- New run:
  `artifacts/gao-pilot-ieee30-seed2027-20261119T064744Z/`.
- Seed 2027; DC OPF; unchanged 0.9 branch-rating scale, renewable placements,
  sampling, feasible-label filtering and training-only normalization.
- 9,425 valid training cases, 929 validation, 1,888/1,881/1,885 test cases.
  Each test split originally attempted 2,000 cases.
- Three 200-neuron tanh hidden layers, batch 64, Adam 0.01.
- M1: **1,000 epochs**, selected final epoch.
- M1-continued, M2, M4 and M5: **200 additional epochs**, each initialized
  independently from the same final M1 checkpoint, with fresh Adam moments.
  All enabled loss weights 1. M1-continued is a training-budget control.
- Primary comparisons use fixed-final checkpoints. Secondary comparisons use
  the best-validation checkpoints saved during those same trainings; neither
  policy is chosen using test accuracy.
- M5-final decoder: two hidden layers of width 128, 1,000 epochs, Adam 0.001,
  seed 2027, physical reconstruction MAE and validation-only 95th-percentile
  threshold. Decoder widths, norm and calibration remain assumptions.
- A separate fixed-6-MW-MAE gate is specified as a norm-sensitivity comparison,
  not an exact recreation of the paper's unspecified decoding score.
- Thirty raw evaluation passes (five methods × two checkpoint policies × three
  regimes), six M5 hybrid passes; **eight evaluation workers**.
- No new scenarios, learning-rate sweep, German training, DeepOPF/M3, FGSM,
  loss rescaling, feasibility repair, online retraining or test-based tuning.

The source is archived before execution. Existing test splits have already
been inspected, so this is a diagnostic controlled comparison, not a fresh
confirmatory generalization test. The decoder is evaluated diagnostically even
if regression fitting remains inadequate; its acceptance is not a certificate.

## Paper reference

Gao et al., DOI
[10.1109/TNNLS.2024.3437741](https://doi.org/10.1109/TNNLS.2024.3437741),
p. 8055, Table II:

| Method | 35.58% | 40.71% | 47.24% |
|---|---:|---:|---:|
| M1 | 99.94% | 89.59% | 69.31% |
| M2 | 99.89% | 96.18% | 88.95% |
| M3 / DeepOPF | 99.94% | 88.21% | 72.57% |
| M4 | 99.96% | 99.15% | 92.66% |
| M5 | 99.92% | 99.16% | 97.94% |

Individual-source accuracy is the inferred comparison metric, with strict
error < 1 MW; joint-scenario accuracy and strict feasibility will be reported
separately. See the [full-text audit](paper-protocol.md) for evidence, details
and remaining ambiguities. Table V has no IEEE-30 decoder row, so the new
IEEE-30 gate results cannot directly reproduce a published Table V entry.

## Execution and integrity

- Frozen source: **`d55faa58890683f9ec423e42853d4eb14b95a4d8`**, containing the
  implementation introduced in `557019c` and this predeclared protocol.
- Start **06:55:27 UTC**, finish **07:06:46 UTC**, 2026-11-19.
- All **39 outer driver stages** completed; the training driver stage contains
  the five individual model trainings.
- All **135 recorded required output hashes** verified afterward.
- All continuations' initializer hashes match the same final M1, epoch 1,000.
  Train/validation-data identity verification passed for every warm start.
- No stage, evaluation PF audit or numerical fallback failed. This does not
  mean that learned outputs were feasible.
- Preflight: **89 tests passed**, `pip check` clean.
- Existing datasets and historical models were not modified. No German
  benchmark was included in this pilot.

Approximate command wall times: training 354.40 s, decoder training 53.12 s,
fixed-threshold calibration 1.88 s, raw evaluations 220.73 s, hybrid evaluations
46.89 s. Eight-worker timings measure throughput, not isolated latency.

The run directory contains manifests, frozen source/archive, training histories
and three checkpoint files per method, decoder artifacts, predictions,
`evaluation_summary.json` and `results.csv`. Post-run `analyze_results.py` and
`analysis.json` record fixed-checkpoint Jacobian/gradient diagnostics and paired
bootstrap calculations. These diagnostics do not update weights.

## Primary fixed-final results

Strict **individual-source error < 1 MW**, conditional on reference-valid
scenarios; six source outputs, including slack, per scenario:

| Method | IID 35.58% | OOD 40.71% | OOD 47.24% |
|---|---:|---:|---:|
| M1 | 84.72% | 84.10% | 82.46% |
| M1-continued | 88.59% | 87.78% | 87.42% |
| M2 | 47.92% | 45.31% | 44.16% |
| M4 | 43.71% | 41.03% | 38.61% |
| M5 | 37.37% | 34.10% | 31.01% |

Direct highest-regime comparison with Table II:

| Method | Paper | This pilot | Difference |
|---|---:|---:|---:|
| M1 | 69.31% | 82.46% | +13.15 pp |
| M2 | 88.95% | 44.16% | -44.79 pp |
| M4 | 92.66% | 38.61% | -54.05 pp |
| M5 | 97.94% | 31.01% | -66.93 pp |

The paper's M5 improves over M1 by **28.63 percentage points** here; our M5
is **51.45 points worse** than M1. A paired, independent-scenario bootstrap
(5,000 resamples) gives a 95% interval of **[-53.25, -49.65] pp** for this
within-pilot difference. This interval does not cover variation across training
seeds, datasets or implementations, and does not undo prior inspection of the
test sets.

M1-continued improves over M1 without physics terms. Thus, the physics-method
performance loss is not explained merely by assigning those methods 200 more
epochs. It still requires a controlled optimization/representation diagnosis;
this is not evidence that the published method itself is ineffective.

### Joint accuracy and physical feasibility are different metrics

Pooled across 5,654 reference-valid test scenarios:

| Method | Source-wise <1 MW | All-source <1 MW | MAE MW | RMSE MW | Mean absolute balance MW |
|---|---:|---:|---:|---:|---:|
| M1 | 83.76% | 51.49% | 0.6215 | 1.0494 | 2.0994 |
| M1-continued | 87.93% | 63.60% | 0.5073 | 0.9668 | 1.5803 |
| M2 | 45.80% | 2.53% | 2.1279 | 4.0033 | 5.7620 |
| M4 | 41.12% | 0.80% | 2.2717 | 4.2384 | 8.3993 |
| M5 | 34.16% | 12.82% | 2.4461 | 3.9849 | 10.7554 |

**Every raw learned checkpoint evaluated, including the secondary
best-validation checkpoints, has 0% strict feasibility.** Balance tolerance
is 0.0001 MW; source/branch-bound tolerance is 0.001 MW. The paper's Fig. 6 also
allows residual branch violations, so its accuracy is not a zero-violation
certificate, but this pilot plainly does not produce deployable raw dispatch.
Feasible-only raw objective gaps are unavailable, not zero.

At 47.24%, M5 has **321 branch-scenario violation events across five distinct
branches**, mean excess 1.088 MW among events. Do not equate these 321 events
with the paper's reported three for M5. The later public-source audit identifies
an event count **above RATE_A + 1 MW after slack rebalance**, not a distinct-branch
count. This pilot uses tighter tolerance and 0.9-scaled ratings, so those raw
counts still do not provide a like-for-like Fig. 6 comparison.

## Checkpoint sensitivity and fitting failure

The predeclared secondary arm evaluates the best-validation checkpoint saved
during each *same* training:

| Method | Best epoch | IID source accuracy | OOD 40.71% | OOD 47.24% |
|---|---:|---:|---:|---:|
| M1 | 17 / 1,000 | 91.80% | 90.37% | 88.59% |
| M1-continued | 60 / 200 | 90.73% | 89.39% | 88.20% |
| M2 | 1 / 200 | 70.64% | 70.50% | 69.56% |
| M4 | 2 / 200 | 45.25% | 44.87% | 42.92% |
| M5 | 1 / 200 | 39.93% | 38.53% | 35.98% |

This does **not** test warm-starting from the best M1. Every continuation still
starts from the predeclared final M1. Nevertheless, simply reporting saved
best-validation weights does not close the observed gap.

Several observations rule out a purely OOD-only explanation:

- Final M1 source accuracy is **86.69% training / 84.84% validation**.
- Final M5 source accuracy is **36.74% training / 37.78% validation**.
- M1 validation MSE is best at epoch 17 (**0.4629 MW²**) and ends at
  **0.9325 MW²**. More epochs at this configuration did not establish the
  paper's near-perfect IID baseline.
- All continuations start with validation RMSE **0.9657 MW**. After just one
  epoch, M2/M4/M5 validation MSE is **2.5776 / 4.9467 / 4.9990 MW²**,
  compared with **0.9325 MW²** at initialization.
- Final M5 validation MSE reaches **12.4968 MW²**.

For historical context, the old 100-epoch, learning-rate-0.001 baseline attained
98.45% IID and 97.77% highest-regime source accuracy on the same tests. This
is **not** an isolated learning-rate ablation: schedule, checkpoint policy and
shuffle-stream handling also differ. It does show that copying the paper's
epoch counts/rate into the current numerical representation is not sufficient.

### Saved-weight optimization diagnostics

Read-only diagnostics on all training inputs find:

| Saved model | Balance-Jacobian loss L2 | Last hidden activations with `abs(tanh)>0.99` |
|---|---:|---:|
| New final M1 | 154.393 | 99.862% |
| New best M1 | 0.719 | 98.176% |
| New final M5 | 383.035 | 99.721% |
| Old baseline, lr 0.001 | 0.667 | 80.604% |

The long high-rate run ends with highly saturated hidden representations and
large, poorly controlled derivatives. On one reproducible 64-case ordinary
training batch at final M1, the hypothetical unit-weight M5 physics-gradient
norm is about **3,387 times** the L0 gradient norm, dominated by L5 at perturbed
inputs. This is a fixed-checkpoint diagnostic, not a measurement of all training
updates; Adam's coordinate normalization also means it is not an update ratio.

Two valid reference cases contain 96.19% of the reference Jacobians' squared
magnitude, with derivatives up to 1,555.846 MW/MW. Independent finite differences
validate those extreme derivatives. They are not automatically bad labels.
Moreover, large physics gradients occur even in the diagnostic batch that
excludes both cases: **the sensitivity tail alone does not establish the cause
of collapse**. See [the sensitivity-tail check](paper-protocol.md#ieee-30-sensitivity-tail-diagnostic).

These observations support optimization/conditioning problems under our chosen
activation, normalization, minibatches, loss reductions and learning rate.
They do not isolate one cause, establish a TensorFlow/PyTorch bug, or justify
silently clipping labels or increasing physics weights.

## Decoder and hybrid results

The M5-final decoder trained for 1,000 epochs; its best-validation checkpoint
is epoch **23**. This selection rule is a declared assumption, not established
as the authors' rule. Its validation-only 95th-percentile physical-MAE
threshold is **1.01633 MW**.

| Regime | Accepted / valid | Neural coverage | Accepted source accuracy | Hybrid source accuracy | Hybrid feasibility |
|---|---:|---:|---:|---:|---:|
| 35.58% | 1,783 / 1,888 | 94.44% | 37.78% | 41.24% | 5.56% |
| 40.71% | 1,774 / 1,881 | 94.31% | 34.89% | 38.60% | 5.69% |
| 47.24% | 1,752 / 1,885 | 92.94% | 32.18% | 36.97% | 7.06% |

Every feasible hybrid output comes from numerical fallback, not the accepted
network. There are no failed fallbacks on this reference-valid subset.
At the highest regime, hybrid all-source 1 MW success is only **17.35%**.
None of the predefined nonempty validation-derived reconstruction-threshold
rows obtains high accepted accuracy. A reconstruction gate does not repair the
poorly fitted IID regressor.

The predeclared **6 MW physical-MAE gate accepts all 5,654 cases**, invokes no
fallbacks, and leaves predictions unchanged. This is a norm-sensitivity result,
not a disproof of the paper's decoder: a mean absolute norm versus a summed
absolute norm differs by 30× for 30 inputs, before considering normalization.
The article does not sufficiently define this score. Table V has no IEEE-30
row, so no direct Table V accuracy reproduction can be claimed.

## Does this count as a faithful reimplementation?

**Not yet as a validated implementation of the reported method and results.**
It is a closer, reproducible implementation of the published algorithmic
skeleton, with an unsuccessful numerical reproduction.

| Aspect | Assessment |
|---|---|
| DC quadratic OPF and L0–L5 derivative identities | General mathematical form matches; chain-rule/second-derivative tests and reference finite-difference checks pass. |
| Three 200-neuron layers; M1 1,000 epochs; 200-epoch common-M1 continuations; Adam 0.01; unit weights | Verified paper settings now followed for IEEE-30. |
| Renewable penetration definition and operating distribution | **Major unresolved issue.** Our denominator is nominal demand; the paper's three rates exactly round from 185/230/300 MW renewables over total installed capacity with 335 MW conventional capacity. This is a strong hypothesis, not author-verified. |
| Branch limits and feasible training count | Additional 0.9 rating stress is not a stated paper setting. We retain only 9,425 valid labels from 10,000 attempts, versus the paper's 10,000 collected samples; its failure/resampling policy is unclear. |
| Activation, batch size, initialization, normalization and loss reductions | Unverified assumptions, with measured numerical consequences here. Equal nominal weights do not establish matching loss/gradient scales. |
| Decoder | Depth/rate/training budget align, but widths, checkpoint selection, reconstruction norm and quantile rule are not established matches. Copying a scalar 6 MW threshold does not fix this. |
| Accuracy metric | Source-wise inference is strongly supported by Table V; vector reduction remains underdefined. Both source-wise and joint metrics are reported rather than silently conflated. |
| Full empirical study | M3/DeepOPF, FGSM attacks, other test systems, multi-seed validation and prospective online adaptation are outside this pilot. |

The current M1's IID-to-highest-regime accuracy drop is only **2.26 percentage
points**, versus **30.63 points** in the paper. Both baseline fit and OOD
severity therefore differ materially. The paper's M5 improvement, >98%
accepted accuracy, and raw/hybrid speed claims have not been reproduced.
Different CPU/GPU/framework/timing protocols also prevent a controlled runtime
comparison; hardware alone is not evidence of a correctness error.

### Highest-value next work

1. Clarify the authors' penetration formula and numerical preprocessing before
   another full claimed reproduction. The 185/230/300 MW hypothesis should be
   author-checked or tested as a separately declared data-definition arm, not
   silently substituted after observing results.
2. Establish accurate, stable M1 training on training/validation data. Isolate
   learning rate, batch size, activation/initialization, numerical units and
   reduction conventions; compare final versus best-M1 initialization explicitly.
   Do not assume "more epochs" or stronger physics penalties is the remedy.
3. Add guidance from the same adequately fitted M1, with gradient diagnostics
   and validation-only decisions. Any label clipping or robustification changes
   the experiment and must be a named ablation.
4. Resolve decoder norm/scaling and evaluate accepted accuracy together with
   coverage, uncertainty, physical violations and full fallback cost.
5. Reserve fresh tests and multiple seeds before a confirmatory claim; add the
   omitted baselines/attacks/systems after the core reproduction works.

No hyperparameters, labels, thresholds or operating assumptions were retuned
using these results, and no additional training run was launched to hide the
negative outcome.
