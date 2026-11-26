# Gao public-example source audit

Audited **2026-11-19**. This supersedes earlier statements that no original
example code was available. Historical runs and their measured results remain
unchanged. **Germany is excluded from this investigation.**

**Later user decision:** proceed without the expired share or author contact.
The [predeclared reconstruction suite](gao-reconstruction-suite.md) implements
explicit hypotheses and separate numerical variants. The audit's original
“wait for source” recommendation is no longer a blocker, but the evidence
classification and paper/source conflicts remain unchanged.

## Source and authority

[AI4POWER/Data-Driven-OPF-Method-Considering-Inherent-Pattern][repo] explicitly
describes itself as example code for the target paper. Its initial commit is
attributed to `maosheng`; the README links a separate full project/data share.
We pinned **`e775ce3a24af0408aed9c02236abd862707cbae0`**, dated 2024-06-06.
There are two commits, one branch, and no forks at inspection time. The initial
2024-05-31 tree contains the same Python files; June 6 adds the README.

The complete pinned tree contains **17 files**: IEEE-30 and IEEE-57 examples,
plots, one branch evaluator and two IEEE-57 decoder scripts. **No dataset,
data-generation script, saved model, dependency lock or generator-accuracy/gate
evaluation script is included.**

The journal remains the primary specification. These examples predate acceptance
(2024-07-29) and conflict with several explicit journal settings. We therefore
distinguish **verified example behavior**, **journal statements**, **inferences**
and **our implementation choices**. Combining whichever settings look favorable
would not establish reproduction of either version.

All 17 downloaded files were checked against their Git blob IDs; SHA256 values
are recorded in [the provenance manifest](gao-source-provenance.json). A local
reference cache is under `artifacts/gao-source-audit-20261119/`; upstream code was
**not executed or copied into application code**. GitHub reports no license and
the tree contains none, so it is not vendored into the tracked project.

## Verified numerical differences

Line numbers below refer to the immutable revision above, not moving `master`.

| Setting | Public IEEE-30 examples | Completed seed-2027 pilot / journal |
|---|---|---|
| Hidden network | Three 200-unit **ReLU** layers, linear six-output head; [M1 lines 50–63][m1] | Pilot tanh. Width/depth match journal; activation is not specified there. |
| Minibatch | **50** in the main blocks; the function's unused default is 15 | Pilot 64; journal silent. |
| Inputs and targets | JSON `pd` and `pg` passed directly as float32; no centering, fitted standardization or active output-rescaling layer | Pilot training-only input/output standardization, raw-MW public forward. |
| Physical conversion | Evaluator multiplies **both** network inputs and outputs by **100** before PYPOWER; [lines 145–151][violation] | Verified consumer expects units of 100 MW. This strongly establishes a per-unit interface; missing data generator still prevents checking JSON production. |
| Prediction loss | **MAE**, despite variables/log headings saying `mse` | Pilot MW² MSE; journal Eq. (11) explicitly says MSE. **Conflict.** |
| Sensitivity loss | Squared Jacobian residual, masked before a mean over **all batch × generator × input entries** | Pilot divides by the number of available entries. Both use squared residuals, but denominators differ when masked. |
| Sensitivity filter | Keep sample's `coff` only if its **sum is nonzero** and its **signed maximum < 10**; otherwise mask L1 to zero; [M2 lines 23–44][m2] | Pilot keeps every available, regular KKT Jacobian, with no magnitude cutoff. Filter affects L1, **not** L0 or sample inclusion. |
| Balance derivative | **MAE** of `sum_g J - 1` | Pilot and journal Eq. (13): mean square. **Conflict.** |
| Active generator derivative | **Not implemented** in the visible IEEE-30 M4/M5 updates | Pilot implements L3; journal Table I and Eq. (14) require it. **Conflict.** |
| Active branch derivative | **MAE**, only branch row **20**, gated by `branch_high_20` / `branch_low_20` and averaged over the whole batch | Pilot squared residual over all locally active branches, normalized by active entries. Journal Eq. (15) is squared. **Conflict and example-specific restriction.** |
| Exploration | M5 adds independent `Normal(0,1)` in **network input units**, then MAE balance-derivative loss; [lines 95–143][m5] | Pilot Gaussian std `0.1 × training input scale` in MW, squared loss. Under the demonstrated 100 MW interface, example std corresponds to **100 MW**, not 1 MW. Final-paper perturbation magnitude is unspecified. |
| Adam rate | **0.001**, `decay=2e-6` | Pilot/journal OPF rate **0.01**. **Conflict.** |
| Optimizer continuation | New optimizer, `load_weights` from M1; no optimizer restore | Pilot fresh Adam agrees with this behavior. |
| Epochs/checkpoint | M1 1,000; M2/M3/M4/M5 200; save model **after final epoch**; tests logged but no best-test selection in visible loop | Pilot primary final checkpoint and IEEE-30 schedule agree. Extra best-validation files remain separately labelled. |
| Reproducibility | No explicit NumPy/TensorFlow seed or pinned environment | Pilot deterministic seed 2027; the final paper does not give a seed. |

The commented-out `Last_Layer` is **not** evidence of output clipping or scaling.
The unused `si_coff=0.1` variable in IEEE-30 M4/M5 is **not** their gradient
weight: inspect the actual additions, which use unit weights.

### Exact visible IEEE-30 objective structure

To avoid falsely calling these the journal's losses, use different notation:

- `A0`: generation MAE in network output units.
- `S1`: masked squared sensitivity residual, mean over the full tensor.
- `A2`: absolute balance-Jacobian residual.
- `A4`: absolute branch-20 derivative residual, zeroed for inactive samples,
  mean over the full batch and all input dimensions.
- `A5`: absolute balance-Jacobian residual at Gaussian-perturbed inputs.

Then the visible update directions correspond to:

| Example | Objective whose gradients are added |
|---|---|
| M1 | `A0` |
| M2 | `A0 + S1` |
| M4 | `A0 + S1 + A2 + A4` |
| M5 | `A0 + S1 + A2 + A4 + A5` |

The M5 **returned/logged total omits A5**, although its gradients are applied.
Its disturbance-history list also appends the sensitivity loss instead of the
returned disturbance loss. Hence a CSV column is not sufficient evidence of
the optimized objective. These are static code observations, not executed
parity tests.

Do not infer a uniform cross-system protocol from these IEEE-30 scripts.
The IEEE-57 examples differ further: M4 weights its branch gradient by 0.1;
M5 uses sensitivity/exploration/branch weights 0.1 and balance weight 0.001,
trains for 1,000 epochs and has **no M1 load-weights call**. Its data path names
an extension with 13,313 records, unlike M1's 10,000-record path. Those filenames
are not a count verified from actual data. The journal instead describes
common-M1 initialization and unit weights in the IEEE-30 study.

`M3.py` adds a normalized squared branch-flow excess penalty to supervised MAE
and warm-starts from M1. This alone does not establish a complete faithful
implementation of the external DeepOPF algorithm cited by the journal. **The
complete external M3 remains unimplemented**. The later reconstruction includes
the explicitly labelled `M3-example` comparator, not a claim of full DeepOPF
or a leave-one-out replacement; see the
[completed results](gao-reconstruction-results.md).

### Framework-dependent deductions, not recorded training metadata

The examples leave Keras initializer and Adam epsilon unspecified. Under the
journal's TensorFlow/Keras 2.8 context:

- [Dense defaults][dense-defaults] are Glorot-uniform kernels and zero biases,
  unlike our PyTorch `Linear` defaults.
- [Adam defaults][adam-defaults] include epsilon **1e-7**, unlike PyTorch's
  **1e-8**. The epsilon conventions also differ; merely substituting the same
  number does not prove identical update rules.
- [Legacy `decay`][decay] is inverse-time learning-rate decay,
  `lr(step) = initial_lr / (1 + 2e-6 × step)`, **not weight decay**.

For 10,000 samples, batch 50, 1,000 epochs, the final update's nominal rate
would be approximately **0.0007143**, with a newly reset continuation optimizer.
This arithmetic is conditional on that sample count/version; there is no
upstream package lock or optimizer checkpoint to verify it directly.

## Decoder evidence and remaining score ambiguity

The IEEE-57 [feature extractor][features] returns the **third/final hidden layer**,
immediately before the output head. Our decoder consumes the **second of three
hidden layers** (`hidden[-2]`), following the journal's phrase “penultimate
hidden layer.” Those are not the same feature representation; shared width 200
can conceal this difference.

The [decoder example][decoder] uses **three** ReLU hidden layers of width **200**,
linear 57-output reconstruction, batch **500**, **MSE** training,
Adam **0.001 / decay 2e-6**, **1,000 epochs**, and final-epoch saving.
The journal instead says **two hidden layers**. These IEEE-57 settings are not
automatically evidence of the unpublished IEEE-30 decoder settings.

Neither decoder script implements the gate or calculates a per-case 6/7 MW
reconstruction score. MSE training does **not** determine the inference score:
MAE, summed absolute error, RMSE and maximum error remain distinguishable.
Unused `thre` variables do not resolve this. The exact gate norm/reduction,
physical conversion and threshold protocol therefore remain **unknown**.

The public files also do **not** calculate the tables' generation-accuracy
probability. `draw_figure.py` plots per-case mean generator errors, not an
accuracy aggregation. Generator-wise `<1 MW` remains a strong inference from
Table V, not newly verified by this source.

## Branch-violation metric: a material clarification

[The public evaluator][violation] uses ordinary `case30()`, runs DCPF after
inserting predictions and lets its slack rebalance. It counts exceedances of
**`RATE_A + 1 MW`**, not exact thermal limits. The reported count is
`sum(violate_rate)` over **branch × scenario events**, not the number of distinct
branches. Its mean excess is above that enlarged bound. Comments give counts
14/7/4/6/3, matching the figure's method sequence.

This is source evidence for the Fig. 6 convention, though not a proof that the
unavailable final-paper run used identical data/settings. It supersedes our
earlier blanket statement that branch-event versus distinct-branch aggregation
was unresolved. Our existing strict PF audits additionally check the **raw**
unbalanced proposal and use 0.001 MW branch tolerance, with 0.9-scaled ratings.
Their counts cannot be compared directly with Fig. 6 without a separately
labelled reporting-convention comparison. **Do not weaken strict feasibility.**

## Penetration and data generation: still unresolved

No inspected file defines the renewable-penetration denominator, renewable
capacity allocation or installation locations, wind/PV conversion curves,
random dependence, exact load sampler, OPF-failure policy or label-generator
units. Filename rates are labels, not their definition.

The candidate `R / (335 + R)` still reproduces 35.58%, 40.71% and 47.24% from
**185/230/300 MW** after rounding. This is an **arithmetic hypothesis**, not a
confirmed author setting. The examples additionally mention **25.56%** and
**50.7%** directories, which this three-rate coincidence does not by itself
explain. Do not retrofit an arbitrary capacity list to more rounded labels.

The README's full-project link is:
<https://pan.baidu.com/s/1tcS1bTrQxFkf6N-MYCu7Qw>, extraction code **`39oo`**.
Anonymous fetches, including `?pwd=39oo`, did not expose files. The raw HTML
reports a share-error page (`errno=117`, `errortype=10`) titled
“百度网盘-链接不存在” (“link does not exist”), alongside a login prompt.
This establishes **unavailable through our anonymous access**, not a verified
permanent deletion or proof that an authenticated user cannot access it.
No access control was bypassed and no author was contacted automatically.

## Measured read-only diagnostic: seed 2027

Executed the diagnostic from **`9a3de55`** against the unchanged existing
IEEE-30 dataset and the pilot's **final M1 and final M5**, without training.
Artifact: `artifacts/gao-source-audit-20261119/preprocessing/audit.json`.

### Sensitivity-label policy

| Split | Supervised cases, unchanged | Example-kept Jacobians | Additionally masked Jacobians | Share of reference squared magnitude masked |
|---|---:|---:|---:|---:|
| Training | 9,425 | 8,901 | **524** | **98.5374%** |
| Validation | 929 | 876 | **53** | **65.4717%** |

All those sensitivities are available under our existing regular-KKT policy.
The source's magnitude predicate is an additional learning mask, **not** evidence
that the labels are invalid. The earlier finite-difference-validated extremes
are among the masked rows. No labels, availability flags or supervised cases
were changed. Validation rows are inspected diagnostically, not used for
Jacobian-loss training.

The 98.54% figure describes **reference-label magnitude**, not actual trained
residuals. At final M1, applying the mask and full-tensor denominator to the
same predicted training Jacobians changes the L1 snapshot from **19.4287 to
9.0838**, not a 98.54% decrease: the model's own derivatives are also poorly
controlled. Thus the filter is a major mismatch, **not an isolated causal
explanation or demonstrated remedy**.

### Units and reductions at unchanged saved weights

Full-training-split snapshots, **not online/minibatch logged losses**:

| Saved checkpoint | L0 MSE, MW² (pilot) | L0 MAE, per-unit (example convention) | L1 pilot / example reduction | Balance MSE / MAE |
|---|---:|---:|---:|---:|
| Final M1, epoch 1,000 | 0.610726 | 0.005231 | 19.4287 / 9.0838 | 154.3927 / 2.1727 |
| Final M5, continuation epoch 200 | 13.325761 | 0.022534 | 28.7181 / 14.0453 | 383.0354 / 2.6832 |

These are counterfactual numeric objectives on the **same tanh networks**,
not a source-compatible retraining or evidence that source-trained ReLU
networks would have these losses. In particular, changing supervised units
alone does not guarantee a better balance of loss gradients.

For a common 100 MW input/output base,
`d(Y/100)/d(X/100) = dY/dX`: per-unit conversion leaves physical Jacobian values
unchanged, while supervised MSE changes by **1/10,000** and MAE by **1/100**.
The historical pilot's input-noise standard deviations range from
**0.04805 to 0.54421 MW**. The example's `stddev=1.0` would be **100 MW** under
its demonstrated interface. This large discrepancy is another reason not to
call an unmodified example a verified final-paper protocol.

### Integrity and limits

- **101 tests passed**, including signed versus absolute maximum, strict
  cutoff, nonzero-sum filtering, masking denominators, per-unit loss conversion,
  no held-out-array access, read-only model checks, and Germany exclusion.
- A fixed-positive ReLU test confirms zero input Hessian with **nonzero mixed
  input/parameter gradients** for Jacobian learning. ReLU is not ruled out by
  its zero Hessian; no trained ReLU result is claimed here.
- The diagnostic verifies matching train/validation hashes and unchanged model
  states/files. All **135 original pilot output hashes** were rechecked after
  the audit and remain unchanged.
- No new benchmark, decoder training, held-out metric selection or penetration
  sensitivity experiment was run.

## Decisions before another run

### Reproducible read-only preprocessing diagnostic

```sh
python -m powersim.research.gao_audit DATASET NEW_AUDIT_OUTPUT \
  --model MATCHING_M1/model.pt --model MATCHING_M5/model.pt
```

Omit `--model` for a NumPy-only label-policy audit. The command is restricted to
IEEE-30 DC, reads **training/validation arrays only**, verifies checkpoint/data
identity when models are supplied, and refuses existing outputs or outputs
inside the input dataset. It records input/model/code hashes and package
versions. It never starts an optimizer, generates data, alters a mask used by
training, modifies defaults, or evaluates held-out accuracy.

The output compares the signed-max sensitivity predicate, reference-label
energy, physical versus per-unit supervised error, and full-population L1/L2
reductions at identical saved predictions/weights. These are **diagnostic
counterfactual loss values**, not a trained source-protocol result, minibatch
gradient measurement or isolated explanation of the earlier collapse.

### Implementation decisions

1. **Preserve the journal M1/M2/M4/M5 mapping and existing defaults/artifacts.**
   Discovery of MAE examples does not license silently rewriting squared
   journal losses or omitting L3.
2. Record the example's sensitivity policy and units as read-only diagnostics
   on training/validation data. Masking a Jacobian is not deleting a supervised
   case, clipping its values or changing its physical validity.
3. If testing numerical settings, predeclare separate **journal-equation** and
   **public-example diagnostic** arms. ReLU/batch 50/per-unit representation are
   evidence-backed candidate controls, but their performance is not established
   by inspection. Keep rate/loss/masking/branch-selection conflicts explicit.
4. Obtain the full project or clarification before changing penetration,
   renewable sites, conversion curves or label-generation policy. Request a
   readable data-generation script and at least one JSON sample containing
   `pd`, `pg`, `coff`, `info`, plus the network modifications.
5. Ask which code revision/settings produced Tables II/V, and clarify the
   learning-rate/loss/L3/decoder discrepancies and generation-accuracy/gate
   reductions. A draft is in [the clarification checklist](gao-author-questions.md).

**No new benchmark training, held-out tuning, regenerated data or German run
was launched for this audit.**

[repo]: https://github.com/AI4POWER/Data-Driven-OPF-Method-Considering-Inherent-Pattern/tree/e775ce3a24af0408aed9c02236abd862707cbae0
[m1]: https://github.com/AI4POWER/Data-Driven-OPF-Method-Considering-Inherent-Pattern/blob/e775ce3a24af0408aed9c02236abd862707cbae0/30-bus/M1.py
[m2]: https://github.com/AI4POWER/Data-Driven-OPF-Method-Considering-Inherent-Pattern/blob/e775ce3a24af0408aed9c02236abd862707cbae0/30-bus/M2.py
[m5]: https://github.com/AI4POWER/Data-Driven-OPF-Method-Considering-Inherent-Pattern/blob/e775ce3a24af0408aed9c02236abd862707cbae0/30-bus/M5.py
[violation]: https://github.com/AI4POWER/Data-Driven-OPF-Method-Considering-Inherent-Pattern/blob/e775ce3a24af0408aed9c02236abd862707cbae0/30-bus/violation_calculation.py#L140-L195
[features]: https://github.com/AI4POWER/Data-Driven-OPF-Method-Considering-Inherent-Pattern/blob/e775ce3a24af0408aed9c02236abd862707cbae0/57-bus/adaptability%20judgement/feature_in_hidden_layer.py#L29-L42
[decoder]: https://github.com/AI4POWER/Data-Driven-OPF-Method-Considering-Inherent-Pattern/blob/e775ce3a24af0408aed9c02236abd862707cbae0/57-bus/adaptability%20judgement/train_decoder_hidden.py
[dense-defaults]: https://github.com/keras-team/keras/blob/v2.8.0/keras/layers/core/dense.py
[adam-defaults]: https://github.com/keras-team/keras/blob/v2.8.0/keras/optimizer_v2/adam.py
[decay]: https://github.com/keras-team/keras/blob/v2.8.0/keras/optimizer_v2/optimizer_v2.py
