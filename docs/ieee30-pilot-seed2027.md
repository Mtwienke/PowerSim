# IEEE-30 shared-M1 pilot: seed 2027

**Protocol fixed before execution on 2026-11-19; results pending.**
Requested scope: IEEE-30 only; Germany-30 is excluded.

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

## Execution and analysis

Pending execution. Preflight: 89 tests passed; `pip check` clean. Results,
checkpoint sensitivity, decoder coverage, physical violations and reimplementation
fidelity will be added here after the frozen run completes.
