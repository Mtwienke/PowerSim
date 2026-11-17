# Verified CPU smoke experiment

Verified 2026-11-18 on macOS arm64, Python 3.13.5, pandapower 3.5.6,
PyTorch 2.15.0, NumPy 2.4.6, SciPy 1.18.1, pandas 2.3.3, without Numba.
Neural training/evaluation used one CPU thread.

These are **plumbing/regression results, not a reproduction of paper accuracy**.
Only 32 training examples and three epochs were used; the learned outputs are
not suitable for operation. No full experiment suite was run.

## Commands

Install with `constraints-ml-macos-arm64-py313.txt`, then use a fresh output root:

```sh
powersim dataset artifacts/final-smoke/data --preset smoke --seed 2026
powersim train artifacts/final-smoke/data artifacts/final-smoke/baseline --method baseline --epochs 3 --hidden 16,16,16 --batch-size 16
powersim train artifacts/final-smoke/data artifacts/final-smoke/physics --method physics-guided --epochs 3 --hidden 16,16,16 --batch-size 16
powersim decoder artifacts/final-smoke/data artifacts/final-smoke/physics/model.pt artifacts/final-smoke/decoder --epochs 3 --hidden 16
powersim evaluate artifacts/final-smoke/data artifacts/final-smoke/baseline/model.pt artifacts/final-smoke/baseline-iid
powersim evaluate artifacts/final-smoke/data artifacts/final-smoke/physics/model.pt artifacts/final-smoke/hybrid-iid --decoder artifacts/final-smoke/decoder/decoder.pt
powersim evaluate artifacts/final-smoke/data artifacts/final-smoke/physics/model.pt artifacts/final-smoke/hybrid-ood --decoder artifacts/final-smoke/decoder/decoder.pt --split ood_test_2
pytest -q
```

Actual experiment files are under the ignored `artifacts/final-smoke/` directory
in the implementation workspace. Commands refuse to overwrite directories.
Numeric arrays are seeded/reproducible in the pinned environment; wall-clock
timing is naturally variable.

## Data and active sets

| Split | Attempts | Valid labels | Cases with active branches |
|---|---:|---:|---:|
| Training, 35.58% nameplate | 32 | 32 | 8 |
| Validation, 35.58% | 8 | 8 | 0 |
| IID test, 35.58% | 8 | 7 | 1 |
| OOD test, 40.71% | 8 | 8 | 1 |
| OOD test, 47.24% | 8 | 8 | 3 |

The one IID OPF failure remains in JSONL and is not a training/test label.
All retained sensitivities were available. No source bounds were active in this
small dataset, so L3 has no selected rows here; separate regression tests
exercise active source bounds and their zero-Jacobian pattern.

## Accuracy and independent feasibility

Success means **all source errors, including slack, are <= 1 MW** for a scenario.
Errors below use reference-valid cases; failed original label attempts are
reported separately and included as unsuccessful in the additional
all-requested denominator.

| Output | Cases | MAE MW | RMSE MW | Max error MW | 1 MW success | Feasible |
|---|---:|---:|---:|---:|---:|---:|
| Baseline, IID | 7 | 3.963 | 6.460 | 22.743 | 0% | 0/7 |
| Physics-guided raw, IID | 7 | 3.970 | 6.452 | 22.731 | 0% | 0/7 |
| Physics-guided raw, highest OOD | 8 | 4.381 | 7.388 | 27.914 | 0% | 0/8 |
| Accepted subset, highest OOD | 7 | 4.685 | 7.840 | 27.914 | 0% | 0/7 |
| Complete hybrid, highest OOD | 8 | 4.099 | 7.334 | 27.914 | 12.5% | 1/8 |

IID decoder acceptance was 100%; highest-OOD acceptance was 87.5%, with 12.5%
fallback and no fallback failures. The successful fallback is the hybrid's only
feasible OOD result. Its objective gap is 0%; objective gaps for the infeasible
learned outputs are **unavailable**, not reported as zero.

Mean absolute raw balance error was 7.559 MW (physics IID), 7.272 MW (raw OOD)
and 6.591 MW (hybrid OOD). There were two branch-violation events in each of
those reports; maximum OOD overload was 3.668 MW. Source-bound violations were
zero. Voltage/Q metrics are unavailable in DC.

These results deliberately illustrate that reconstruction acceptance is **not
a feasibility/accuracy certificate**. The tiny physics-guided model does not
outperform the baseline; no generalization improvement is claimed.

## Measured runtime

One warm-up is excluded. Model loading is excluded; single-scenario decision
latency includes normalization, decoder and fallback when applicable, but not
the separately timed independent PF audit.

| Measurement | Observed |
|---|---:|
| Baseline IID mean single-scenario latency | 0.196 ms |
| Hybrid IID mean latency, no fallback | 0.392 ms |
| Hybrid OOD mean latency, including fallback | 40.879 ms |
| Hybrid OOD total decision time, 8 cases | 327.032 ms |
| Hybrid OOD total fallback time, 1 solve | 324.011 ms |
| Baseline model-only batch, 7 cases | 0.044 ms |
| Physics model-only batch, 8 OOD cases | 0.107 ms |

Tiny batches and concurrent host activity make these timings illustrative only;
they are not speedup claims or substitutes for replicated latency benchmarks.
Batch timing is **not** hybrid throughput.

## Additional verification

- **58 tests passed**, including optional ML tests; no skipped ML module.
- Real DC/AC OPF smoke tests on IEEE 14 and IEEE 30, including actual AC
  reactive quadratic costs.
- Finite-difference sensitivity checks at interior and congested operating
  points, active-bound identities, chain-rule scaling and second derivatives.
- Altering held-out inputs/labels leaves training weights, decoder weights and
  calibration threshold unchanged.
- Loaded-artifact dashboard inference works in DC and is unavailable in AC.
- Online CLI nominal-input inference returned the untouched learned vector.
  A separate three-times-nominal-demand production input was rejected; failed
  OPF fallback returned `status: failed`, `generation_mw: null`, exit code 1
  and a separate production rejection record.
- Explicit threshold-only recalibration to MW² used validation inputs only.
- `pip check`: no broken requirements.

Twelve test warnings are upstream case14 transformer/tap deprecations.
Pandapower 3.5.6 also logs missing-Numba warnings during DC PF despite the
requested `numba=False`; see the [protocol](research.md).

## Publication differences

The full text/original code were unavailable for verifying exact M1–M5
combinations. Methods therefore have descriptive names. Renewable conversion,
nameplate penetration definition, placement/seeds, 90% branch-limit stress,
validation size, activation, optimizer, decoder representation and quantile are
explicit configurable assumptions. This smoke run validates software paths,
not the publication's accuracy, robustness or runtime conclusions.
