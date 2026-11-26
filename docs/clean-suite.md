# Clean-slate multi-grid DCOPF suite

This is the new experimental cohort for the report. It does not load archived
PowerSim datasets, numerical labels, checkpoints, predictions, or test scores.
The runner snapshots current source bytes (including uncommitted project
changes), hashes the source and stage inputs/outputs, and records the exact
versions and settings in `suite_manifest.json`. No Git commit is created by the
runner. Do not pool clean rows with historical ones.

## Registered protocol

| Benchmark | Test nameplate rates (%) | Basis |
|---|---|---|
| IEEE-30 | 35.58, 40.71, 47.24 | Total installed; reconstruction's explicit 185/230/300 MW hypothesis |
| Germany-30 v2 | 35.58, 40.71, 47.24 | Nominal gross demand; separate transfer experiment, 90% ratings |
| IEEE-57 | 34.28, 39.51, 45.96; stress 50.31 | Total installed proxy |
| IEEE-118 | 32.11, 33.91, 38.78; stress 44.26 | Total installed proxy |
| IEEE-300 | 19.50, 23.27, 26.05 | Total installed proxy |
| PEGASE-1354 | 18.89, 21.85, 24.59 | Total installed proxy |

IEEE rates are from Gao et al., [DOI 10.1109/TNNLS.2024.3437741](https://doi.org/10.1109/TNNLS.2024.3437741),
Tables II/III (plus the two high stress rates from Table V). The IEEE-57 low
rate is printed as 34.28% in Table III and 34.25% in Table V; we use 34.28%.
We treat the lowest printed rate as IID and the higher ones as OOD. The paper
does not disclose the exact renewable capacity denominator, placements,
generator revisions, cross-system training schedule, or all operating choices.
The `renewable nameplate / (renewable nameplate + conventional Pmax)` convention
and source-informed IEEE-30 training protocol are transparent extensions, not
verified author settings for the larger networks. Ratings of the bundled IEEE
and PEGASE cases remain unscaled. For Germany see [the v2 provenance and license](germany30.md).

- 10,000 validated training rows, 1,000 validation rows, and 2,000 test
  rows **per regime**; at most five attempts per target row. Failed OPFs
  and attempts to reach a quota remain in the data logs. Independent
  preflight draws must have at least one validated label in every split before
  the corresponding full grid runs. A failed grid is recorded as unavailable;
  its data and results are not synthesized. Nominal failures also get a
  SciPy/HiGHS feasibility diagnostic under the unrevised DC constraints.
- Data/placement seed **20261201**, held-out seed **20261202**, independent
  preflight data/test seeds **20261203/20261204**. A demand-only test keeps
  train renewable placement, nameplate and weather distributions and shifts the
  mean relative gross demand from 1.0 to 1.10 with the same 0.2 SD.
  Predeclared wind/PV installation pairs are 2/2/4/8/16/32 in table order to
  avoid concentrating the entire nameplate at a few buses on larger networks.
  IEEE-300 and PEGASE keep their built-in fixed active injections inside the
  net-load vector; Germany's illustrative snapshot injections are replaced as
  in the benchmark documentation.
- Three independent training seeds **20261211–20261213** use the same data
  on each grid. Each seed starts with a fresh M1 and shares its final weights
  across the 200-epoch continuations; M1 gets 1,000 epochs. Run M1,
  M1-continued, M2, M3-example, M4 and M5 (PowerSim analogues).
  **M3-example is not DeepOPF.** This suite freezes one source-informed
  protocol rather than selecting among the exploratory profiles on test scores.
- A separate frozen-feature decoder per method and seed uses 1,000 epochs,
  validation-q95 physical RMSE, and a secondary explicit 6 MW RMSE gate for
  M5. The paper's gate norm is not recovered. Each evaluation saves both raw
  and hybrid dispatch, including fresh OPF fallback outcomes.
- As in the IEEE-30 study, the IID and first OOD cohorts also get the
  previous 1 MW per-coordinate self-FGSM diagnostic with both original-target
  and freshly solved reference conventions. Attacks are not performed on the
  larger systems or Germany; the paper does not report those attacks there.
- OPF generation and evaluation each use **four independent workers**,
  one numerical thread each. Model and decoder fit sequentially, each with
  **four PyTorch CPU threads**. The optional matched-latency analysis uses
  isolated single-CPU-thread inference, not sums of parallel worker timings.
- To fit the larger cases, reference Jacobians are retained only for training;
  all test dispatch/flow/attempt records are still kept. The training-side
  dense-layer chain rule and masked branch reduction replace materialization
  of unused derivatives without changing the loss equations; value and
  parameter-gradient parity are tested. JSONL records are compressed.

## Commands and artifacts

```sh
source .venv/bin/activate
RUN=artifacts/clean-suite-YYYYMMDD
python -m powersim.research.clean_suite "$RUN" --plan --workers 4
python -m powersim.research.clean_suite "$RUN" --workers 4
```

Use a new output directory. `--smoke` uses a different data seed, tiny
budgets and M1/M5 only: it is not a report result. The full run is lengthy and
can be monitored using `$RUN/runner.log`, `$RUN/status.json` and
`$RUN/preflight_summary.json`. Each case's `stages/.../attempt-NNN` directory
and log are preserved on failure. Stage outputs are hash-checked on resume:

```sh
REPO="$PWD"
RUN="$REPO/artifacts/clean-suite-YYYYMMDD"
(cd "$RUN/source" && "$REPO/.venv/bin/python" -m powersim.clean_suite "$RUN" --workers 4 --resume)
```

Resume only after the previous runner **and** its worker processes have
stopped; it is stage-level, not epoch-level. If a hard kill leaves a lock, check
the recorded PID(s) before removing only the stale lock. A completed suite
with a blocked grid says `complete_with_unavailable`, not that all grids ran.

After completion, produce a report containing **only** the new cohorts:

```sh
python -m powersim.output_generation.report_evaluation --clean-suite "$RUN" \
  --output artifacts/clean-report-YYYYMMDD
```

Add `--benchmark-runtime` to time matched numerical, neural and hybrid
workloads freshly using the first registered seed. Historical run roots,
their source snapshots and the prior report are in `artifacts/archive/`.
The historical report reducer recognizes the archive layout automatically.
