# Rebuild the scientific evaluation

This report analyzes completed runs, not the authors' code or telemetry. See
the [methodology audit](report-evaluation-audit.md) and
[metric contract](report-evaluation-metrics.md).

The runs and numbers on this page are historical. The report's newer
primary protocol is documented in [the clean-slate suite](clean-suite.md).

## Reproduce

Run from the repository checkout. The frozen run directories are local,
git-ignored inputs (use `--artifacts DIR` to relocate them). Install `.[report]`
in the existing virtual environment; `--no-figures` only needs the ML extra.

```sh
source .venv/bin/activate
python -m pip install -e '.[dev,report]'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# Inventory only; no output files, inference or solver calls.
python -m powersim.output_generation.report_evaluation --plan

# Optional: generate the small, reference-only current Germany-30 cohort.
# Skip if it already exists.
python -m powersim.cli dataset artifacts/report-germany30-v2-seed2027 \
  --benchmark germany30 --config examples/report-germany30-v2.json --workers 4

# Regenerate into a NEW output directory. No reference OPF calls or training.
python -m powersim.output_generation.report_evaluation \
  --output artifacts/report_evaluation_recheck

# Full workbook with fresh matched-workload CPU timing; no training.
python -m powersim.output_generation.report_evaluation --benchmark-runtime \
  --output artifacts/report_evaluation_timed_recheck

pytest -q
```

The original output is now `artifacts/archive/report_evaluation/`. The
evaluator refuses to overwrite outputs. A directory containing only the
initial audit is allowed. The v2 cohort and historical runs are auto-discovered
under `artifacts/archive/` (or their previous locations under `artifacts/`).
`--runs reconstruction` restricts the reducer to the largest IEEE suite;
`--repeats` and `--runtime-limit` are explicit timing knobs (defaults 3 and
2,000). Historical Germany v1 has fewer than 2,000 valid cases in the timed
OOD2 split; no resampling or duplication pads it. Run fresh timing without
other CPU-intensive jobs. Clock values vary across repeats and machines.

## Outputs and joins

| Output | Meaning |
|---|---|
| `experiment_manifest.json` | Per-dataset, per-checkpoint and decoder provenance, consumed file hashes, source and runtime environment, unavailable experiments |
| `metrics.csv` | One summary per frozen configuration/scenario, plus numerical reference and separate FGSM target rows |
| `sample_metrics.csv` | Scored sample-level results, including raw gate inputs/decisions on hybrid rows |
| `reference_attempts.csv` | All original attempts and solver statuses; distinguish invalid labels from infeasible predictions |
| `matrices/*.npz` | Ordered prediction/ref/error matrices and line/source violations per experiment |
| `threshold_curves.csv`, `comparisons.csv` | Validation-grid test diagnostics and pairwise deltas |
| `historical_seed_summary.csv` | Descriptive joint data/training protocol-seed variation, not isolated seed uncertainty |
| `runtime.csv` | Fresh matched timing and archived timings, distinguished by `measurement` |
| `figures/*.pdf`, `*.tex`, `report_findings.md` | Report-ready plots, LaTeX fragments (`booktabs`), and interpretation |

Use `experiment_id` for metrics-to-matrix joins, `(dataset_id, sample_id)`
for clean model-to-model comparisons, and the attack's `source_sample_id`
for pairing with the corresponding clean split. Only reference-valid
predictions were archived; failed label attempts remain explicit in the
attempt ledger and conservative denominators. Legacy grids require decoder
inference on **validation** inputs only. Newer archived predictions have
ordered NPZ rows and matching frozen score/checkpoint bindings. The reducer
rebuilds each DC map from the *frozen* network and checks it against the
saved maps, solver flows and objectives.

Budget about 0.7 GB for the full report, mostly the uncompressed sample CSV
and compressed error/flow matrices. An independent spot check joined the
first ten raw highest-regime predictions by sample ID to the archived
pandapower PF audits: maximum flow differences were below `1e-10 MW` on
both IEEE-30 (source-informed M5) and Germany-30 v1 (guided). Branch- and
generator-violation event counts and combined feasibility counts also
matched all 327 original evaluation summaries. The
[metric contract](report-evaluation-metrics.md) also includes a hand-worked
example of all three accuracy aggregations.

## Main findings from the completed run

- **IEEE-30, highest renewable regime:** The source-informed M5 analogue
  scores 89.38% generator-wise at strict <1 MW, versus 90.17% for the M1
  analogue and 97.94% for published M5. The M4 analogue scores 92.51%, ahead of
  local M1, so that ordering agrees qualitatively. The published M5 advantage
  does not reproduce. For the *same* M5 outputs, joint and mean-error
  accuracies are 83.40% and 88.20%, and strict raw DC feasibility is 0%.
- **Selective acceptance is not feasibility.** The validation-q95 RMSE gate
  accepts 63.50% of raw M5 OOD2 cases and has 96.06% joint accuracy among accepted
  predictions, but none of them is strictly feasible. The 730 fresh OPF
  replacements bring completed hybrid joint accuracy to 97.50% and feasibility
  to 36.50%. These percentages use 2,000 valid labels from 2,057 attempts. Of
  332 inaccurate raw predictions, 50 are accepted (15.06% false acceptance);
  of 1,668 accurate raw predictions, 448 are rejected (26.86% false rejection).
- **Germany-30:** Historical v1's highest-regime label coverage is 1,431/2,000.
  None of its raw learned methods satisfies the joint 1 MW or strict DC
  feasibility check there. Current v2 produced validated numerical labels for
  600/600 sampled test attempts. It has no matching learned checkpoint and
  should not be credited with any historical neural/hybrid results. All
  feasibility claims are relative to modeled constraints.
