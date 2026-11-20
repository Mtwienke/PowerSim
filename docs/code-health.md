# Code health cleanup

Started **2026-11-20**, using desloppify on branch
`desloppify/code-health`. This is maintenance, not another research experiment.

**User stopping rule:** stop cleanup when the tool reports a strict score of
**95 or higher**. This is an execution limit, not a score supplied to blind
reviewers or a reason to change their evidence-based assessments.

## Scope and safety

- Scan the coherent Python project at the repository root.
- Exclude `.venv`, `__pycache__` and generated/frozen `artifacts` from health
  scoring. Do not modify or remove historical datasets, models or results.
- Preserve public CLI commands, optional-ML import boundaries, checkpoint
  formats, loss definitions, RNG streams and historical numerical behavior.
- Do not retune the reconstruction protocol or try to solve its accuracy or
  feasibility gaps as part of maintenance.
- Do not launch another full suite or any German experiment.
- Inspect findings before fixing them. Dynamic array keys, serialized
  provenance and intentional optional imports can produce false positives;
  document evidence rather than deleting useful behavior to satisfy a detector.
- Run tests, replay saved checkpoints, verify old hashes, then commit logical
  cleanup batches.

## Initial assessment

The pre-cleanup revision is `5fd52f2`. Existing ignored desloppify state was
backed up to
`artifacts/desloppify-state-before-cleanup-20261120T002622Z/`.

First mechanical scan: **100 detections**, **92 open tracked issues** after
state reconciliation. Reported scores were **92.2 overall**, **89.8 objective**,
**92.2 strict**, **86.2 verified**. These include inherited subjective scores,
six stale assessments and initially reduced security coverage (Bandit absent).
They are not a fresh independent review baseline.

The tool runs outside the project's scientific environment:

```sh
uvx --with bandit --with pillow --with ruff \
  --from git+https://github.com/peteromallet/desloppify.git@3a7735d531a96b6a226bfbdc9fd662b14195f857 \
  desloppify --lang python scan --path .
```

Tool version: **desloppify 1.0**, revision
`3a7735d531a96b6a226bfbdc9fd662b14195f857`. Its isolated runner uses Python
3.15; the scientific environment remains Python 3.13.5.

Refreshing inherited subjective assessments uses immutable blind review packets
and one dimension per reviewer, in bounded parallel waves. Reviewers receive no
score history or target. The explicit review-rerun override is used to replace
inherited assessments before changing code, not to hide unresolved defects.

Baseline checks:

- **143 tests passed**, with 14 existing pandapower transformer-data deprecation
  warnings.
- **829 reconstruction output hashes**, frozen source files/archive, and
  **135 prior-pilot output hashes** verified.
- Saved **22 model predictions across 66 fixed input batches** for an exact
  post-refactor comparison, under
  `artifacts/desloppify-compatibility-20261120/`.
- Captured all nine commands' help text and tiny deterministic training states,
  loss histories and metrics for **18 profile/method combinations and three
  decoders**. These small compatibility fixtures do not inform research
  parameters or replace any experiment result.
- Independent Bandit baseline: eight low-severity subprocess review warnings,
  no scanner errors.
- All **165** cached evaluation reports and **36** cached attack metadata
  records match their hash-verified stage originals. Strengthening the analyzer
  to consume those originals does not require correcting the completed results.

## Refreshed baseline before production changes

All **20** independent blind batches were imported, producing **30** review
findings. Failed review batches were retried sequentially after a service usage
limit; none were omitted or assigned invented scores. The subsequent scan used
Ruff, Bandit and Pillow from the isolated tool environment.

The canonical post-scan state reports **87.9 overall**, **89.6 objective**,
**87.9 strict**, **85.3 verified**. This replaces inherited subjective
assessments and improves scanner coverage; the reduction from the initial score
does not reflect a production-code regression. The complete state is preserved
at `artifacts/desloppify-compatibility-20261120/fresh-baseline-state.json`.

Every baseline health/strict pair is equal:

| Dimension | Baseline health / strict |
|---|---:|
| Code quality | 81.2 |
| Duplication | 99.5 |
| File health | 82.7 |
| Security | 100.0 |
| Test health | 89.5 |
| Abstraction fit | 91.0 |
| AI-generated debt | 97.0 |
| API coherence | 87.0 |
| Authorization consistency | 100.0 |
| Contract coherence | 91.0 |
| Convention consistency | 93.0 |
| Cross-module architecture | 86.0 |
| Dependency health | 88.0 |
| Design coherence | 89.0 |
| Error consistency | 87.0 |
| High-level elegance | 88.0 |
| Migration completeness | 96.0 |
| Initialization coupling | 93.0 |
| Logic clarity | 94.0 |
| Low-level elegance | 88.0 |
| Mid-level elegance | 84.0 |
| Naming quality | 90.0 |
| Package organization | 84.0 |
| Test strategy | 84.0 |
| Type safety | 78.0 |

## Results

Cleanup, fresh review scores and post-change verification will be recorded here
after the execution queue has been worked through. No score gain is claimed
from the initial scan or from correcting a false-positive classification.
