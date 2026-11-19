# Research protocol and scientific assumptions

Target: Gao et al., *A Trustable Data-Driven Optimal Power Flow Computational
Method With Robust Generalization Ability*, DOI
[10.1109/TNNLS.2024.3437741](https://doi.org/10.1109/TNNLS.2024.3437741).
This implementation is paper-inspired, not a claim of numerical reproduction.
The initial implementation and historical suites used the metadata/abstract.
The full article was supplied on 2026-11-19; the [full-text audit and shared-M1
pilot](paper-protocol.md) records verified methods, training settings, metric
interpretation and remaining ambiguities. Historical results are not rewritten.

Pandapower is pinned to 3.5.6 because the DC map/branch adapter uses its internal
PPC ordering and limits. Upgrade deliberately with the real reference/PF,
transformer and sensitivity regression tests, not by assuming private API
compatibility. Other versions are recorded per artifact.

## Physics and contracts

Default IEEE 30 DC numerical reference uses `rundcopp`. IEEE 14 is selectable.
AC reference uses `runopp`. Independent evaluation uses only `rundcpp`/`runpp`.
The [Germany-30](germany30.md) regional equivalent is a separate DC-only benchmark.
No learned/heuristic prediction is replaced with OPF before reporting its errors.
DTOs are frozen and have stable bus/source IDs. Source order is ascending
external grids, then ascending generators; **all include slack**. Bus order is
ascending in-service bus index. Actual polynomial EUR/h cost coefficients,
including quadratic terms, come from the bundled network tables. Unsupported
piecewise costs/controllable static generators fail explicitly.

Structural SHA256 identity includes topology, switches, transformer parameters,
taps, source bounds and cost tables, but excludes scenario load P/Q and dispatch
P. Scenario identity and configuration identity are separate. Load row placement
is deliberately not structural: data inputs are an ordered nodal vector.

Reference labels require converged OPF, finite complete results and independent
constraint validation. DC bounds/active masks use **0.001 MW** tolerance,
lossless balance **0.0001 MW**. AC voltage tolerance is **0.00001 pu** and loading
tolerance **0.001 percentage point**. Line and transformer limits use configured
`max_loading_percent`, not a hardcoded 100%. DC branch constraints use pandapower
RATE_A in MW; AC uses applicable current/transformer loading. Voltage and Q
metrics are `null` in DC; DC voltage is not an AC prediction.

PF slack is inherently free to balance: its proposed and realized output and
difference are reported. DC feasibility also requires the **raw proposal**
balance, so slack cannot conceal prediction error. AC raw net-injection residual
includes losses and is diagnostic, not a lossless constraint. There is no repair
or projection. AC P-only policies retain network voltage targets unless explicit
voltage targets are supplied.

## Scenario and split definitions

Smoke: 32 training, 8 validation, 8 cases for each of three test regimes.
Paper-inspired: 10,000 training, 1,000 validation (additional assumption),
2,000 tests per regime. Train/validation/IID use 35.58% nameplate penetration;
OOD tests use 40.71% and 47.24%. These are **assumed installed nameplate MW /
nominal gross load MW**, NOT a claim about the paper's unspecified definition.
Actual sampled renewable MW / gross load MW is recorded separately.

At each bus `load = nominal * max(0, Normal(mean=1, std=0.2))`: the quoted
0.2 is interpreted as an independent relative multiplier, giving 20% of each
bus's nominal demand in MW. Zero nominal loads remain zero before renewables.

Two wind and two PV installations are placed at distinct seeded PQ buses.
Nameplate capacity is split equally across technologies, then equally per site.
Wind: independent Weibull scale 5.089 m/s, shape 2.016, converted using a cubic
curve between cut-in 3 and rated 12 m/s, rated output until cut-out 25 m/s, zero
otherwise. Solar: independent Beta(2.06,2.5) times reference 1000 W/m², converted
linearly with assumed performance ratio 0.85 and clipped to nameplate.
All distribution/curve/installation settings are configurable via JSON.
Net load is gross load minus these **fixed injections**, applied once only.

Dataset defaults scale bundled branch limits to 90% to exercise congestion;
this deliberate stress condition is NOT a verified paper setting. Reference
command/dashboard retain unscaled limits. Every manifest reports actual active
generator and branch statistics, failures, seeds, ordering, units and versions.
Never assume that every sample is congested. Failed/infeasible samples remain
in JSONL but are excluded from supervised arrays (no hidden resampling).

Streams are separate for installation, training, validation, IID and OOD.
`dataset.npz` contains raw MW inputs/outputs, flows, objectives, masks, local
sensitivities and training-only normalization statistics. JSONL retains complete
scenarios, solver outputs/failures and conversion quantities.

## Reference sensitivities

For fixed active set, differentiate the strictly convex DC quadratic program's
KKT equations. Include balance, locally active P bounds and branch equalities.
The branch relation is `flow = M_G generation - M_D net_load + constant`;
the map includes transformer taps, phase shifts and shunt constants from
pandapower. Active branch derivatives satisfy `M_G[j] J = M_D[j]`.
Rank-deficient/ill-conditioned KKT systems (condition > 1e10) are unavailable,
never silently pseudoinverted or treated as zero reference sensitivities.
Dataset sensitivities also require positive active inequality multipliers
(strict-complementarity tolerance 1e-5 in the cost/constraint convention);
weakly active inequalities are marked unavailable.
Finite-difference tests check an interior operating point; sensitivities at
active-set transitions are not guaranteed. Tolerance-based masks are local
approximations; inspect conditioning/active-set stability for published runs.

## Neural methods and loss conventions

Optional `.[ml]` installs PyTorch; basic reference/data commands never import it.
`train --method baseline` uses L0 only. `--method physics-guided` enables L0–L5
with equal weights (confirmed in the full paper).
Explicit aliases `M1`, `M2`, `M4`, `M5` select the Table I objectives:
L0, L0+L1, L0–L4 and L0–L5, respectively. M3 is DeepOPF, not a loss ablation.
`--loss-weights '{"L1":0,"L5":0.1}'` overrides any terms independently.
Three 200-neuron hidden layers are the default. Assumptions: tanh activation,
linear output, Adam with learning rate 0.001, batch size 64, 100 epochs,
CPU float32, deterministic algorithms, best validation generation MSE checkpoint.
Architecture, activation, batch size, epochs, seed and all TrainConfig fields
are configurable. Smoke commands use three 16-neuron layers and three epochs.
These generic defaults are retained for compatibility, not claimed to match the
paper's training schedule. `pilot DATASET OUTPUT` explicitly performs 1,000 M1
epochs, then 200 warm-started epochs for M1-continued/M2/M4/M5 at learning rate
0.01 on unchanged data. It reports train/validation metrics, not held-out tests.
`--initialize-from` loads checked compatible weights with fresh Adam state.
Every training saves best-validation, final and selected checkpoints; use
`--checkpoint-selection last` to select the final epoch. New checkpoints record
initializer and train/validation hashes. Shuffle and perturbation streams are
independent, preserving batch order across loss ablations.

Inputs/outputs are centered and scaled **using training statistics only**.
Constant dimensions have scale 1 MW. Public forward evaluation takes raw MW
and returns raw MW. Autograd through internal normalization therefore computes
`J_physical[g,d] = scale_y[g] * J_normalized[g,d] / scale_x[d]`.
Jacobians retain their computation graph so training differentiates through J.
There is no output clipping or projection.

Implemented terms (means over the relevant entries, masks normalize by the
number of selected entries; empty masks contribute zero):

- L0: mean squared predicted/reference generation error, in MW².
- L1: mean squared physical Jacobian error, only for available KKT sensitivities.
- L2: mean square of `sum_g J[g,d] - 1`.
- L3: mean square of J rows for **locally active** reference source bounds.
- L4: mean square of `M_G[j] J - M_D[j]` for **locally active** reference branches.
- L5: L2 at independent Gaussian perturbations of training inputs, default
  standard deviation 0.1 * training input scale per bus, **without labels**.
  These are mathematical nodal perturbations (may include negative net loads),
  not a claim of physically sampled weather. L3/L4 are never imposed there.

L1–L5 have units (MW/MW)². Their weighting relative to L0 is configurable;
equal weights are not scale invariant. ReLU has zero input Hessians almost
everywhere, but mixed input/parameter derivatives can still train Jacobian
penalties; the smooth tanh default is an unverified activation assumption. No test
labels enter training, early stopping, decoder training or calibration.

## Decoder and online decisions

The separate tanh decoder reconstructs **normalized nodal net loads** from
the OPF network's **penultimate hidden layer (`hidden[-2]`, second of three)**.
The layer choice is confirmed by the full paper. `--hidden 128,128` selects the
paper's two-hidden-layer topology (widths remain assumptions); a single width
retains the legacy one-hidden-layer decoder. The paper uses 1,000 decoder epochs
and learning rate 0.001; generic smoke-compatible defaults remain 100 epochs.
The OPF network is frozen; decoder Adam training
uses training inputs only, selecting decoder weights on validation reconstruction
error. Threshold is the `higher` empirical quantile (default 95%) of
**validation-only** per-scenario reconstruction errors.

Aggregation is the mean of squared reconstruction errors across **all ordered
buses**. Default units are dimensionless normalized-input squared units.
`decoder --error-units mw2` instead rescales each residual to MW before averaging,
yielding MW²; artifact metadata records the choice and threshold. Decoder files
are cryptographically bound to the OPF checkpoint.
`mae_mw` and `rmse_mw` expose physical MAE/RMSE in MW. `--threshold` explicitly
sets a fixed threshold in the selected score units, recorded separately from
quantile calibration. The paper's reported 6/7 MW thresholds cannot be copied
into normalized MSE without an assumption about its undisclosed error norm.
`powersim calibrate DATASET MODEL DECODER OUTPUT --quantile 0.95` is an explicit
threshold-only recalibration command (also restricted to validation inputs).

Accepted -> untouched learned output. Rejected -> fresh numerical DC-OPF for
the identical net-load vector and network configuration. Failed/invalid fallback
-> explicit failure with no substituted output. Acceptance does not check or
certify constraints; independent audits remain necessary. The production `infer`
command appends rejected inputs and diagnostics to a separate caller-specified
log. `evaluate` never appends held-out inputs to that log. Retraining is explicit;
never merge held-out evaluation records into a training dataset.

## Metrics and timing

`evaluate` accepts only held-out splits and rejects training/validation ID overlap.
Accuracy includes **all conventional sources including slack**:
MAE/RMSE over finite source errors, maximum absolute source error, and success
rate = fraction of reference-valid scenarios with complete output and
**maximum error across every source <= 1 MW**. Failed hybrid outputs count as
unsuccessful, not as zero error. Accuracy sample count is reported; reference
failures are excluded from error metrics and reported separately. A second
conservative success rate includes all originally requested scenarios in its
denominator. Accepted-subset and complete-hybrid metrics are both reported.
New reports additionally expose strict individual-source and joint-scenario
`< 1 MW` accuracy, their counts/denominators, accepted-only physics, fallback-only
accuracy and explicit neural coverage. Generator-wise aggregation is strongly
suggested by Table V, but Eq. (26) does not define vector reduction precisely.
The legacy `success_rate_1mw` remains inclusive `<= 1 MW`. Empty selected subsets
have null conditional accuracy, not perfect accuracy. Accuracy/physical-risk
curves use validation-derived/explicit thresholds without test-time selection.

Independent raw/hybrid PF audits report absolute raw balance residuals,
generator/branch violation counts and feasibility. Objective gap is
`100 * (predicted_cost - reference_cost) / abs(reference_cost)` **only for
feasible predictions**, with coverage count. If none are feasible, gap is null,
not zero. No AC voltage or Q error is inferred from a DC model.

Single-scenario CPU decision latency includes normalization, decoder computation,
and fallback OPF on a prepared worker network when invoked. Production logging overhead
is also included when logging is requested. Model loading and one explicit
warm-up are excluded. Independent PF audit time is excluded from decision latency
but included in separately measured evaluation-loop wall time. Batch throughput
is separately measured for the **OPF neural network only**, not mislabeled as
hybrid latency. All runtimes are measured wall clock, not derived from FLOPs.
For [parallel evaluations](performance.md), concurrently measured worker
durations are exposed separately, without claiming isolated latency.
Calls request `numba=False`. Pandapower 3.5.6's `rundcpp` ignores that keyword
and auto-detects Numba; the validated environment has no Numba and uses the
no-JIT CPU path. Its repeated missing-Numba warnings are upstream behavior.
Numba's installed version (or absence) is recorded; do not compare these timings
with an unspecified accelerated solver.

## Configuration and artifact layout

`dataset --config FILE` accepts DatasetConfig JSON overrides. Nested
`simulation` and `renewable` objects use their dataclass field names.
`train --config FILE` accepts TrainConfig overrides. Examples:

```json
{
  "load_std_fraction": 0.2,
  "load_mean_factor": 1.0,
  "penetration_rates": [0.3558, 0.4071, 0.4724],
  "renewable": {
    "wind_scale": 5.089, "wind_shape": 2.016,
    "solar_alpha": 2.06, "solar_beta": 2.5,
    "cut_in_ms": 3.0, "rated_ms": 12.0, "cut_out_ms": 25.0,
    "irradiance_reference_wm2": 1000, "pv_performance_ratio": 0.85
  }
}
```

- Dataset: `network.json` (source pandapower network), `manifest.json`, split
  JSONL (including failures), `dataset.npz`.
- OPF model: selected `model.pt`, `best.pt`, `last.pt`, `training.json`
  (loss/validation history, fitting accuracy and initialization provenance).
- Decoder: `decoder.pt`, `decoder.json` (frozen model binding, calibration IDs).
- Evaluation: `evaluation.json`, `predictions.jsonl`, `predictions.npz`.
- Production rejections: separate JSONL selected with `--production-log`.

Output directories must not already exist. Failed generation leaves diagnostic
records rather than overwriting or silently deleting prior work.
Check manifest active-set fractions and failure rates before investing in a
large run. A numerical failure is recorded as solver failure, not automatically
proved mathematical infeasibility.

## Remaining differences from the publication

No claim of matching published accuracy/runtime. The full text now confirms
method objectives and a two-stage training schedule, but original code,
seed/placement/cost/limit settings and numerical representation remain unavailable.
Conversion curves, nameplate interpretation of penetration, IID/OOD design,
90% thermal stress, independent load/weather draws, decoder widths/norm/threshold
calibration, activation, batch size and checkpoint selection remain configurable
assumptions. The German benchmark is a separate transfer experiment. KKT
derivatives cover regular DC QPs only, not active-set
transitions, nonconvex AC OPF, or unsupported network devices. There is no
retraining service, feasibility repair or robustness certificate.
