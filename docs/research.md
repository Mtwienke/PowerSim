# Research protocol and scientific assumptions

Target: Gao et al., *A Trustable Data-Driven Optimal Power Flow Computational
Method With Robust Generalization Ability*, DOI
[10.1109/TNNLS.2024.3437741](https://doi.org/10.1109/TNNLS.2024.3437741).
This implementation is paper-inspired, not a claim of numerical reproduction.
The DOI metadata/abstract were accessible during implementation, not the full
Section III or experiment tables. Exact M1–M5 combinations are unverified;
descriptive configurable ablations are used instead.

## Physics and contracts

Default IEEE 30 DC numerical reference uses `rundcopp`. IEEE 14 is selectable.
AC reference uses `runopp`. Independent evaluation uses only `rundcpp`/`runpp`.
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
Finite-difference tests check an interior operating point; sensitivities at
active-set transitions are not guaranteed. Tolerance-based masks are local
approximations; inspect conditioning/active-set stability for published runs.
