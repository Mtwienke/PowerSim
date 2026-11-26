"""Seeded headless scenarios, validated labels and leakage-safe dataset artifacts."""
from dataclasses import asdict, dataclass, field
import gzip
from importlib.metadata import PackageNotFoundError, version
import json
from itertools import islice
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandapower as pp

from ..core.config import SimulationConfig
from ..core.dc import dc_relation, reference_jacobian
from ..core.problem import OPFInstance
from ..core.solve import solve_reference
from ..core.workers import ordered_pool, validate_workers
from .network import (NodalLoadWriter, apply_nodal_loads, build_dispatch_inputs, digest, make_case,
                      network_fingerprint, nodal_power)


@dataclass(frozen=True)
class RenewableConfig:
    wind_scale: float = 5.089
    wind_shape: float = 2.016
    solar_alpha: float = 2.06
    solar_beta: float = 2.5
    cut_in_ms: float = 3.0
    rated_ms: float = 12.0
    cut_out_ms: float = 25.0
    irradiance_reference_wm2: float = 1000.0
    pv_performance_ratio: float = .85
    installation_pairs: int = 2
    wind_capacity_fraction: float = .5

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Renewable parameters must be finite")
        if not 0 <= self.cut_in_ms < self.rated_ms < self.cut_out_ms:
            raise ValueError("Wind speeds must be ordered")
        if min(self.wind_scale, self.wind_shape, self.solar_alpha, self.solar_beta,
               self.irradiance_reference_wm2) <= 0:
            raise ValueError("Distribution/reference parameters must be positive")
        if not 0 < self.pv_performance_ratio <= 1 or not 0 <= self.wind_capacity_fraction <= 1:
            raise ValueError("Invalid renewable capacity/performance fraction")
        if not isinstance(self.installation_pairs, int) or self.installation_pairs < 1:
            raise ValueError("installation_pairs must be positive")


@dataclass(frozen=True)
class DatasetConfig:
    simulation: SimulationConfig = field(default_factory=lambda: SimulationConfig(branch_limit_scale=.9))
    renewable: RenewableConfig = field(default_factory=RenewableConfig)
    seed: int = 2026
    train_cases: int = 32
    validation_cases: int = 8
    test_cases: int = 8
    load_std_fraction: float = .2
    load_mean_factor: float = 1.0
    penetration_rates: tuple[float, ...] = (.3558, .4071, .4724)
    penetration_basis: str = "nominal_demand"
    renewable_capacities_mw: tuple[float, ...] = ()
    sampling_policy: str = "attempts"
    max_attempt_factor: int = 10
    placement_seed: int | None = None
    test_seed: int | None = None
    demand_shift_mean_factor: float | None = None
    preserve_fixed_injections: bool = False
    training_sensitivities_only: bool = False
    record_format: str = "jsonl"

    def __post_init__(self):
        if self.simulation.physics_mode != "dc":
            raise ValueError("Paper dataset generation is DC only")
        if not all(isinstance(v, int) and v > 0 for v in (self.train_cases, self.validation_cases, self.test_cases)):
            raise ValueError("Every split must be nonempty")
        if not np.isfinite([self.load_std_fraction, self.load_mean_factor]).all() or self.load_std_fraction < 0 or self.load_mean_factor <= 0:
            raise ValueError("Invalid load distribution")
        if not self.penetration_rates or any(not np.isfinite(r) or not 0 <= r <= 1 for r in self.penetration_rates):
            raise ValueError("Invalid nameplate penetration rates")
        if self.penetration_basis not in ("nominal_demand", "total_installed"):
            raise ValueError("Unknown penetration basis")
        if self.penetration_basis == "total_installed" and any(r >= 1 for r in self.penetration_rates):
            raise ValueError("Total-installed penetration must be below 1")
        if self.renewable_capacities_mw and (len(self.renewable_capacities_mw) != len(self.penetration_rates)
                or not np.isfinite(self.renewable_capacities_mw).all() or min(self.renewable_capacities_mw) < 0):
            raise ValueError("One finite nonnegative renewable capacity per regime required")
        if self.sampling_policy not in ("attempts", "valid"):
            raise ValueError("sampling_policy must be attempts or valid")
        if isinstance(self.max_attempt_factor, bool) or not isinstance(self.max_attempt_factor, int) or self.max_attempt_factor < 1:
            raise ValueError("max_attempt_factor must be a positive integer")
        if self.demand_shift_mean_factor is not None and (
            not np.isfinite(self.demand_shift_mean_factor) or self.demand_shift_mean_factor <= 0
            or self.demand_shift_mean_factor == self.load_mean_factor
        ):
            raise ValueError("Demand OOD mean must be positive and differ from the training mean")
        if self.record_format not in ("jsonl", "jsonl.gz"):
            raise ValueError("record_format must be jsonl or jsonl.gz")
        for seed in (self.seed, self.placement_seed, self.test_seed):
            if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
                raise ValueError("Seeds must be nonnegative integers")

    @classmethod
    def preset(cls, name="smoke", **kwargs):
        if name not in ("smoke", "paper"):
            raise ValueError("Unknown dataset preset")
        values = {} if name == "smoke" else {"train_cases": 10000, "validation_cases": 1000, "test_cases": 2000}
        return cls(**{**values, **kwargs})


def wind_capacity_factor(speed, config: RenewableConfig):
    speed = np.asarray(speed)
    fraction = (speed**3-config.cut_in_ms**3) / (config.rated_ms**3-config.cut_in_ms**3)
    return np.where((speed >= config.cut_in_ms) & (speed < config.cut_out_ms),
                    np.clip(fraction, 0, 1), 0.0)


def solar_capacity_factor(irradiance_wm2, config: RenewableConfig):
    return np.clip(np.asarray(irradiance_wm2) / config.irradiance_reference_wm2 *
                   config.pv_performance_ratio, 0, 1)


def package_versions():
    result = {"python": platform.python_version()}
    for name in ("powersim-research", "pandapower", "numpy", "scipy", "pandas", "torch", "numba"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _init_reference_worker(config):
    global _reference_worker
    _reference_worker = NodalLoadWriter(make_case(config)), config


def _solve_scenario(record):
    writer, config = _reference_worker
    writer.apply(record["net_load_mw"])
    return record, solve_reference(writer.net, config)


def installed_capacity(config, rate, nominal_demand_mw, conventional_pmax_mw, explicit=None):
    """Explicitly chosen capacity hypothesis; never infer it from model accuracy."""
    if config.penetration_basis == "total_installed":
        inferred = conventional_pmax_mw*rate/(1-rate)
    else:
        inferred = nominal_demand_mw*rate
    capacity = float(inferred if explicit is None else explicit)
    denominator = conventional_pmax_mw+capacity if config.penetration_basis == "total_installed" else nominal_demand_mw
    actual_rate = capacity/denominator if denominator else 0.
    if explicit is not None and abs(actual_rate-rate) > .00005+1e-12:
        raise ValueError("Explicit nameplate does not match the printed regime to two percentage decimals")
    return capacity, actual_rate


def _scenarios(config, split, count, rate, stream, nominal, sites, capacity_mw=None, fixed_mw=0.):
    """Generate in the parent, so process count cannot change any RNG stream."""
    rng = np.random.default_rng(stream)
    renewable_config = config.renewable
    nwind = renewable_config.installation_pairs
    capacity = float(rate * nominal.sum()) if capacity_mw is None else float(capacity_mw)
    load_mean = config.demand_shift_mean_factor if split == "ood_demand" else config.load_mean_factor
    for number in range(count):
        gross = nominal * np.maximum(0, rng.normal(load_mean, config.load_std_fraction, len(nominal)))
        wind = renewable_config.wind_scale * rng.weibull(renewable_config.wind_shape, nwind)
        irradiance = rng.beta(renewable_config.solar_alpha, renewable_config.solar_beta, nwind) * renewable_config.irradiance_reference_wm2
        renewable = np.concatenate([
            wind_capacity_factor(wind, renewable_config) * capacity * renewable_config.wind_capacity_fraction/nwind,
            solar_capacity_factor(irradiance, renewable_config) * capacity * (1-renewable_config.wind_capacity_fraction)/nwind])
        loads = gross - fixed_mw
        loads[sites] -= renewable
        net_load = loads.tolist()
        yield {"scenario_id": digest({"split": split, "number": number, "seed": config.seed, "x": net_load}),
               "split": split, "number": number, "net_load_mw": net_load,
               "input_id": digest(net_load), "gross_load_mw": gross.tolist(),
               "wind_speed_ms": wind.tolist(), "irradiance_wm2": irradiance.tolist(),
               "renewable_mw": renewable.tolist(), "installed_capacity_mw": capacity,
               "nameplate_penetration": rate,
               "realized_penetration": float(renewable.sum()/gross.sum()) if gross.sum() else None}


def _valid_results(scenarios, map_cases, target, cap, gens, relation):
    """Bounded rounds cannot overshoot the successful-label target.

    Every round contains at most the number of valid labels still needed, so
    even process prefetch never creates unrecorded surplus attempts.
    """
    source = iter(scenarios)
    accepted = attempted = 0
    while accepted < target and attempted < cap:
        block = list(islice(source, min(128, target-accepted, cap-attempted)))
        if not block:
            break
        for record, result in map_cases(_solve_scenario, block):
            attempted += 1
            accepted += int(valid_label(result, gens, relation.branch_ids))
            yield record, result
    if accepted < target:
        raise ValueError(f"Valid-label attempt cap exhausted: {accepted}/{target} valid in {attempted}/{cap} attempts")


def scenario_design(config, net, inputs):
    """Freeze the same case/split design for pilot feasibility and full labels."""
    nominal = np.asarray(nodal_power(net, "load"))
    fixed_mw = np.asarray(nodal_power(net, "sgen")) if config.preserve_fixed_injections else 0.
    buses = [int(b.split(":")[1]) for b in inputs["context"].bus_ids]
    pq = [i for i, b in enumerate(buses) if b not in set(net.gen.bus) | set(net.ext_grid.bus)]
    if len(pq) < 2 * config.renewable.installation_pairs:
        raise ValueError("Not enough PQ buses for requested installations")
    seeds = np.random.SeedSequence(config.seed if config.placement_seed is None else config.placement_seed).spawn(2 + len(config.penetration_rates))
    install_rng = np.random.default_rng(seeds[0])
    sites = install_rng.choice(pq, 2 * config.renewable.installation_pairs, replace=False)
    nwind = config.renewable.installation_pairs
    splits = [("train", config.train_cases, config.penetration_rates[0]),
              ("validation", config.validation_cases, config.penetration_rates[0])]
    splits += [("iid_test" if i == 0 else f"ood_test_{i}", config.test_cases, rate)
               for i, rate in enumerate(config.penetration_rates)]
    regimes = [0, 0, *range(len(config.penetration_rates))]
    if config.demand_shift_mean_factor is not None:
        splits.append(("ood_demand", config.test_cases, config.penetration_rates[0]))
        regimes.append(0)
    # Installation seed independent of ALL split RNG streams.
    split_seeds = np.random.SeedSequence([config.seed, 917]).spawn(len(splits))
    if config.test_seed is not None:
        split_seeds[2:] = np.random.SeedSequence([config.test_seed, 917]).spawn(len(splits)-2)
    pmax = sum(g.p_max_mw for g in inputs["generators"])
    capacities = [installed_capacity(config, rate, float(nominal.sum()), pmax,
                     config.renewable_capacities_mw[i] if config.renewable_capacities_mw else None)
                  for i, rate in enumerate(config.penetration_rates)]
    return nominal, fixed_mw, sites, capacities, [(split, stream, capacities[i])
                                                  for split, stream, i in zip(splits, split_seeds, regimes)]


def generate_dataset(directory, config: DatasetConfig | None = None, *, workers=1):
    config = config or DatasetConfig()
    validate_workers(workers)
    started = perf_counter()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    net = make_case(config.simulation)
    pp.to_json(net, str(directory / "network.json"))
    inputs = build_dispatch_inputs(net, config.simulation)
    nominal, fixed_mw, sites, capacities, design = scenario_design(config, net, inputs)
    relation = dc_relation(net)
    gens = inputs["generators"]
    buses = [int(b.split(":")[1]) for b in inputs["context"].bus_ids]
    nwind = config.renewable.installation_pairs
    manifest = {
        "schema_version": "3.0", "configuration": asdict(config), "versions": package_versions(),
        "model_id": inputs["context"].network_fingerprint,
        "configuration_id": inputs["context"].configuration_id,
        "source_provenance": net.get("source_provenance"),
        "nominal_gross_load_mw": nominal.tolist(),
        "retained_benchmark_fixed_injection_mw": np.broadcast_to(fixed_mw, nominal.shape).tolist(),
        "record_format": config.record_format,
        "bus_ids": list(inputs["context"].bus_ids), "generator_ids": [g.asset_id for g in gens],
        "generators": [asdict(g) for g in gens], "branch_ids": relation.branch_ids,
        "installations": [{"bus_id": f"bus:{buses[int(site)]}", "kind": "wind" if i < nwind else "pv",
                           "capacity_fraction": (config.renewable.wind_capacity_fraction if i < nwind else
                                                 1-config.renewable.wind_capacity_fraction)/nwind}
                          for i, site in enumerate(sites)],
        "units": {"x": "MW net load per bus", "y": "MW conventional generation incl. ext_grid",
                  "objective": "EUR/h", "branch_flows": "MW", "jacobian": "MW/MW",
                  "wind_speed": "m/s", "irradiance": "W/m^2"},
        "penetration_definition": ("total installed renewable nameplate MW / nominal gross demand MW"
                                  if config.penetration_basis == "nominal_demand" else
                                  "renewable nameplate MW / (conventional Pmax MW + renewable nameplate MW); hypothesis"),
        "capacity_regimes": [{"printed_rate": rate, "renewable_capacity_mw": capacity,
                             "actual_nameplate_fraction": actual}
                            for rate, (capacity, actual) in zip(config.penetration_rates, capacities)],
        "realized_penetration_definition": "sample renewable MW / sample gross demand MW",
        "assumptions": ["Independent per-bus load multiplier max(0, Normal(mean_factor, 0.2)); "
                        "0.2 is relative to each bus nominal MW, not 0.2 MW.",
                        "Independent wind/PV outputs per site. Cubic wind curve, linear clipped PV.",
                        "Rates are assumed nameplate fractions, not verified paper electrical-output rates.",
                        "Load row placement is not structural; renewables enter only as negative nodal load."],
        "splits": {}, "normalization_fit_split": "train",
        "sampling_policy": config.sampling_policy,
        "sampling_note": "Attempt failures remain in JSONL. Valid-sample collection is conditional on successful reference solves, not a proof of infeasibility.",
    }
    arrays = {}
    with ordered_pool(workers, _init_reference_worker, (config.simulation,)) as map_cases:
        for (split, count, rate), stream, (capacity, actual) in design:
            cap = count if config.sampling_policy == "attempts" else count*config.max_attempt_factor
            scenarios = _scenarios(config, split, cap, rate, stream, nominal, sites, capacity, fixed_mw)
            results = (map_cases(_solve_scenario, scenarios) if config.sampling_policy == "attempts" else
                       _valid_results(scenarios, map_cases, count, cap, gens, relation))
            try:
                stats = _write_split(directory, split, count, rate, stream, results,
                                      arrays, nominal, gens, relation,
                                      compute_sensitivities=(split == "train" or not config.training_sensitivities_only),
                                      record_format=config.record_format)
            except ValueError as exc:
                write_json(directory / "generation_error.json", {"split": split, "diagnostic": str(exc),
                           "sampling_policy": config.sampling_policy, "attempt_cap": cap})
                raise
            stats.update(target_valid_cases=count if config.sampling_policy == "valid" else None,
                         attempt_cap=cap, renewable_capacity_mw=capacity, actual_nameplate_fraction=actual,
                          seed_entropy=stream.entropy, load_mean_factor=(
                              config.demand_shift_mean_factor if split == "ood_demand" else config.load_mean_factor))
            manifest["splits"][split] = stats
    for name, values in (("x", arrays["train_x"]), ("y", arrays["train_y"])):
        arrays[f"{name}_mean"] = values.mean(axis=0)
        std = values.std(axis=0)
        arrays[f"{name}_scale"] = np.where(std > 1e-6, std, 1.0)
    arrays.update(m_g=relation.m_g, m_d=relation.m_d, branch_constant=relation.constant)
    from pandapower.pypower.idx_brch import RATE_A
    arrays["branch_limits_mw"] = net._ppc["branch"][:, RATE_A].real.copy()
    np.savez_compressed(directory / "dataset.npz", **arrays)
    elapsed = perf_counter() - started
    manifest["execution"] = {"workers": workers, "wall_seconds": elapsed,
                             "attempts_per_second": sum(s["requested"] for s in manifest["splits"].values()) / elapsed}
    write_json(directory / "manifest.json", manifest)
    return manifest


def split_record_file(directory, split, manifest):
    return Path(directory) / f"{split}.{manifest.get('record_format', 'jsonl')}"


def load_dataset_attempt(directory, split, sample):
    """Read a saved network and one original attempt, including failed references."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if split not in manifest["splits"] or sample < 0:
        raise ValueError("Unknown split or negative sample number")
    path = split_record_file(directory, split, manifest)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as records:
        for line in records:
            record = json.loads(line)
            if record["number"] == sample:
                break
        else:
            raise ValueError(f"Sample {sample} not found in {split}")
    net = pp.from_json(str(directory / "network.json"))
    if network_fingerprint(net) != manifest["model_id"]:
        raise ValueError("Saved source network does not match dataset identity")
    return net, manifest, record


def load_dataset_instance(directory, split="iid_test", sample=0) -> OPFInstance:
    """Recreate the exact nodal inputs presented to the original solver."""
    net, manifest, record = load_dataset_attempt(directory, split, sample)
    config = SimulationConfig(**manifest["configuration"]["simulation"])
    apply_nodal_loads(net, record["net_load_mw"])
    return OPFInstance(net, config)


def _write_split(directory, split, count, rate, stream, results, arrays, nominal, gens, relation, *,
                 compute_sensitivities=True, record_format="jsonl"):
    x, y, lower, upper, active, flows, objectives, jacobians, jvalid, sample_ids = ([] for _ in range(10))
    failures = []
    attempted = 0
    path = directory / f"{split}.{record_format}"
    stream_out = gzip.open(path, "wt", compresslevel=3) if record_format == "jsonl.gz" else path.open("w")
    with stream_out as records:
        for record, result in results:
            attempted += 1
            record["reference"] = result.to_dict()
            loads, scenario_id = record["net_load_mw"], record["scenario_id"]
            if valid_label(result, gens, relation.branch_ids):
                j, condition = None, float("nan")
                if compute_sensitivities:
                    j, condition = reference_jacobian([g.cost_quadratic for g in gens], relation,
                        result.active_lower, result.active_upper, [b.active for b in result.branches],
                        generation=np.asarray([result.generation_mw[g.asset_id] for g in gens]),
                        cost_linear=np.asarray([g.marginal_cost_per_mwh for g in gens]),
                        branch_flows=[b.p_from_mw for b in result.branches])
                record["sensitivity"] = {"available": j is not None,
                                         "kkt_condition": condition if np.isfinite(condition) else None,
                                         "unavailable_reason": None if j is not None else
                                         ("Singular/ill-conditioned KKT or weakly active inequality"
                                          if compute_sensitivities else "Not requested for this split")}
                x.append(loads); y.append([result.generation_mw[g.asset_id] for g in gens])
                lower.append(result.active_lower); upper.append(result.active_upper)
                active.append([b.active for b in result.branches])
                flows.append([b.p_from_mw for b in result.branches]); objectives.append(result.objective)
                if compute_sensitivities:
                    jacobians.append(j if j is not None else np.zeros((len(gens), len(loads))))
                jvalid.append(j is not None); sample_ids.append(scenario_id)
            else:
                failures.append({"scenario_id": scenario_id, "status": result.status,
                                 "diagnostics": result.diagnostics})
            records.write(json.dumps(record, allow_nan=False) + "\n")
    if not x and split == "train":
        write_json(directory / "failed_generation.json", {"split": split, "failures": failures})
        raise ValueError("No valid training labels; inspect persisted failure records")
    shape = {"x": (0, len(nominal)), "y": (0, len(gens)),
             "lower": (0, len(gens)), "upper": (0, len(gens)),
             "active_branch": (0, len(relation.branch_ids)),
             "flows": (0, len(relation.branch_ids)),
             "jacobian": (0, len(gens), len(nominal))}
    for key, value in {"x": x, "y": y, "lower": lower, "upper": upper, "active_branch": active,
                       "flows": flows, "objective": objectives, "jacobian": jacobians,
                       "jacobian_valid": jvalid, "scenario_ids": sample_ids}.items():
        arrays[f"{split}_{key}"] = np.asarray(value) if len(value) else np.empty(shape.get(key, (0,)))
    return {"requested": attempted, "valid": len(x), "failures": failures,
            "seed_spawn_key": stream.spawn_key, "nameplate_penetration": rate,
            "active_generator_fraction": float(np.mean(np.asarray(lower) | np.asarray(upper))) if x else None,
            "active_branch_fraction": float(np.mean(active)) if x else None,
            "samples_with_active_branch": int(np.any(active, axis=1).sum()) if x else 0,
            "sensitivity_available": int(sum(jvalid)), "sensitivity_computed": compute_sensitivities}


def valid_label(result, gens, branch_ids):
    """Never turn a failed, incomplete or non-finite solve into supervised data."""
    return (result.status == "ok" and result.converged and result.metrics.get("feasible") is True
            and set(result.generation_mw) == {g.asset_id for g in gens}
            and tuple(b.asset_id for b in result.branches) == tuple(branch_ids)
            and len(result.active_lower) == len(gens) and len(result.active_upper) == len(gens)
            and all(np.isfinite(list(result.generation_mw.values())))
            and all(np.isfinite([b.p_from_mw, b.p_to_mw, b.loading_percent]).all() for b in result.branches)
            and result.objective is not None and np.isfinite(result.objective))


def load_dataset(directory, *, sensitivities=True):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    with np.load(directory / "dataset.npz", allow_pickle=False) as values:
        arrays = {k: values[k] for k in values.files if sensitivities or not k.endswith(("_jacobian", "_jacobian_valid"))}
    return manifest, arrays
