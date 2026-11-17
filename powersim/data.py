"""Seeded headless scenarios, validated labels and leakage-safe dataset artifacts."""
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

import numpy as np

from .config import SimulationConfig
from .dc import dc_relation, reference_jacobian
from .network import apply_nodal_loads, build_dispatch_inputs, digest, make_case, nodal_loads
from .solver import solve_reference


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
        if not 0 <= self.cut_in_ms < self.rated_ms < self.cut_out_ms:
            raise ValueError("Wind speeds must be ordered")
        if min(self.wind_scale, self.wind_shape, self.solar_alpha, self.solar_beta,
               self.irradiance_reference_wm2) <= 0:
            raise ValueError("Distribution/reference parameters must be positive")
        if not 0 < self.pv_performance_ratio <= 1 or not 0 <= self.wind_capacity_fraction <= 1:
            raise ValueError("Invalid renewable capacity/performance fraction")
        if self.installation_pairs < 1:
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

    def __post_init__(self):
        if self.simulation.physics_mode != "dc":
            raise ValueError("Paper dataset generation is DC only")
        if min(self.train_cases, self.validation_cases, self.test_cases) < 1:
            raise ValueError("Every split must be nonempty")
        if self.load_std_fraction < 0 or self.load_mean_factor <= 0:
            raise ValueError("Invalid load distribution")
        if not self.penetration_rates or any(not 0 <= r <= 1 for r in self.penetration_rates):
            raise ValueError("Invalid nameplate penetration rates")

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
    for name in ("powersim-research", "pandapower", "numpy", "scipy", "pandas", "torch"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def generate_dataset(directory, config: DatasetConfig | None = None):
    config = config or DatasetConfig()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    net = make_case(config.simulation)
    inputs = build_dispatch_inputs(net, config.simulation)
    nominal = np.asarray(nodal_loads(net))
    relation = dc_relation(net)
    gens = inputs["generators"]
    buses = [int(b.split(":")[1]) for b in inputs["context"].bus_ids]
    pq = [i for i, b in enumerate(buses) if b not in set(net.gen.bus) | set(net.ext_grid.bus)]
    if len(pq) < 2 * config.renewable.installation_pairs:
        raise ValueError("Not enough PQ buses for requested installations")
    seeds = np.random.SeedSequence(config.seed).spawn(2 + len(config.penetration_rates))
    install_rng = np.random.default_rng(seeds[0])
    sites = install_rng.choice(pq, 2 * config.renewable.installation_pairs, replace=False)
    nwind = config.renewable.installation_pairs
    splits = [("train", config.train_cases, config.penetration_rates[0]),
              ("validation", config.validation_cases, config.penetration_rates[0])]
    splits += [("iid_test" if i == 0 else f"ood_test_{i}", config.test_cases, rate)
               for i, rate in enumerate(config.penetration_rates)]
    # Installation seed independent of ALL split RNG streams.
    split_seeds = np.random.SeedSequence([config.seed, 917]).spawn(len(splits))
    manifest = {
        "schema_version": "2.0", "configuration": asdict(config), "versions": package_versions(),
        "model_id": inputs["context"].network_fingerprint,
        "configuration_id": inputs["context"].configuration_id,
        "bus_ids": list(inputs["context"].bus_ids), "generator_ids": [g.asset_id for g in gens],
        "generators": [asdict(g) for g in gens], "branch_ids": relation.branch_ids,
        "installations": [{"bus_id": f"bus:{buses[int(site)]}", "kind": "wind" if i < nwind else "pv",
                           "capacity_fraction": (config.renewable.wind_capacity_fraction if i < nwind else
                                                 1-config.renewable.wind_capacity_fraction)/nwind}
                          for i, site in enumerate(sites)],
        "units": {"x": "MW net load per bus", "y": "MW conventional generation incl. ext_grid",
                  "objective": "EUR/h", "branch_flows": "MW", "jacobian": "MW/MW"},
        "penetration_definition": "total installed renewable nameplate MW / nominal gross demand MW",
        "realized_penetration_definition": "sample renewable MW / sample gross demand MW",
        "assumptions": ["Independent per-bus load multiplier max(0, Normal(mean_factor, 0.2)); "
                        "0.2 is relative to each bus nominal MW, not 0.2 MW.",
                        "Independent wind/PV outputs per site. Cubic wind curve, linear clipped PV.",
                        "Rates are assumed nameplate fractions, not verified paper electrical-output rates.",
                        "Load row placement is not structural; renewables enter only as negative nodal load."],
        "splits": {}, "normalization_fit_split": "train",
    }
    arrays = {}
    train_x = train_y = None
    for split, count, rate in splits:
        stream = split_seeds[len(manifest["splits"])]
        rng = np.random.default_rng(stream)
        x, y, lower, upper, active, flows, objectives, jacobians, jvalid, sample_ids = ([] for _ in range(10))
        failures = []
        with (directory / f"{split}.jsonl").open("w") as records:
            for number in range(count):
                gross = nominal * np.maximum(0, rng.normal(config.load_mean_factor, config.load_std_fraction, len(nominal)))
                wind = config.renewable.wind_scale * rng.weibull(config.renewable.wind_shape, nwind)
                irradiance = rng.beta(config.renewable.solar_alpha, config.renewable.solar_beta, nwind) * config.renewable.irradiance_reference_wm2
                capacity = rate * nominal.sum()
                renewable = np.concatenate([
                    wind_capacity_factor(wind, config.renewable) * capacity * config.renewable.wind_capacity_fraction/nwind,
                    solar_capacity_factor(irradiance, config.renewable) * capacity * (1-config.renewable.wind_capacity_fraction)/nwind])
                loads = gross.copy()
                loads[sites] -= renewable
                scenario_id = digest({"split": split, "number": number, "seed": config.seed, "x": loads.tolist()})
                apply_nodal_loads(net, loads)
                result = solve_reference(net, config.simulation)
                record = {"scenario_id": scenario_id, "split": split, "number": number, "net_load_mw": loads.tolist(),
                          "gross_load_mw": gross.tolist(), "wind_speed_ms": wind.tolist(),
                          "irradiance_wm2": irradiance.tolist(), "renewable_mw": renewable.tolist(),
                          "installed_capacity_mw": float(capacity), "nameplate_penetration": rate,
                          "realized_penetration": float(renewable.sum()/gross.sum()) if gross.sum() else None,
                          "reference": result.to_dict()}
                if valid_label(result, gens, relation.branch_ids):
                    j, condition = reference_jacobian([g.cost_quadratic for g in gens], relation,
                        result.active_lower, result.active_upper, [b.active for b in result.branches])
                    record["sensitivity"] = {"available": j is not None,
                                              "kkt_condition": condition if np.isfinite(condition) else None}
                    x.append(loads); y.append([result.generation_mw[g.asset_id] for g in gens])
                    lower.append(result.active_lower); upper.append(result.active_upper)
                    active.append([b.active for b in result.branches])
                    flows.append([b.p_from_mw for b in result.branches]); objectives.append(result.objective)
                    jacobians.append(j if j is not None else np.zeros((len(gens), len(loads))))
                    jvalid.append(j is not None); sample_ids.append(scenario_id)
                else:
                    failures.append({"scenario_id": scenario_id, "status": result.status,
                                     "diagnostics": result.diagnostics})
                records.write(json.dumps(record, allow_nan=False) + "\n")
        if not x and split == "train":
            write_json(directory / "failed_generation.json", {"split": split, "failures": failures})
            raise ValueError("No valid training labels; inspect persisted failure records")
        for key, value in {"x": x, "y": y, "lower": lower, "upper": upper, "active_branch": active,
                           "flows": flows, "objective": objectives, "jacobian": jacobians,
                           "jacobian_valid": jvalid, "scenario_ids": sample_ids}.items():
            shape = {"x": (0, len(nominal)), "y": (0, len(gens)),
                     "lower": (0, len(gens)), "upper": (0, len(gens)),
                     "active_branch": (0, len(relation.branch_ids)),
                     "flows": (0, len(relation.branch_ids)),
                     "jacobian": (0, len(gens), len(nominal))}
            arrays[f"{split}_{key}"] = np.asarray(value) if len(value) else np.empty(shape.get(key, (0,)))
        manifest["splits"][split] = {"requested": count, "valid": len(x), "failures": failures,
                                    "seed_spawn_key": stream.spawn_key,
                                    "nameplate_penetration": rate,
                                    "active_generator_fraction": float(np.mean(np.asarray(lower) | np.asarray(upper))) if x else None,
                                    "active_branch_fraction": float(np.mean(active)) if x else None,
                                    "samples_with_active_branch": int(np.any(active, axis=1).sum()) if x else 0,
                                    "sensitivity_available": int(sum(jvalid))}
        if split == "train":
            train_x, train_y = np.asarray(x), np.asarray(y)
    for name, values in (("x", train_x), ("y", train_y)):
        arrays[f"{name}_mean"] = values.mean(axis=0)
        std = values.std(axis=0)
        arrays[f"{name}_scale"] = np.where(std > 1e-6, std, 1.0)
    arrays.update(m_g=relation.m_g, m_d=relation.m_d, branch_constant=relation.constant)
    np.savez_compressed(directory / "dataset.npz", **arrays)
    write_json(directory / "manifest.json", manifest)
    return manifest


def valid_label(result, gens, branch_ids):
    """Never turn a failed, incomplete or non-finite solve into supervised data."""
    return (result.status == "ok" and result.converged and result.metrics.get("feasible") is True
            and set(result.generation_mw) == {g.asset_id for g in gens}
            and tuple(b.asset_id for b in result.branches) == tuple(branch_ids)
            and len(result.active_lower) == len(gens) and len(result.active_upper) == len(gens)
            and all(np.isfinite(list(result.generation_mw.values())))
            and all(np.isfinite([b.p_from_mw, b.p_to_mw, b.loading_percent]).all() for b in result.branches)
            and result.objective is not None and np.isfinite(result.objective))


def load_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    with np.load(directory / "dataset.npz", allow_pickle=False) as values:
        arrays = {k: values[k] for k in values.files}
    return manifest, arrays
