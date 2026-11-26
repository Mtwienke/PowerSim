"""Clean-slate, resumable, four-worker DC study across the six requested grids."""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import tarfile

import numpy as np
from pandapower.pypower.idx_brch import RATE_A
from scipy.optimize import linprog

from ..core.config import SimulationConfig
from ..core.dc import dc_relation
from ..core.solve import solve_reference
from ..core.workers import ordered_pool, validate_workers
from ..instance_generation.data import (DatasetConfig, RenewableConfig, _init_reference_worker, _scenarios,
                                        _solve_scenario, package_versions, scenario_design, valid_label, write_json)
from ..instance_generation.network import build_dispatch_inputs, make_case, nodal_loads
from ..solvers.gao_double_neural.ml import artifact_hash
from .reproduction import (METHODS, StageRunner, SuiteConfig, _configuration_file, _now, _save,
                           _summarize, decoder_configuration, suite_plan, training_configuration, verify_suite)


BENCHMARKS = ("case30", "germany30", "case57", "case118", "case300", "case1354pegase")
RATES = {
    "case30": (.3558, .4071, .4724),
    "germany30": (.3558, .4071, .4724),
    "case57": (.3428, .3951, .4596, .5031),
    "case118": (.3211, .3391, .3878, .4426),
    "case300": (.1950, .2327, .2605),
    "case1354pegase": (.1889, .2185, .2459),
}
INSTALLATION_PAIRS = dict(zip(BENCHMARKS, (2, 2, 4, 8, 16, 32), strict=True))
TABLE_III = {
    "case57": ((99.99, 66.21, 44.55), (99.98, 74.34, 67.55), (99.98, 61.17, 47.20),
               (99.98, 88.37, 83.10), (99.98, 98.91, 97.06)),
    "case118": ((100, 94.95, 34.86), (100, 94.66, 32.99), (99.99, 95.24, 34.08),
                (99.99, 99.98, 97.45), (100, 99.99, 98.23)),
    "case300": ((99.71, 55.94, 45.55), (99.74, 56.83, 46.53), (99.72, 50.31, 43.57),
                (99.34, 97.87, 95.60), (99.72, 97.77, 96.91)),
    "case1354pegase": ((98.95, 97.25, 91.62), (98.89, 97.57, 92.16), (99.77, 98.86, 94.50),
                      (99.85, 99.32, 96.76), (99.88, 99.39, 97.04)),
}
PAPER_METHODS = ("M1", "M2", "M3", "M4", "M5")
CLEAN_PROFILE = "clean-source-informed"


@dataclass(frozen=True)
class CleanConfig:
    benchmarks: tuple[str, ...] = BENCHMARKS
    workers: int = 4
    smoke: bool = False
    data_seed: int = 20261201
    test_seed: int = 20261202
    probe_seed: int = 20261203
    probe_test_seed: int = 20261204
    training_seeds: tuple[int, ...] = (20261211, 20261212, 20261213)

    def __post_init__(self):
        validate_workers(self.workers)
        if not self.benchmarks or len(set(self.benchmarks)) != len(self.benchmarks) or set(self.benchmarks)-set(BENCHMARKS):
            raise ValueError("Choose distinct supported benchmarks")
        seeds = (self.data_seed, self.test_seed, self.probe_seed, self.probe_test_seed, *self.training_seeds)
        if not self.training_seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0:
            raise ValueError("Training, data, preflight and reserved-test seeds must be distinct")

    @property
    def seeds(self):
        return self.training_seeds[:1] if self.smoke else self.training_seeds

    @property
    def methods(self):
        return ("M1", "M5") if self.smoke else METHODS


def data_config(cfg, network, *, probe=False):
    seed_offset = 10000 if cfg.smoke else 0
    return DatasetConfig(
        simulation=SimulationConfig(benchmark=network, branch_limit_scale=.9 if network == "germany30" else 1.),
        renewable=RenewableConfig(installation_pairs=INSTALLATION_PAIRS[network]),
        seed=(cfg.probe_seed if probe else cfg.data_seed)+seed_offset,
        placement_seed=cfg.data_seed+seed_offset,
        test_seed=(cfg.probe_test_seed if probe else cfg.test_seed)+seed_offset,
        train_cases=16 if probe else 8 if cfg.smoke else 10000,
        validation_cases=8 if probe else 4 if cfg.smoke else 1000,
        test_cases=8 if probe else 4 if cfg.smoke else 2000,
        penetration_rates=RATES[network],
        penetration_basis="nominal_demand" if network == "germany30" else "total_installed",
        renewable_capacities_mw=(185., 230., 300.) if network == "case30" else (),
        sampling_policy="attempts" if probe else "valid",
        max_attempt_factor=5,
        demand_shift_mean_factor=1.10, preserve_fixed_injections=network in ("case300", "case1354pegase"),
        training_sensitivities_only=True, record_format="jsonl.gz",
    )


def held_out_splits(data):
    return ["iid_test", *(f"ood_test_{i}" for i in range(1, len(data.penetration_rates))), "ood_demand"]


def plan(cfg):
    budgets = SuiteConfig.smoke() if cfg.smoke else SuiteConfig()
    settings = {}
    for seed in cfg.seeds:
        budget = replace(budgets, seed=seed)
        settings[str(seed)] = {
            "training": {k: {**asdict(training_configuration(budget, "source-informed", k)),
                             "jacobian_mode": "chain", "cpu_threads": cfg.workers} for k in cfg.methods},
            "decoder": {**decoder_configuration(budget, "source-informed"), "threshold": None, "cpu_threads": cfg.workers},
        }
    return json.loads(json.dumps({
        "configuration": asdict(cfg), "methods": cfg.methods, "seeds": cfg.seeds, "settings": settings,
        "profile": CLEAN_PROFILE, "protocol": "docs/clean-suite.md",
        "paper_table_ii": suite_plan(budgets)["paper_table_ii"], "paper_table_iii": {
            "source": "Gao et al., DOI 10.1109/TNNLS.2024.3437741, p. 8056, Table III",
            "methods": PAPER_METHODS, "networks": {net: {"rates": RATES[net][:3], "accuracy_pct": rows}
                                                 for net, rows in TABLE_III.items()}},
        "networks": {net: {"full": asdict(data_config(cfg, net)), "preflight": asdict(data_config(cfg, net, probe=True)),
                           "held_out": held_out_splits(data_config(cfg, net))} for net in cfg.benchmarks},
        "preflight_policy": "Independent draws at the same placement; any empty split blocks the full run for that grid.",
        "checkpoint_policy": "Final epoch fixed; one M1 parent per grid and training seed; all comparisons share test cases.",
        "gate_policy": "Every model validation q95 physical RMSE; secondary fixed 6 MW RMSE diagnostic for M5.",
        "attack_policy": "Saved 1 MW self-FGSM protocol on IEEE-30 IID and renewable OOD1 only; not on other grids.",
        "M3_status": "M3-example branch penalty is not genuine DeepOPF.",
    }))


def _case_lp(relation, net, inputs, limits):
    """Independent nominal feasibility diagnostic, never another label solver."""
    x = np.asarray(nodal_loads(net))
    keep = (limits > 0) & (limits < 1e8)
    m = relation.m_g[keep]
    c = (relation.constant-relation.m_d @ x)[keep]
    bounds = [(g.p_min_mw, g.p_max_mw) for g in inputs["generators"]]
    p = linprog(np.zeros(len(bounds)), A_ub=np.vstack((m, -m)), b_ub=np.r_[limits[keep]-c, limits[keep]+c],
                A_eq=np.ones((1, len(bounds))), b_eq=[x.sum()+relation.balance_constant_mw],
                bounds=bounds, method="highs-ipm")
    return {"solver": "SciPy HiGHS IPM", "status": int(p.status), "feasible": bool(p.success),
            "message": p.message, "scope": "Nominal DC constraints only; feasibility diagnostic, not labels."}


def preflight(output, config_file, workers):
    values = json.loads(Path(config_file).read_text())
    config = DatasetConfig(**{**values, "simulation": SimulationConfig(**values["simulation"]),
                              "renewable": RenewableConfig(**values["renewable"])})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    net = make_case(config.simulation)
    inputs = build_dispatch_inputs(net, config.simulation)
    relation = dc_relation(net)
    bounds = net._ppc["branch"][:, RATE_A].real.copy()
    nominal = solve_reference(net, config.simulation)
    gross, fixed, sites, _, design = scenario_design(config, net, inputs)
    generators = inputs["generators"]
    splits = {}
    with ordered_pool(workers, _init_reference_worker, (config.simulation,)) as map_cases:
        for (split, count, rate), stream, (capacity, _) in design:
            attempts = _scenarios(config, split, count, rate, stream, gross, sites, capacity, fixed)
            outcomes = []
            for row, solve in map_cases(_solve_scenario, attempts):
                outcomes.append({
                    "scenario_id": row["scenario_id"], "attempt": row["number"],
                    "status": solve.status, "converged": solve.converged,
                    "valid": bool(valid_label(solve, generators, relation.branch_ids)), "diagnostics": solve.diagnostics,
                    "seconds": solve.solver_seconds,
                })
            splits[split] = {"attempted": len(outcomes), "valid": sum(row["valid"] for row in outcomes),
                             "statuses": dict(Counter(row["status"] for row in outcomes)), "cases": outcomes}
    write_json(output/"preflight.json", {
        "benchmark": config.simulation.benchmark, "model_id": inputs["context"].network_fingerprint, "workers": workers,
        "nominal_reference": {"status": nominal.status, "diagnostics": nominal.diagnostics,
                              "converged": nominal.converged, "objective": nominal.objective},
        "nominal_lp": _case_lp(relation, net, inputs, bounds) if nominal.status != "ok" else None,
        "uniform_linear_generation_cost": len({g.marginal_cost_per_mwh for g in generators}) == 1
                                           and not any(g.cost_quadratic for g in generators),
        "split_results": splits, "eligible": all(v["valid"] > 0 for v in splits.values()),
    })


def _snapshot(root):
    """Freeze current working bytes; do not require or create a Git commit."""
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo))
    files = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                                    cwd=repo).decode().split("\0")
    archive, source = root/"source.tar", root/"source"
    with tarfile.open(archive, "x") as tar:
        for filename in sorted(set(files)):
            if filename and (Path(filename).parts[0] in ("powersim", "docs", "tests", "examples")
                             or filename in ("pyproject.toml", "README.md") or filename.startswith("constraints-")):
                path = repo/filename
                if path.is_file() and not path.is_symlink():
                    tar.add(path, arcname=filename, recursive=False)
    source.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(source, filter="data")
    return revision, dirty


def _dataset_stage(runner, cfg, net, data_file):
    data = data_config(cfg, net)
    required = ["manifest.json", "dataset.npz", "network.json",
                *(f"{s}.{data.record_format}" for s in ("train", "validation", *held_out_splits(data)))]
    try:
        return runner.run(f"{net}--dataset", lambda out: [
            "dataset", out, "--config", data_file, "--workers", cfg.workers], required, inputs=[data_file])
    except RuntimeError:
        attempted = runner.root / runner.state["stages"][f"{net}--dataset"]["directory"]
        diagnostic = attempted/"generation_error.json"
        if not diagnostic.is_file():
            raise
        stage = runner.state["stages"][f"{net}--dataset"]
        stage.update(status="unavailable", diagnostic=json.loads(diagnostic.read_text()), finished_utc=_now())
        runner.save()
        return None


def _evaluate(runner, cfg, entries, key, model, dataset, split, seed, mode, decoder=None):
    args = ["--decoder", decoder] if decoder else []
    directory = runner.run(f"{key}--{split}--{mode}", lambda out: [
        "evaluate", dataset, model, out, "--split", split, "--workers", cfg.workers,
        "--prediction-format", "jsonl.gz", *args],
        ["evaluation.json", "predictions.npz", "predictions.jsonl.gz"],
        inputs=[dataset/"manifest.json", dataset/"dataset.npz", model, *([decoder] if decoder else [])])
    entries.append({"benchmark": key.split("--")[0], "seed": seed, "profile": CLEAN_PROFILE,
                    "method": key.split("--")[-1], "checkpoint": "last", "split": split, "mode": mode,
                    "directory": str(directory.relative_to(runner.root)),
                    "report": json.loads((directory/"evaluation.json").read_text())})
    _summarize(runner.root, entries)


def _grid(runner, cfg, network, dataset, planned, entries):
    configs = runner.root/"configurations"
    inputs = [dataset/name for name in ("manifest.json", "dataset.npz", "network.json")]
    for seed in cfg.seeds:
        seed_settings = planned["settings"][str(seed)]
        decoder_file = _configuration_file(configs/f"seed-{seed}--decoder.json", seed_settings["decoder"])
        parent = None
        for method in cfg.methods:
            key = f"{network}--seed-{seed}--{method}"
            train_file = _configuration_file(configs/f"{key}.json", seed_settings["training"][method])
            warm = [parent] if parent is not None else []
            model = runner.run(f"{key}--train", lambda out: [
                "train", dataset, out, "--config", train_file, *(["--initialize-from", parent] if warm else [])],
                ["model.pt", "best.pt", "last.pt", "training.json"], inputs=[*inputs, train_file, *warm])/"model.pt"
            if method == "M1":
                parent = model
            decoder = runner.run(f"{key}--decoder", lambda out: [
                "decoder", dataset, model, out, "--config", decoder_file],
                ["decoder.pt", "decoder-best.pt", "decoder-last.pt", "decoder.json"],
                inputs=[*inputs, model, decoder_file])/"decoder.pt"
            fixed = None
            if method == "M5":
                fixed = runner.run(f"{key}--fixed6", lambda out: [
                    "calibrate", dataset, model, decoder, out, "--error-units", "rmse_mw", "--threshold", 6.],
                    ["decoder.pt", "decoder.json"], inputs=[*inputs, model, decoder])/"decoder.pt"
            for split in planned["networks"][network]["held_out"]:
                # This stage already saves and assesses both raw and hybrid dispatch.
                _evaluate(runner, cfg, entries, key, model, dataset, split, seed, "hybrid-q95-rmse", decoder)
                if fixed:
                    _evaluate(runner, cfg, entries, key, model, dataset, split, seed, "hybrid-fixed6-rmse", fixed)
            if network == "case30":
                for source, attack in (("iid_test", "attack1"), ("ood_test_1", "attack2")):
                    data = runner.run(f"{key}--{attack}--data", lambda out: [
                        dataset, model, out, "--split", source, "--epsilon-mw", 1., "--workers", cfg.workers],
                        ["manifest.json", "dataset.npz", "network.json", "attack.json", f"{attack}.jsonl.gz"],
                        inputs=[*inputs, model], module="powersim.research.attacks")
                    _evaluate(runner, cfg, entries, key, model, data, attack, seed, "raw-fgsm")


def run(output, cfg, *, resume=False):
    root = Path(output).resolve()
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
    lock = root/".suite.lock"
    with lock.open("x") as stream:
        stream.write(f"{os.getpid()}\n")
    runner, handlers = None, {}
    source, archive, manifest_file = root/"source", root/"source.tar", root/"suite_manifest.json"
    try:
        planned = plan(cfg)
        if resume:
            manifest = json.loads(manifest_file.read_text())
            if (manifest["plan"] != planned or manifest["versions"] != package_versions()
                    or manifest["runner_sha256"] != artifact_hash(__file__)
                    or manifest["source_archive_sha256"] != artifact_hash(archive)):
                raise ValueError("Resume with the same frozen source, settings, and environment")
            state = json.loads((root/"status.json").read_text())
        else:
            revision, dirty = _snapshot(root)
            state = {"stages": {}, "started_utc": _now()}
            manifest = {"plan": planned, "source_revision": revision, "source_worktree_dirty": dirty,
                        "source_archive_sha256": artifact_hash(archive), "runner_sha256": artifact_hash(__file__),
                        "source_file_sha256": {str(path.relative_to(source)): artifact_hash(path)
                                               for path in sorted(source.rglob("*")) if path.is_file()},
                        "versions": package_versions(), "platform": platform.platform()}
            _save(manifest_file, manifest)
        (root/"logs").mkdir(exist_ok=True)
        state.update(status="running", runner_pid=os.getpid(), child_pid=None)
        state.pop("diagnostic", None)
        runner = StageRunner(root, source, state, manifest["source_file_sha256"], manifest["versions"])
        runner.save()

        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")

        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, interrupt)

        blockers, entries, probes, data_files = {}, [], {}, {}
        for net in cfg.benchmarks:
            net_plan = planned["networks"][net]
            pre_file = _configuration_file(root/"configurations"/f"{net}--probe.json", net_plan["preflight"])
            data_files[net] = _configuration_file(root/"configurations"/f"{net}--data.json", net_plan["full"])
            probe = runner.run(f"{net}--preflight", lambda out: [
                out, "--probe-config", pre_file, "--workers", cfg.workers],
                ["preflight.json"], inputs=[pre_file], module="powersim.research.clean_suite")
            info = json.loads((probe/"preflight.json").read_text())
            empty = [split for split, result in info["split_results"].items() if not result["valid"]]
            probes[net] = {"directory": str(probe.relative_to(root)), "eligible": info["eligible"],
                           "empty_splits": empty, "nominal": info["nominal_reference"], "nominal_lp": info["nominal_lp"]}
            if empty:
                blockers[net] = "No validated independent preflight labels for " + ", ".join(empty)
                runner.log(f"UNAVAILABLE {net}: {blockers[net]}")
            _save(root/"preflight_summary.json", probes)
        state["unavailable"] = blockers
        runner.save()
        datasets = {}
        for net in cfg.benchmarks:
            if net in blockers:
                continue
            dataset = _dataset_stage(runner, cfg, net, data_files[net])
            if dataset is None:
                blockers[net] = state["stages"][f"{net}--dataset"]["diagnostic"]["diagnostic"]
                runner.log(f"UNAVAILABLE {net}: {blockers[net]}")
                runner.save()
                continue
            datasets[net] = dataset
        for net, dataset in datasets.items():
            _grid(runner, cfg, net, dataset, planned, entries)
        _summarize(root, entries)
        verified = verify_suite(root, {"stages": {k: v for k, v in state["stages"].items() if v["status"] == "complete"}})
        state.update(status="complete_with_unavailable" if blockers else "complete", unavailable=blockers,
                     verified_output_hashes=verified, completed_evaluations=len(entries), current_stage=None, finished_utc=_now())
        runner.save()
        runner.log(f"SUITE {state['status']}: {len(entries)} evaluations; {len(blockers)} benchmark(s) unavailable")
        return {"root": str(root), "status": state["status"], "evaluations": len(entries), "unavailable": blockers}
    except BaseException as exc:
        if runner:
            runner.stop_child()
            current = runner.state["stages"].get(runner.state.get("current_stage"))
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            if current and current["status"] == "running":
                current["status"] = status
            runner.state.update(status=status, child_pid=None, diagnostic=f"{type(exc).__name__}: {exc}")
            runner.save()
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=BENCHMARKS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="Separate tiny plumbing run; never a scientific result")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--plan", action="store_true")
    operation.add_argument("--resume", action="store_true")
    operation.add_argument("--probe-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.probe_config:
        preflight(args.output, args.probe_config, args.workers)
    else:
        cfg = CleanConfig(benchmarks=tuple(args.benchmarks), workers=args.workers, smoke=args.smoke)
        print(json.dumps(plan(cfg) if args.plan else run(args.output, cfg, resume=args.resume), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
