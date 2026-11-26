"""Read-only adapters for completed PowerSim suites and standalone references."""
from dataclasses import dataclass
from functools import cached_property, lru_cache
import gzip
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import pandapower as pp

from ..core.dc import dc_relation
from ..instance_generation.data import split_record_file
from ..instance_generation.network import generators, network_fingerprint
from .report_metrics import BOUND_MW, Physics


RUNS = {
    "legacy-2026": "full-suite-20261118T044204Z",
    "legacy-2027": "full-suite-seed2027-20261119T021858Z",
    "pilot": "gao-pilot-ieee30-seed2027-20261119T064744Z",
    "reconstruction": "gao-reconstruction-full-20261119T133302Z",
}


def read_json(path):
    return json.loads(path.read_text())


def records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def sha256(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


class Inventory:
    def __init__(self, base, requested, *, clean_suites=()):
        self.base = base.resolve()
        self.runs, self.expected, self.inputs, self.unavailable = {}, {}, {}, []
        self.clean_suites = {f"clean:{Path(path).resolve().name}": Path(path).resolve() for path in clean_suites}
        if len(self.clean_suites) != len(clean_suites):
            raise ValueError("Clean suites must have distinct run directory names")
        for name, root in [*((k, self.run_root(RUNS[k])) for k in requested), *self.clean_suites.items()]:
            if not root.is_dir():
                self.unavailable.append({"item": name, "reason": f"Run directory missing: {root}"})
                continue
            summary_path = root / ("evaluation_summary.json" if name == "pilot" else "summary.json")
            state_path = root / "status.json"
            source_path = root / ("experiment_manifest.json" if name == "pilot" else "suite_manifest.json")
            state = read_json(state_path)
            statuses = ("complete", "complete_with_unavailable") if name in self.clean_suites else ("complete",)
            if state["status"] not in statuses:
                raise ValueError(f"Only completed runs may be indexed: {root}")
            if name in self.clean_suites:
                self.unavailable.extend({"item": f"{name}: {net}", "reason": why}
                                        for net, why in state.get("unavailable", {}).items())
            stages = {}
            for stage in state["stages"].values():
                if name in self.clean_suites and stage["status"] != "complete":
                    continue
                directory = root / (stage.get("directory") or stage["output"])
                stages[directory] = stage
                for path, checksum in stage["sha256"].items():
                    file = (directory if "/" not in path else root) / path
                    previous = self.expected.setdefault(file, checksum)
                    if previous != checksum:
                        raise ValueError(f"Conflicting frozen hashes: {file}")
            source = read_json(source_path)
            provenance = source.get("configuration", source)
            self.runs[name] = {
                "root": self.relative(root), "state": state_path, "summary": summary_path,
                "source": source_path, "source_revision": provenance["source_revision"],
                "source_archive_sha256": provenance.get("source_archive_sha256"),
                "versions": provenance.get("versions"), "stages": stages,
            }
            # The pilot reuses a historical dataset even if only the pilot is selected.
            if name == "pilot":
                dataset_path = self.remap(provenance["dataset"]) if isinstance(provenance["dataset"], str) else None
                if dataset_path is not None:
                    for filename, checksum in provenance["dataset_sha256"].items():
                        self.expected.setdefault(dataset_path / filename, checksum)
        self._datasets = {}

    def relative(self, path):
        return str(path.relative_to(self.base)) if path.is_relative_to(self.base) else str(path)

    def run_root(self, folder):
        active = self.base / folder
        archived = self.base / "archive" / folder
        return archived if not active.is_dir() and archived.is_dir() else active

    def remap(self, path, run=None):
        parts = Path(path).parts
        if run in self.clean_suites:
            root = self.clean_suites[run]
            # Stored absolute commands are historical; reconstruct from the
            # artifact layout even when a completed suite has been relocated.
            if "stages" in parts:
                idx = len(parts)-1-parts[::-1].index("stages")
                return root.joinpath(*parts[idx:])
            raise ValueError(f"Expected a staged path in {run}: {path}")
        for i, part in enumerate(parts):
            if part in RUNS.values():
                return self.run_root(part).joinpath(*parts[i+1:])
        raise ValueError(f"Cannot resolve a frozen suite path: {path}")

    def checked(self, path):
        key = self.relative(path)
        if key not in self.inputs:
            actual = sha256(path)
            expected = self.expected.get(path)
            if expected is not None and actual != expected:
                raise ValueError(f"Frozen artifact hash mismatch: {path}")
            self.inputs[key] = {"sha256": actual, "verified_against_frozen_index": expected is not None}
        return path

    def dataset(self, path):
        if path not in self._datasets:
            self._datasets[path] = Dataset(path, self)
        return self._datasets[path]

    def evaluations(self):
        for run, source in self.runs.items():
            self.checked(source["source"])
            self.checked(source["state"])
            entries = read_json(self.checked(source["summary"]))["evaluations"]
            for row in entries:
                directory = self.base / source["root"] / row["directory"]
                stage = source["stages"][directory]
                call = stage["command"]
                idx = call.index("evaluate")
                dataset, model = (self.remap(value, run) for value in call[idx+1:idx+3])
                decoder = self.remap(call[call.index("--decoder")+1], run) if "--decoder" in call else None
                common = (run, row.get("profile", run), row.get("method", row.get("variant")),
                          row.get("checkpoint", "best_validation"), row["split"])
                if run in self.clean_suites and row["mode"] == "hybrid-q95-rmse":
                    yield Evaluation(*common, "raw", directory, dataset, model, None, stage["seconds"], self, True)
                yield Evaluation(*common, row["mode"], directory, dataset, model, decoder, stage["seconds"], self)


@dataclass(frozen=True)
class Evaluation:
    run: str
    profile: str
    method: str
    checkpoint: str
    split: str
    mode: str
    directory: Path
    dataset: Path
    model: Path
    decoder: Path | None
    stage_seconds: float
    inventory: Inventory
    raw_view_of_hybrid_stage: bool = False

    @property
    def report(self):
        return read_json(self.inventory.checked(self.directory / "evaluation.json"))

    def predictions(self, ids, report):
        inventory = self.inventory
        with np.load(inventory.checked(self.directory / "predictions.npz"), allow_pickle=False) as a:
            raw, chosen, accepted = (a[k] for k in ("raw_generation_mw", "hybrid_generation_mw", "accepted"))
            scores = a["reconstruction_error"] if "reconstruction_error" in a else None
        if len(raw) != len(ids) or chosen.shape != raw.shape or accepted.shape != (len(ids),):
            raise ValueError(f"Prediction/label coverage mismatch: {self.directory}")
        if self.decoder and scores is None:
            # Old NPZ files lack scores. Check every archived JSONL ID against
            # the label order while recovering the original single-case score.
            detail = self.directory / report.get("prediction_file", "predictions.jsonl")
            values = []
            for sid, row in zip(ids, records(inventory.checked(detail)), strict=True):
                if row["scenario_id"] != sid:
                    raise ValueError(f"Prediction/label order mismatch: {detail}")
                values.append(row["reconstruction_error"])
            scores = np.asarray(values, float)
        return np.asarray(raw, float), np.asarray(chosen, float), accepted.astype(bool), scores


@dataclass
class LabelSplit:
    x: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    cost: np.ndarray
    flows: np.ndarray
    attempts: list
    active: np.ndarray

    @cached_property
    def lookup(self):
        return {row["sample_id"]: row for row in self.attempts}


class Dataset:
    def __init__(self, path, inventory):
        self.path = path
        self.inventory = inventory
        self.manifest = read_json(self.checked("manifest.json"))
        self.config = self.manifest["configuration"]
        if self.config["simulation"]["physics_mode"] != "dc":
            raise ValueError(f"Report evaluator requires DC data: {path}")
        self.npz = self.checked("dataset.npz")
        self.id = inventory.inputs[inventory.relative(self.npz)]["sha256"]
        self.benchmark = self.config["simulation"]["benchmark"]
        self.gen_ids = self.manifest["generator_ids"]
        self.branch_ids = self.manifest["branch_ids"]
        self._cached_splits = {}

    def checked(self, name):
        return self.inventory.checked(self.path / name)

    @cached_property
    def net(self):
        net = pp.from_json(self.checked("network.json"))
        if network_fingerprint(net) != self.manifest["model_id"]:
            raise ValueError(f"Frozen network does not match its manifest: {self.path}")
        return net

    @cached_property
    def network(self):
        net = self.net
        provenance = self.manifest.get("source_provenance") or {}
        version = ("v2" if "ward" in provenance.get("region_partition", "").lower()
                   or provenance.get("schema_version") == 2 else "v1")
        return self.benchmark if self.benchmark != "germany30" else f"germany30-{version}"

    @cached_property
    def provenance(self):
        net = self.net
        return {
            "path": self.inventory.relative(self.path), "network": self.network,
            "model_id": self.manifest["model_id"], "configuration": self.config,
            "source_provenance": self.manifest.get("source_provenance"), "base_mva": float(net.sn_mva),
            "bus_count": len(self.manifest["bus_ids"]), "loads_in_snapshot": len(net.load),
            "gross_demand_nodes": int(np.count_nonzero(self.manifest["nominal_gross_load_mw"])),
            "nominal_gross_mw": sum(self.manifest["nominal_gross_load_mw"]),
            "conventional_sources": len(self.gen_ids), "nominal_fixed_injections": len(net.sgen),
            "branches": len(self.branch_ids), "generator_ids": self.gen_ids, "branch_ids": self.branch_ids,
            "generators": self.manifest["generators"], "installations": self.manifest["installations"],
            "penetration_definition": self.manifest.get("penetration_definition") or
            "renewable nameplate MW / nominal gross demand MW",
            "capacity_regimes": self.manifest.get("capacity_regimes"),
            "splits": {k: {a: v for a, v in s.items() if a != "failures"} for k, s in self.manifest["splits"].items()},
            "attack": self.manifest.get("attack"), "versions": self.manifest.get("versions"),
        }

    def arrays(self, *keys):
        with np.load(self.npz, allow_pickle=False) as arrays:
            return [arrays[key] for key in keys]

    def split(self, name):
        if name not in self._cached_splits:
            self._cached_splits[name] = split_arrays(self, name)
        return self._cached_splits[name]

    @cached_property
    def relation(self):
        return dc_relation(self.net)

    def physics(self, split, labels):
        m_g, m_d, constant = self.arrays("m_g", "m_d", "branch_constant")
        relation = self.relation
        if (list(relation.branch_ids) != self.branch_ids
                or any(not np.allclose(a, b, atol=1e-7, rtol=0)
                       for a, b in zip((m_g, m_d, constant), (relation.m_g, relation.m_d, relation.constant)))):
            raise ValueError(f"Saved DC map disagrees with frozen network: {self.path}")
        gens = generators(self.net)
        if [g.asset_id for g in gens] != self.gen_ids:
            raise ValueError(f"Generator ordering differs: {self.path}")
        def gen(field):
            return np.array([getattr(g, field) for g in gens])
        limit = next((row["limits_mw"] for row in labels.attempts if row["reference_valid"]), None)
        if limit is None:
            from pandapower.pypower.idx_brch import RATE_A
            limit = self.net._ppc["branch"][:, RATE_A].real
            limit = np.where((limit > 0) & (limit < 1e8), limit, np.inf)
        if any(not np.array_equal(limit, row["limits_mw"]) for row in labels.attempts if row["reference_valid"]):
            raise ValueError(f"Scenario-varying branch limits are unsupported: {self.path}/{split}")
        model = Physics(m_g, m_d, constant, relation.balance_constant_mw,
                        gen("p_min_mw"), gen("p_max_mw"), -limit, limit,
                        gen("cost_constant"), gen("marginal_cost_per_mwh"), gen("cost_quadratic"))
        check = model.evaluate(labels.y, labels.x, labels.cost)
        if (not np.allclose(check["flow_mw"], labels.flows, atol=BOUND_MW, rtol=0)
                or not np.allclose(check["predicted_cost_eur_h"], labels.cost, atol=1e-6, rtol=1e-10)
                or not check["feasible"].all()):
            raise ValueError(f"Saved labels failed DC map, feasibility or cost validation: {self.path}/{split}")
        return model


def split_arrays(dataset, name):
    x, y, ids, cost, flows, active = dataset.arrays(
        *(f"{name}_{key}" for key in ("x", "y", "scenario_ids", "objective", "flows", "active_branch")))
    valid_ids = set(ids)
    attempts = []
    valid = []
    for row in records(dataset.inventory.checked(split_record_file(dataset.path, name, dataset.manifest))):
        sid, ref = row["scenario_id"], row["reference"]
        labeled = sid in valid_ids
        attempted = {
            "sample_id": sid, "attempt": row["number"], "source_sample_id": row.get("source_scenario_id"),
            "input_id": row.get("input_id"), "reference_valid": labeled, "reference_status": ref["status"],
            "reference_converged": ref["converged"], "reference_diagnostics": " | ".join(ref["diagnostics"]),
            "realized_penetration": row.get("realized_penetration"),
            "reference_solver_seconds": ref["solver_seconds"],
        }
        if labeled:
            if ref["status"] != "ok" or not ref["converged"] or not ref["metrics"]["feasible"]:
                raise ValueError(f"Unverified OR failed OPF included as reference: {sid}")
            attempted["limits_mw"] = np.array([
                b["limit_mw"] if b["limit_mw"] is not None else np.inf for b in ref["branches"]
            ])
            if [b["asset_id"] for b in ref["branches"]] != dataset.branch_ids:
                raise ValueError(f"Branch ordering differs: {sid}")
            idx = len(valid)
            if idx >= len(ids) or ids[idx] != sid or not np.array_equal(x[idx], row["net_load_mw"]):
                raise ValueError(f"Reference/NPZ order mismatch: {sid}")
            valid.append(sid)
        attempts.append(attempted)
    stats = dataset.manifest["splits"][name]
    if (len(attempts) != stats["requested"] or len(valid) != stats["valid"] or len(valid) != len(ids)
            or len({a["sample_id"] for a in attempts}) != len(attempts)):
        raise ValueError(f"Attempt coverage mismatch: {dataset.path}/{name}")
    return LabelSplit(x, y, ids, cost, flows, attempts, active)


@lru_cache(maxsize=64)
def load_gate(model_path, decoder_path, dataset):
    """Use recorded validation grids, reconstructing only legacy validation scores."""
    import torch
    from ..solvers.gao_double_neural.ml import (RECONSTRUCTION_SCORES, calibrate_threshold, load_decoder,
                                               load_model, reconstruction_errors, tensor)

    torch.set_num_threads(1)
    model, metadata = load_model(model_path)
    decoder, meta = load_decoder(decoder_path, model_path, model)
    meta = {**meta, "score_units": RECONSTRUCTION_SCORES[meta["error_units"]][0]}
    validation_ids, validation_x = dataset.arrays("validation_scenario_ids", "validation_x")
    if (not np.array_equal(meta["validation_scenario_ids"], validation_ids)
            or metadata["model_id"] != dataset.manifest["model_id"]):
        raise ValueError("Checkpoint or decoder does not match the archived validation set")
    thresholds = meta.get("validation_thresholds")
    source = "saved_validation_quantiles"
    if thresholds is None:
        source = "inferred_on_frozen_validation_inputs_only"
        with torch.no_grad():
            scores = reconstruction_errors(model, decoder, tensor(validation_x), meta["error_units"]).numpy()
        thresholds = [calibrate_threshold(scores, q) for q in (.05, .1, .25, .5, .75, .9, .95, .99, 1.)]
    return meta, thresholds, source
