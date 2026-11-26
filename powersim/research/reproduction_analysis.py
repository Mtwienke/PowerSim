"""Post-completion descriptive analysis of a frozen reconstruction suite.

No training, checkpoint selection, norm selection or calibration occurs here.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

from ..instance_generation.data import write_json
from ..solvers.gao_double_neural.ml import artifact_hash
from .reproduction import METHODS, PROFILES, SPLITS, verify_suite


def pooled_reports(reports):
    cases = sum(r["labeled_cases"] for r in reports)
    source_count = sum(r["raw_accuracy"]["generator_accuracy_denominator"] for r in reports)
    finite = [(r, r["raw_accuracy"].get("accuracy_samples", r["labeled_cases"])) for r in reports
              if r["raw_accuracy"]["mae_mw"] is not None and r["raw_accuracy"]["rmse_mw"] is not None]
    finite_cases = sum(count for _, count in finite)
    balances = [(r["raw_physics"]["balance_mean_absolute_mw"],
                 r["labeled_cases"]-r["raw_physics"].get("pf_failures", 0)) for r in reports
                if r["raw_physics"]["balance_mean_absolute_mw"] is not None]
    balance_cases = sum(count for _, count in balances)
    return {
        "scenarios": cases,
        "generator_accuracy_lt_1mw": sum(r["raw_accuracy"]["generator_accuracy_count"] for r in reports)/source_count if source_count else None,
        "joint_accuracy_lt_1mw": sum(r["raw_accuracy"]["joint_accuracy_count"] for r in reports)/cases if cases else None,
        "mae_mw": sum(r["raw_accuracy"]["mae_mw"]*count for r, count in finite)/finite_cases if finite_cases else None,
        "rmse_mw": math.sqrt(sum(r["raw_accuracy"]["rmse_mw"]**2*count for r, count in finite)/finite_cases) if finite_cases else None,
        "finite_error_scenarios": finite_cases,
        "feasible_rate": sum(r["raw_physics"]["feasible_predictions"] for r in reports)/cases if cases else None,
        "balance_mean_absolute_mw": sum(value*count for value, count in balances)/balance_cases if balance_cases else None,
        "source_like_branch_events_beyond_limit_plus_1mw": sum(
            r["raw_physics"]["source_like_branch_events_beyond_limit_plus_1mw"] for r in reports),
    }


def _fmt(value, *, percent=False):
    if value is None:
        return "—"
    return f"{100*value:.2f}%" if percent else f"{value:.4f}"


def _table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |",
                      "| "+" | ".join("---" for _ in headers)+" |",
                      *["| "+" | ".join(str(value) for value in row)+" |" for row in rows]])


def analyze_suite(root, output=None):
    root = Path(root).resolve()
    output = Path(output).resolve() if output is not None else root/"analysis"
    if output.exists():
        raise ValueError("Analysis output already exists; use a new --output directory")
    state = json.loads((root/"status.json").read_text())
    if state["status"] != "complete":
        raise ValueError("Analyze only a completed frozen suite")
    manifest = json.loads((root/"suite_manifest.json").read_text())
    entries = json.loads((root/"summary.json").read_text())["evaluations"]
    if len(entries) != manifest["plan"]["counts"]["evaluations"]:
        raise ValueError("Evaluation count does not match the frozen plan")
    verified = verify_suite(root, state)
    for filename, checksum in manifest["source_file_sha256"].items():
        if artifact_hash(root/"source"/filename) != checksum:
            raise ValueError(f"Frozen source changed: {filename}")
    if artifact_hash(root/"source.tar") != manifest["source_archive_sha256"]:
        raise ValueError("Frozen source archive changed")
    dataset = root/state["stages"]["dataset"]["directory"]
    data = json.loads((dataset/"manifest.json").read_text())

    def entry(profile, method, split, mode="raw", checkpoint="last"):
        matches = [e for e in entries if (e["profile"], e["method"], e["split"], e["mode"], e["checkpoint"])
                   == (profile, method, split, mode, checkpoint)]
        if len(matches) != 1:
            raise ValueError(f"Missing/duplicate evaluation: {profile}/{method}/{split}/{mode}/{checkpoint}")
        return matches[0]

    raw, hybrid, attacks, best, fitting, gaps = [], [], [], [], [], []
    paper = manifest["plan"]["paper_table_ii"]
    for profile in PROFILES:
        parent_stage = state["stages"][f"{profile}--M1--train"]
        parent_hash = artifact_hash(root/parent_stage["directory"]/"model.pt")
        for method in METHODS:
            reports = [entry(profile, method, split)["report"] for split in SPLITS]
            raw.append({"profile": profile, "method": method,
                        "accuracy_by_regime": {split: r["raw_accuracy"]["generator_accuracy_lt_1mw"]
                                               for split, r in zip(SPLITS, reports)},
                        "by_regime": {split: {"accuracy": r["raw_accuracy"], "physics": r["raw_physics"],
                                              "timing": r["timing"]} for split, r in zip(SPLITS, reports)},
                        "pooled": pooled_reports(reports)})
            train_stage = state["stages"][f"{profile}--{method}--train"]
            train = json.loads((root/train_stage["directory"]/"training.json").read_text())
            if method != "M1" and train["initialization"]["checkpoint_sha256"] != parent_hash:
                raise ValueError("Continuation does not share its profile's final M1 parent")
            fitting.append({
                "profile": profile, "method": method, "configuration": train["configuration"],
                "selected_epoch": train["selected_epoch"], "best_epoch": train["best_epoch"],
                "train": train["training_accuracy"], "validation": train["validation_accuracy"],
                "initial_validation": train["initial_validation_accuracy"],
                "best_validation_mse_mw2": train["best_validation_mse_mw2"],
                "last_validation_mse_mw2": train["last_validation_mse_mw2"],
                "sensitivity_learning": train["sensitivity_learning"],
                "last_epoch_losses": train["history"][-1], "initialization": train["initialization"],
            })
            for split, report in zip(SPLITS, reports):
                h = entry(profile, method, split, "hybrid-fixed6-rmse")["report"]
                hybrid.append({"profile": profile, "method": method, "split": split,
                               **{key: h[key] for key in (
                                   "coverage", "accepted_accuracy", "accepted_physics", "hybrid_accuracy",
                                   "hybrid_physics", "fallback_accuracy", "fallback_failures", "timing",
                                   "decoder", "decoder_norm_sensitivity")}})
                if method in paper:
                    reference = paper[method][SPLITS.index(split)]
                    gaps.append({"profile": profile, "method": method, "split": split,
                                 "paper_percent": reference,
                                 "new_percent": 100*report["raw_accuracy"]["generator_accuracy_lt_1mw"],
                                 "difference_percentage_points": 100*report["raw_accuracy"]["generator_accuracy_lt_1mw"]-reference})
            for attack_split in ("attack1", "attack2"):
                attacked = entry(profile, method, attack_split, "raw-fgsm")
                metadata = attacked["attack"]
                attacks.append({
                    "profile": profile, "method": method, "split": attack_split,
                    "fixed_original_targets_all_attacks": metadata["fixed_original_target_accuracy_all_attacks"],
                    "fresh_opf_targets_valid_attacks": attacked["report"]["raw_accuracy"],
                    "fresh_opf_physics": attacked["report"]["raw_physics"],
                    "attacks": metadata["original_scenarios"], "fresh_reference_valid": metadata["new_reference_valid"],
                    "maximum_linf_mw": metadata["maximum_linf_mw"],
                    "paper_percent_if_comparable": paper[method][3 if attack_split == "attack1" else 4] if method in paper else None,
                    "note": "Paper attack label convention is unknown; do not conflate fixed-target and re-optimized-target metrics.",
                })
            if profile == "source-informed":
                best.append({"method": method, "accuracy_by_regime": {
                    split: entry(profile, method, split, checkpoint="best_validation")["report"]["raw_accuracy"]
                    for split in SPLITS}})
    primary_q95 = [{"split": split, "report": entry(
        "source-informed", "M5", split, "hybrid-q95-rmse")["report"]} for split in SPLITS]
    groups = {key: 0. for key in ("dataset", "training", "decoder", "calibration", "attack_generation", "evaluation")}
    for name, stage in state["stages"].items():
        group = ("dataset" if name == "dataset" else "training" if name.endswith("--train") else
                 "decoder" if name.endswith("--decoder") else "calibration" if name.endswith("--q95-calibration") else
                 "attack_generation" if name.endswith("--dataset") else "evaluation")
        groups[group] += stage["seconds"]
    report = {
        "scope": "Descriptive frozen-suite analysis. No retraining or test-based profile/checkpoint/norm/threshold selection.",
        "root": str(root), "source_revision": manifest["source_revision"],
        "verified_output_hashes": verified, "stages": len(state["stages"]), "evaluations": len(entries),
        "wall_seconds": (datetime.fromisoformat(state["finished_utc"])-datetime.fromisoformat(state["started_utc"])).total_seconds(),
        "stage_seconds": groups, "dataset_configuration": data["configuration"],
        "capacity_regimes": data["capacity_regimes"], "installations": data["installations"],
        "reference_coverage": {split: {key: s.get(key) for key in (
            "requested", "valid", "target_valid_cases", "samples_with_active_branch", "sensitivity_available")}
            for split, s in data["splits"].items()},
        "fitting": fitting, "raw": raw, "paper_gaps": gaps, "hybrid_fixed6": hybrid,
        "primary_best_validation_raw": best, "primary_m5_q95": primary_q95, "attacks": attacks,
        "caveats": [
            "Capacity/site/weather/conversion and numerical choices are explicit hypotheses, not recovered author settings.",
            "All specifications are reported; source-informed was primary before execution.",
            "M3-example is not full external DeepOPF. No other paper systems or Germany were run.",
            "Changes in data and numerical conventions are bundled, not isolated causal ablations.",
            "Single training seed; cases/generator outputs are not independent training repetitions.",
            "Accuracy, decoder acceptance and strict feasibility remain distinct.",
            "No online adaptation or hidden optimization/projection of learned outputs.",
        ],
        "analysis_source_sha256": artifact_hash(__file__),
        "input_sha256": {name: artifact_hash(root/name) for name in ("suite_manifest.json", "status.json", "summary.json")},
    }
    lines = [
        "# IEEE-30 source-informed reconstruction: measured results", "",
        f"Frozen source `{report['source_revision']}`. **{report['stages']} stages, "
        f"{report['evaluations']} evaluations, {verified} required output hashes verified.**",
        f"Wall time: **{report['wall_seconds']/60:.2f} minutes**. Germany excluded.", "",
        "Primary variant was **source-informed** before execution; other variants are controls, not post-hoc replacements.",
        "These are approximate reconstruction hypotheses, not recovered author settings.", "",
        "## Reference coverage", "",
        _table(["Split", "Attempts", "Valid labels", "Active-branch scenarios"],
               [[split, s["requested"], s["valid"], s["samples_with_active_branch"]]
                for split, s in data["splits"].items()]), "",
        "Failed attempts are retained; valid-label resampling changes the evaluated conditional distribution.", "",
    ]
    for profile in PROFILES:
        rows = [r for r in raw if r["profile"] == profile]
        lines += [
            f"## {profile}: fixed-final raw predictions", "",
            _table(["Method", "IID <1 MW", "OOD1 <1 MW", "OOD2 <1 MW", "Pooled MAE MW", "Strict feasible"],
                   [[r["method"], *[_fmt(r["accuracy_by_regime"][s], percent=True) for s in SPLITS],
                     _fmt(r["pooled"]["mae_mw"]), _fmt(r["pooled"]["feasible_rate"], percent=True)] for r in rows]), "",
        ]
    lines += [
        "## Training and validation fit", "",
        _table(["Variant", "Method", "Training <1 MW", "Validation <1 MW", "Best epoch"],
               [[r["profile"], r["method"], _fmt(r["train"]["generator_accuracy_lt_1mw"], percent=True),
                 _fmt(r["validation"]["generator_accuracy_lt_1mw"], percent=True), r["best_epoch"]]
                for r in fitting]), "",
        "## Fixed 6 MW RMSE gate: M5", "",
        _table(["Variant", "Split", "Coverage", "Accepted source accuracy", "Hybrid source accuracy", "Hybrid feasible"],
               [[r["profile"], r["split"], _fmt(r["coverage"]["neural_coverage"], percent=True),
                 _fmt(r["accepted_accuracy"]["generator_accuracy_lt_1mw"], percent=True),
                 _fmt(r["hybrid_accuracy"]["generator_accuracy_lt_1mw"], percent=True),
                 _fmt(r["hybrid_physics"]["feasible_rate"], percent=True)]
                for r in hybrid if r["method"] == "M5"]), "",
        "All-model gates, physical risks, score-norm sensitivity and q95 results are in `analysis.json`.", "",
        "## FGSM: fixed targets versus freshly re-solved targets", "",
        _table(["Variant", "Method", "Attack", "Original-target <1 MW", "New-OPF-target <1 MW", "Valid new OPFs"],
               [[r["profile"], r["method"], r["split"],
                 _fmt(r["fixed_original_targets_all_attacks"]["generator_accuracy_lt_1mw"], percent=True),
                 _fmt(r["fresh_opf_targets_valid_attacks"]["generator_accuracy_lt_1mw"], percent=True),
                 f"{r['fresh_reference_valid']}/{r['attacks']}"] for r in attacks]), "",
        "## Interpretation limits", "",
        *[f"- {caveat}" for caveat in report["caveats"]], "",
        "No result in this report selects new training settings or alters earlier negative results.", "",
    ]
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"analysis.json", report)
    (output/"report.md").write_text("\n".join(lines))
    return {"output": str(output), "verified_output_hashes": verified,
            "stages": report["stages"], "evaluations": report["evaluations"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    print(json.dumps(analyze_suite(args.suite, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
