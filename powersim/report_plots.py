"""Report tables, scientific summary and PDF plots for analyzed runs."""
from collections import defaultdict

import numpy as np


SPLITS = ("iid_test", "ood_test_1", "ood_test_2")
RECONSTRUCTION = "source-informed"


def _label(method):
    if method in ("M1", "M2", "M4", "M5"):
        return method + " analogue"
    return method


def _group(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return groups


def _primary(rows):
    return [r for r in rows if r["profile"] == RECONSTRUCTION and r.get("checkpoint") == "last"
            and r["evaluation"] == "raw" and r["scenario"] in SPLITS]


def _germany(rows):
    return [r for r in rows if r["network"] == "germany30-v1" and r["evaluation"] == "raw"]


def _format(value, scale=1., decimals=3):
    return "—" if value is None else f"{value*scale:.{decimals}f}"


def _display(value):
    return f"{100*value:.2f}%" if value is not None else "unavailable"


def _md(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |",
                      *["| " + " | ".join(map(str, r)) + " |" for r in rows]])


def _tex(output, name, headers, rows):
    def cell(value):
        return str(value).replace("—", "--").replace("_", r"\_").replace("%", r"\%")

    line_end = r" \\"
    text = [r"% Requires \usepackage{booktabs}.", r"\begin{tabular}{" + "l"*len(headers) + "}", r"\toprule",
            " & ".join(map(cell, headers))+line_end, r"\midrule",
            *[" & ".join(map(cell, row))+line_end for row in rows], r"\bottomrule", r"\end{tabular}", ""]
    (output / name).write_text("\n".join(text), encoding="utf-8")


def _counts(output, primary, germany, paper):
    indexed = {(r["method"], r["scenario"]): r for r in primary}
    rows = []
    for method in ("M1", "M2", "M3", "M4", "M5"):
        if method in paper:
            p = paper[method]
            rows.append([method, "Gao et al., Table II", *[_format(v, decimals=2) for v in p[:3]]])
        values = [indexed.get((method, s), {}).get("element_lt_1mw") for s in SPLITS]
        if any(v is not None for v in values) or method == "M3":
            rows.append([_label(method) if method != "M3" else "M3 / DeepOPF", "PowerSim reconstruction",
                         *[_format(v, 100, 2) for v in values]])
    if primary:
        rows.append(["M3-example", "PowerSim comparator (not DeepOPF)",
                     *[_format(indexed["M3-example", s]["element_lt_1mw"], 100, 2) if ("M3-example", s) in indexed
                       else "—" for s in SPLITS]])
    header = ["Method", "Source", "IID (%)", "OOD1 (%)", "OOD2 (%)"]
    _tex(output, "ieee_paper_comparison.tex", header, rows)
    header_g = ["Method", "Scenario", "Valid / attempts", "Element (%)", "Joint (%)", "MAE (MW)", "Feasible (%)"]
    germany_rows = [[r["method"], r["scenario"], f"{r['reference_valid']}/{r['requested']}",
                     _format(r["element_lt_1mw"], 100, 2), _format(r["max_lt_1mw"], 100, 2),
                     _format(r["mae_mw"]), _format(r["feasible_rate"], 100, 2)] for r in germany]
    if germany_rows:
        _tex(output, "germany30_v1_results.tex", header_g, germany_rows)
    return _md(header, rows), _md(header_g, germany_rows)


def _distributions(plt, output, rows, label, filename):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.25))
    groups = sorted(_group(rows, "method").items())
    for ax, scenario in zip(axes, SPLITS):
        for i, (_, group) in enumerate(groups):
            r = next((r for r in group if r["scenario"] == scenario), None)
            if r:
                with np.load(output / "matrices" / f"{r['experiment_id']}.npz", allow_pickle=False) as a:
                    error = a["absolute_error_mw"]
                data = error[np.isfinite(error).all(axis=1)].ravel()
                if data.size:
                    ax.boxplot(data, positions=[i], widths=.65, showfliers=False)
        ax.set_xticks(range(len(groups)), [_label(m) for m, _ in groups],
                      rotation=55, ha="right", fontsize=7)
        ax.set_title(scenario.replace("_test", ""))
        ax.set_yscale("symlog", linthresh=.001)
        ax.axhline(1, ls="--", color=".45", lw=.8)
    axes[0].set_ylabel("Absolute generator error (MW)")
    fig.suptitle(f"{label}: error distribution (outliers omitted from drawing)")
    _save(plt, fig, output / "figures", filename)


def _save(plt, fig, output, name):
    fig.tight_layout()
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def _penetration(plt, output, rows, title, filename, *, reference=()):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for method, group in sorted(_group(rows, "method").items()):
        group.sort(key=lambda r: r["nameplate_penetration"])
        x = [100*r["nameplate_penetration"] for r in group]
        for ax, metric in zip(axes, ("element_lt_1mw", "max_lt_1mw")):
            ax.plot(x, [100*r[metric] for r in group], marker="o", label=_label(method))
    for ax, label in zip(axes, ("Generator-wise", "Joint all-source")):
        ax.set(xlabel="Printed installed renewable penetration (%)", ylabel=f"{label} <1 MW (%)")
        ax.grid(alpha=.2)
    if reference:
        ax = axes[1].twinx()
        ax.plot([100*r["nameplate_penetration"] for r in reference], [100*r["reference_coverage"] for r in reference],
                marker="s", color="black", ls="--")
        ax.set_ylabel("Reference label coverage (%)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(title)
    _save(plt, fig, output, filename)


def _constraints(plt, output, rows, filename, title):
    rows = sorted((r for r in rows if r["scenario"] == "ood_test_2"), key=lambda r: r["method"])
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.75))
    x = np.arange(len(rows))
    for ax, metrics, name in (
        (axes[0], ["line_violation_events", "line_limit_plus_1mw_violation_events"], "Branch × case events"),
        (axes[1], ["line_mean_violation_mw", "line_limit_plus_1mw_mean_violation_mw"], "Mean excess among events (MW)"),
        (axes[2], ["balance_pass_rate", "generator_pass_rate", "line_pass_rate", "feasible_rate"], "Pass rate (%)"),
    ):
        width = .82/len(metrics)
        for j, metric in enumerate(metrics):
            y = [np.nan if r.get(metric) is None else r[metric]*(100 if metric.endswith("_rate") else 1) for r in rows]
            legend = ("Numerical tolerance" if j == 0 else "Limit + 1 MW") if len(metrics) == 2 else metric.replace("_pass_rate", "").replace("_rate", "")
            ax.bar(x + (j-(len(metrics)-1)/2)*width, y, width=width, label=legend)
        ax.set_xticks(x, [_label(r["method"]) for r in rows], rotation=55, ha="right", fontsize=7)
        ax.set_ylabel(name)
        ax.legend(fontsize=7)
    fig.suptitle(f"{title}: modeled constraints, highest renewable regime")
    _save(plt, fig, output, filename)


def _gate_figures(plt, output, rows, curves, filename, title):
    if not rows:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(4.15*len(rows), 3.4), squeeze=False)
    for ax, row in zip(axes[0], rows):
        key = row["experiment_id"]
        # Load only a few selected experiments without materializing the whole CSV.
        with (output / "matrices" / f"{key}.npz").open("rb") as f, np.load(f, allow_pickle=False) as a:
            scores, keep = a["reconstruction_error"], a["accepted"]
            e = a["raw_absolute_error_mw"].max(axis=1)
        for accepted, label in ((True, "Accepted"), (False, "Rejected")):
            ax.scatter(scores[keep == accepted], e[keep == accepted], s=5, alpha=.35, label=label, rasterized=False)
        ax.axhline(1, lw=.8, ls="--", color=".4")
        ax.axvline(row["threshold"], lw=.8, ls=":", color=".4")
        ax.set(xlabel=f"Decoder reconstruction ({row['score_units']})", ylabel="Raw max error (MW)",
               title=row["scenario"].replace("_test", ""), yscale="symlog")
        ax.legend(fontsize=7)
    fig.suptitle(title)
    _save(plt, fig, output / "figures", f"{filename}_scatter")
    fig, ax = plt.subplots(2, 4, figsize=(13.5, 6.8))
    for row in rows:
        points = sorted((r for r in curves if r["experiment_id"] == row["experiment_id"] and r["threshold"] is not None),
                        key=lambda r: r["threshold"])
        label = row["scenario"].replace("_test", "")
        xs = [r["threshold"] for r in points]
        coverages = [100*r["acceptance_rate"] for r in points]
        plans = (
            (ax[0, 0], xs, "acceptance_rate", "Accepted (%)"),
            (ax[0, 1], xs, "accuracy_false_acceptance_rate", "Inaccurate cases accepted (%)"),
            (ax[0, 2], xs, "estimated_total_worker_decision_seconds", "Estimated summed worker seconds"),
            (ax[0, 3], xs, "feasibility_false_acceptance_rate", "Infeasible cases accepted (%)"),
            (ax[1, 0], coverages, "accepted_max_lt_1mw", "Accepted joint accuracy (%)"),
            (ax[1, 1], coverages, "accepted_feasible_rate", "Accepted feasible (%)"),
            (ax[1, 2], xs, "accepted_max_lt_1mw", "Accepted joint accuracy (%)"),
            (ax[1, 3], xs, "accepted_feasible_rate", "Accepted feasible (%)"),
        )
        for axis, x, field, ylabel in plans:
            scale = 1 if field.endswith("seconds") else 100
            axis.plot(x, [np.nan if r[field] is None else scale*r[field] for r in points], marker=".", label=label)
            axis.set(xlabel="Accepted (%)" if x is coverages else f"Threshold ({row['score_units']})", ylabel=ylabel)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(title + " — descriptive validation-grid sweep")
    _save(plt, fig, output / "figures", f"{filename}_curves")


def _runtime_table(output, runtime):
    groups = _group([r for r in runtime if r["measurement"] == "controlled_fresh_cpu"], "network")
    components = ("numerical_opf", "neural_single", "decoder_score_with_features", "hybrid",
                  "neural_batch", "data_load_preprocess", "hybrid_including_load")
    headers = ["Network", "Component", "Cases/repeat", "Median total (s)", "Median p95 (ms)", "Median speedup"]
    rows = []
    for network, group in groups.items():
        by_component = _group(group, "component")
        for component in components:
            if component not in by_component:
                continue
            stats = by_component[component]
            def med(field):
                values = [r[field] for r in stats if r.get(field) is not None]
                return float(np.median(values)) if values else None
            rows.append([network, component, stats[0]["samples"], _format(med("total_seconds")),
                         _format(med("p95_latency_seconds"), 1000), _format(med("speedup_vs_numerical"), decimals=2)])
    if rows:
        _tex(output, "runtime_summary.tex", headers, rows)
    return groups, _md(headers, rows) if rows else "Controlled timing was not run."


def _runtime_plot(plt, output, groups):
    fig, axes = plt.subplots(1, len(groups), figsize=(4.9*len(groups), 3.6), squeeze=False)
    for ax, (name, group) in zip(axes[0], groups.items()):
        valid = _group([r for r in group if r["component"] in ("numerical_opf", "neural_single", "hybrid")], "component")
        for i, (component, rows) in enumerate(valid.items()):
            t = [r["total_seconds"] for r in rows]
            ax.scatter([i]*len(t), t, s=16, color="tab:blue", zorder=3)
            ax.bar(i, np.median(t), color="tab:blue", alpha=.32)
        names = {"numerical_opf": "Numerical OPF", "neural_single": "Neural (batch 1)", "hybrid": "Hybrid"}
        ax.set_xticks(range(len(valid)), [names[k] for k in valid], rotation=20)
        ax.set(ylabel="Decision wall time (s, log)", yscale="log",
               title=f"{name}: {group[0]['samples']:,} matched inputs")
    _save(plt, fig, output, "runtime_comparison")


def _findings(output, metrics, primary, paper_table, germany_table, runtime_table, manifest):
    lines = [
        "# PowerSim DCOPF — evaluation findings", "",
        "This is a descriptive re-analysis of frozen results, not an exact replication. No model, physics data or "
        "threshold was tuned on the analyzed test outcomes.", "",
        "## IEEE-30 versus Gao et al.", "",
        "Published values: Gao et al., [DOI 10.1109/TNNLS.2024.3437741](https://doi.org/10.1109/TNNLS.2024.3437741), "
        "p. 8055, Table II. PowerSim values are strict generator-wise fractions among validated references. The "
        "paper's aggregation is inferred, not established. Equal printed penetration levels do not imply equal tasks.", "",
        paper_table, "",
    ]
    primary_i = {(r["method"], r["scenario"]): r for r in primary}
    if ("M5", "ood_test_2") in primary_i:
        high, iid = primary_i["M5", "ood_test_2"], primary_i["M5", "iid_test"]
        m1, m4 = (primary_i[m, "ood_test_2"] for m in ("M1", "M4"))
        gap = lambda a, b: f"{100*(a['element_lt_1mw']-b['element_lt_1mw']):+.2f} pp"
        gated = [r for r in metrics if r["profile"] == RECONSTRUCTION and r["method"] == "M5"
                 and r["scenario"] == "ood_test_2" and r["evaluation"].startswith("hybrid")]
        lines += [
            f"**Observation.** Raw M5-analogue accuracy changes by {gap(high, iid)} from IID to the highest OOD regime. "
            f"At that regime it is {gap(high, m1)} against M1 and {gap(high, m4)} against M4, "
            f"rather than outperforming both. M4 versus M1 is {gap(m4, m1)} (qualitatively agrees with Table II); "
            "numerical closeness of one entry does not establish protocol fidelity.",
            "",
            f"For the same raw M5 outputs, strict generator-wise, all-source and mean-error accuracies are "
            f"**{_display(high['element_lt_1mw'])}**, **{_display(high['max_lt_1mw'])}** and "
            f"**{_display(high['mean_lt_1mw'])}**, respectively. Strict DC feasibility is "
            f"**{_display(high['feasible_rate'])}**. {high['reference_valid']}/{high['requested']} attempts produced "
            "validated OPF labels; conditional metrics cannot be applied to failed attempts.",
            "",
            f"The highest regime has {high['line_violation_events']} branch–sample events beyond numerical tolerance "
            f"and {high['line_limit_plus_1mw_violation_events']} beyond modeled limit + 1 MW. These are not distinct "
            "lines, and neither count is a controlled replication of Fig. 6. Low balance-derivative loss "
            "does not mathematically enforce zero absolute imbalance.",
            "",
            "**Hypotheses, not isolated effects.** Differences in installed capacity, renewable placement/curves, "
            "limit scaling, solver-success conditioning, input normalization, learning schedule and loss reductions can "
            "change both test difficulty and fitting. The profiles are controlled variants, not independent seeds; "
            "the older pilot is not a paired improvement comparison with the primary run.", "",
            "## Adaptability and fallback", "",
        ]
        if gated:
            headers = ["Gate", "Accepted", "Accepted joint <1 MW", "Accepted feasible", "Fallback calls",
                       "Hybrid joint <1 MW", "Hybrid feasible"]
            gate_rows = [[g["evaluation"], _display(g["acceptance_rate"]), _display(g["accepted_max_lt_1mw"]),
                          _display(g["accepted_feasible_rate"]), g["fallback_calls"], _display(g["max_lt_1mw"]),
                          _display(g["feasible_rate"])] for g in gated]
            _tex(output, "ieee30_gate_summary.tex", headers, gate_rows)
            lines += [_md(headers, gate_rows), ""]
            q95 = next((r for r in gated if r["evaluation"] == "hybrid-q95-rmse"), None)
            if q95 is not None:
                lines += [
                    f"The validation-q95 score yields accuracy-risk AUROC **{_format(q95['accuracy_auroc'])}** and "
                    f"average precision **{_format(q95['accuracy_auprc_ap'])}** at "
                    f"{_display(q95['accuracy_ranking_prevalence'])} inaccurate-case prevalence. It accepts "
                    f"{_display(q95['accuracy_false_acceptance_rate'])} of inaccurate cases and rejects "
                    f"{_display(q95['accuracy_false_rejection_rate'])} of accurate cases. Feasibility discrimination "
                    "has no AUROC when all raw outputs fail the strict constraints.", "",
                ]
        lines += ["Coverage describes selection, not a completed workload. `metrics.csv` distinguishes both notions "
                  "of unreliable and includes their class prevalence. At these gates hybrid gains come from actual "
                  "fresh OPF replacements, not retraining or projection. A reconstruction error is not a safety "
                  "certificate, and the 6 and 7 MW numbers refer to distinct paper experiments and an unspecified norm.", ""]
        attacks = [r for r in metrics if r["profile"] == RECONSTRUCTION and r["method"] == "M5"
                   and r["evaluation"] in ("fgsm_fixed_original_target", "raw-fgsm")]
        if attacks:
            lines += ["## FGSM diagnostics", "",
                      _md(["Scenario", "Target", "Generator-wise <1 MW", "MAE (MW)", "Fresh OPF labels"],
                          [[r["scenario"], r["accuracy_reference"], _display(r["element_lt_1mw"]),
                            _format(r["mae_mw"]), f"{r['reference_valid']}/{r['requested']}"] for r in attacks]), "",
                      "The perturbation is bounded by 1 MW **per input coordinate**, not 1 MW total. Good performance "
                      "against fresh OPF labels does not show robustness against corrupted measurements at unchanged "
                      "load. Low original-target accuracy does not imply inability to solve changed physical loads. "
                      "The author's exact target convention is unresolved, so neither is a verified paper attack reproduction.", ""]
    de_v2 = [r for r in metrics if r["network"] == "germany30-v2" and r["evaluation"] == "reference"]
    lines += ["## Germany-30", "",
              "Topology source: Xiong et al., PyPSA-Eur prebuilt network v0.7, [Zenodo](https://doi.org/10.5281/zenodo.18619025); "
              "ODbL 1.0, © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright). "
              "Ratings/reactances and synthetic operating assumptions do not certify feasibility of the actual German grid.", ""]
    if _germany(metrics):
        de = [r for r in _germany(metrics) if r["method"] in ("baseline", "physics-guided") and r["scenario"] == "ood_test_2"]
        lines += ["**Historical v1.** At the highest penetration, reference coverage falls and none of the learned "
                  "methods passes the joint 1 MW or strict feasibility criteria in the observed v1 tests. Selected raw "
                  "results (complete ablations in `germany30_v1_results.tex`):", "",
                  _md(["Method", "Coverage", "Generator-wise <1 MW", "Joint <1 MW", "Feasible", "MAE (MW)"],
                      [[r["method"], _display(r["reference_coverage"]), _display(r["element_lt_1mw"]),
                        _display(r["max_lt_1mw"]), _display(r["feasible_rate"]), _format(r["mae_mw"])] for r in de]), ""]
    if de_v2:
        lines += ["**Current v2: numerical validation only.** The revised network has a different topology and "
                  "operating point. This separate cohort tests reference functionality, not v1-to-v2 "
                  "neural generalization or a causal improvement from any single network change.", "",
                  _md(["Scenario", "Validated / attempted", "Active-branch cases", "Mean realized renewable share"],
                      [[r["scenario"], f"{r['reference_valid']}/{r['requested']}", r["reference_samples_with_active_branch"],
                        _display(r["realized_penetration_mean_reference_valid"])] for r in de_v2]), ""]
    lines += ["## Runtime", "", runtime_table, "",
              "Fresh measurements cover identical reference-valid inputs, on CPU. This is not a comparison to "
              "the paper's hardware, a failed-reference workload, or an audit-and-write production pipeline. "
              "Archived worker totals in `runtime.csv` are not used as speedup denominators.", "",
              "## Suggested report framing", "",
              "- **Experimental Evaluation:** Present the IEEE table with provenance, then report the strict joint "
              "metric and constraint results alongside generator-wise accuracy. Use a separate gate table for "
              "coverage, accepted quality, fallback calls and complete hybrid quality.",
              "- **Critical Discussion:** The M4-vs-M1 ordering at highest OOD agrees qualitatively but the claimed "
              "M5 advantage does not. Separate observed numerical results from hypotheses about data, objectives and "
              "optimization. Do not rank systems of different size/units from their 1 MW accuracy alone.",
              "- Treat v1 and v2 as distinct benchmarks. Do not present conditional-on-valid reference coverage or "
              "case-level uncertainty as independent seed replication; the two historical protocol seeds vary data "
              "and training jointly. No test-based threshold or checkpoint selection is warranted.", "",
              "### Unavailable evidence", ""]
    lines.extend(f"- **{a['item']}:** {a['reason']}" for a in manifest["unavailable"])
    (output / "report_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clean_report(output, metrics, curves, runtime, manifest, figures):
    lines = ["# Clean-slate DCOPF report", "",
             "These runs did not reuse checkpoints or labels from the archived PowerSim suites. "
             "Gao comparisons are descriptive: the original generator, scoring convention and complete DeepOPF baseline are unavailable.",
             "", "Germany-30 uses the v2 modeled network, derived from Xiong et al., "
             "[PyPSA-Eur v0.7](https://doi.org/10.5281/zenodo.18619025), ODbL 1.0; "
             "© [OpenStreetMap contributors](https://www.openstreetmap.org/copyright). It is not an operational grid model.", ""]
    headers = ["Grid", "Method", "Source", "IID (%)", "OOD1 (%)", "OOD2 (%)"]
    timing_groups, timing_table = _runtime_table(output, runtime)
    figure_groups = []
    for run, plan in manifest["clean_suites"].items():
        lines += [f"## {run}", ""]
        if plan["configuration"]["smoke"]:
            lines += ["**Smoke only: these tiny workloads are not scientific evidence.**", ""]
        output_rows = []
        raw = [r for r in metrics if r["run"] == run and r["evaluation"] == "raw"]
        published = plan["paper_table_iii"]
        for benchmark in plan["networks"]:
            network = "germany30-v2" if benchmark == "germany30" else benchmark
            cases = [r for r in raw if r["network"] == network]
            p = (dict(zip(published["methods"], published["networks"][benchmark]["accuracy_pct"], strict=True))
                 if benchmark in published["networks"] else plan["paper_table_ii"] if benchmark == "case30" else {})
            for method in ("M1", "M2", "M3", "M3-example", "M4", "M5"):
                if method in p:
                    output_rows.append([network, method, "Gao II" if benchmark == "case30" else "Gao III",
                                        *[_format(v, decimals=2) for v in p[method][:3]]])
                if method != "M3" and any(r["method"] == method for r in cases):
                    values = [[100*r["element_lt_1mw"] for r in cases if r["method"] == method and r["scenario"] == split]
                              for split in SPLITS]
                    output_rows.append([network, _label(method), "PowerSim seed mean",
                                        *[_format(float(np.mean(v)), decimals=2) if v else "—" for v in values]])
            plot_rows = [r for r in cases if r["seed"] == plan["seeds"][0] and r["scenario"] != "ood_demand"]
            plot_gate = [r for r in metrics if r["run"] == run and r["network"] == network and r["method"] == "M5"
                         and r["evaluation"] == "hybrid-q95-rmse" and r["seed"] == plan["seeds"][0]
                         and r["scenario"] in SPLITS]
            if plot_rows:
                figure_groups.append((run.split(":", 1)[-1], network, plot_rows, plot_gate))
        lines += ["Generator-wise strict `<1 MW`; the paper’s vector reduction is inferred, not established.", "",
                  _md(headers, output_rows), "",
                  "Demand-only results and the additional Table V stress rates are in `metrics.csv`; "
                  "independent-seed mean, SD and t intervals are in `training_seed_summary.csv`.", ""]
        _tex(output, f"{run.split(':', 1)[-1]}_paper_comparison.tex", headers, output_rows)
    lines += ["## Runtime", "", timing_table, "", "## Unavailable evidence", ""]
    lines.extend(f"- **{item['item']}:** {item['reason']}" for item in manifest["unavailable"])
    (output/"report_findings.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
        dest = output/"figures"
        dest.mkdir()
        for run, network, raw, gated in figure_groups:
            stem = f"{run}_{network}"
            title = f"{network}: first declared seed ({raw[0]['seed']})"
            _penetration(plt, dest, raw, title, f"{stem}_generalization")
            _constraints(plt, dest, raw, f"{stem}_constraints", title)
            _gate_figures(plt, output, gated, curves, f"{stem}_gate", title + " — M5 analogue")
        if timing_groups:
            _runtime_plot(plt, dest, timing_groups)


def render(output, metrics, curves, runtime, manifest, *, figures=True):
    if manifest.get("clean_suites"):
        _clean_report(output, metrics, curves, runtime, manifest, figures)
        return
    primary, german = _primary(metrics), _germany(metrics)
    paper_table, german_table = _counts(output, primary, german, manifest["paper_table_ii"])
    timing_groups, timing_table = _runtime_table(output, runtime)
    _findings(output, metrics, primary, paper_table, german_table, timing_table, manifest)
    if not figures:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9})
    figures_path = output / "figures"
    figures_path.mkdir()
    for rows, title, stem in ((primary, "IEEE-30 reconstruction (source-informed)", "ieee30"),
                              (german, "Germany-30 v1 (historical modeled network)", "germany30_v1")):
        if rows:
            _penetration(plt, figures_path, rows, title, f"{stem}_generalization")
            _distributions(plt, output, rows, title, f"{stem}_error_distribution")
            _constraints(plt, figures_path, rows, f"{stem}_constraint_violations", title)
    for spec, title in (("hybrid-q95-rmse", "IEEE-30 M5 analogue, validation q95 RMSE"),
                        ("hybrid", "Germany-30 v1 guided model, validation q95 normalized MSE")):
        gated = [r for r in metrics if r["evaluation"] == spec and (spec == "hybrid-q95-rmse"
                 or (r["network"] == "germany30-v1" and r["method"] == "physics-guided"))]
        if gated:
            _gate_figures(plt, output, gated, curves, "ieee30_gate" if spec != "hybrid" else "germany30_v1_gate", title)
    if timing_groups:
        _runtime_plot(plt, figures_path, timing_groups)
