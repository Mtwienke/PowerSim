import json

import numpy as np
import pytest

from powersim.cli import main
from powersim.config import SimulationConfig
from powersim.contracts import DispatchDecision
from powersim.data import DatasetConfig, generate_dataset
from powersim.dc import dc_relation
from powersim.instance_generation.german_case import germany_case_data
from powersim.network import make_case, nodal_loads
from powersim.solution_view import dataset_solution, solution_figure
from powersim.solver import evaluate_decision, solve_reference


def test_germany30_reference_and_provenance():
    data = germany_case_data()
    assert data["schema_version"] == 2
    assert data["provenance"]["license"] == "ODbL-1.0"
    assert len({node for bus in data["buses"] for node in bus["source_buses"]}) == 780
    region_sizes = [len(bus["source_buses"]) for bus in data["buses"]]
    assert min(region_sizes) >= 15 and max(region_sizes) <= 40
    assert data["provenance"]["region_bus_count_max"] == max(region_sizes)
    assert data["provenance"]["border_ac_lines_excluded"] == 42
    assert data["provenance"]["hvdc_capacity_mw_excluded"] == 3600.
    config = SimulationConfig(benchmark="germany30")
    net = make_case(config)
    result = solve_reference(net, config)
    assert len(net.bus) == 30 and len(result.branches) == 54
    assert result.status == "ok", result.diagnostics
    assert not any(branch.violated for branch in result.branches)
    assert max(branch.loading_percent for branch in result.branches) > 70.
    assert sum(nodal_loads(net)) == pytest.approx(63400.)
    assert len(result.generator_ids) == 18
    audit = evaluate_decision(net, DispatchDecision(active_power_mw=result.generation_mw), config)
    assert audit.status == "ok" and audit.metrics["feasible"]
    relation = dc_relation(net)
    modeled = relation.m_g @ list(result.generation_mw.values()) - relation.m_d @ nodal_loads(net) + relation.constant
    np.testing.assert_allclose(modeled, [branch.p_from_mw for branch in result.branches], atol=1e-6)
    with pytest.raises(ValueError, match="DC-only"):
        SimulationConfig("ac", "germany30")


def test_graph_has_branches_arrows_slack_and_distinct_unknowns():
    config = SimulationConfig(benchmark="case14")
    net = make_case(config)
    result = solve_reference(net, config).to_dict()
    # Construct explicit idle/reverse/unavailable cases without a second solve.
    reverse, idle, unavailable = result["branches"][:3]
    reverse.update(p_from_mw=-20, p_to_mw=20, violated=True)
    idle.update(p_from_mw=0, p_to_mw=0, violated=False)
    result["branches"] = [b for b in result["branches"] if b is not unavailable]
    result["proposed_generation_mw"] = {"ext_grid:0": 12.}
    result["slack_adjustment_mw"] = {"ext_grid:0": 14.}
    figure = solution_figure(net, result)
    by_id = {trace.meta["asset_id"]: trace for trace in figure.data if trace.meta}
    r_trace = by_id[reverse["asset_id"]]
    assert r_trace.name == "Over limit"
    assert by_id[idle["asset_id"]].name == "Idle"
    assert by_id[unavailable["asset_id"]].name == "Unavailable / out of service"
    assert {trace.line.dash for trace in by_id.values()} == {"solid", "dash"}
    arrow = figure.layout.annotations[0]
    assert (arrow.ax, arrow.ay) == pytest.approx(tuple(
        axis[1]+.18*(axis[2]-axis[1]) for axis in (r_trace.x, r_trace.y)))
    slack = next(trace for trace in figure.data if trace.name == "Slack / external grid")
    assert "proposed: 12.000 MW" in slack.hovertext[0]
    assert "slack adjustment: +14.000 MW" in slack.hovertext[0]
    assert "Voltage:" not in slack.hovertext[0]
    assert "schematic" in figure.layout.title.text
    with pytest.raises(ValueError, match="topology"):
        solution_figure(make_case(), result)


def test_replay_german_renewables_once_and_export_offline(tmp_path, capsys):
    config = DatasetConfig(simulation=SimulationConfig(benchmark="germany30"), train_cases=2,
                           validation_cases=1, test_cases=1)
    folder = tmp_path/"data"
    manifest = generate_dataset(folder, config)
    record = json.loads((folder/"train.jsonl").read_text().splitlines()[0])
    net, result, gross, fixed = dataset_solution(folder, "train", 0)
    assert sum(manifest["nominal_gross_load_mw"]) == pytest.approx(70000.)
    np.testing.assert_allclose(np.array(gross)-fixed, record["net_load_mw"], atol=1e-10)
    fig = solution_figure(net, result, demand=gross, renewables=fixed)
    assert "OpenStreetMap" in fig.layout.title.text
    filename = tmp_path/"germany.html"
    assert main(["plot-solution", str(filename), "--dataset", str(folder), "--split", "train",
                 "--sample", "0"]) == 0
    assert "ok:" in capsys.readouterr().out
    document = filename.read_text()
    assert "Plotly.newPlot" in document and "cdn.plot.ly" not in document[:500]
