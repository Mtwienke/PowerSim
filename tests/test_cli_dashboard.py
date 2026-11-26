import gzip
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from powersim.cli import main
from powersim.contracts import DispatchDecision
from powersim.data import DatasetConfig, generate_dataset


def test_reference_cli(capsys):
    assert main(["reference"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["physics_mode"] == "dc" and result["operation"] == "opf"


def test_reference_cli_can_reuse_a_saved_instance_with_another_solver(tmp_path, monkeypatch, capsys):
    directory = tmp_path / "dataset"
    generate_dataset(directory, DatasetConfig(train_cases=2, validation_cases=1, test_cases=1,
                                               record_format="jsonl.gz"))
    with gzip.open(directory / "train.jsonl.gz", "rt") as stream:
        record = json.loads(next(stream))
    original = record["reference"]
    assert original["status"] == "ok"

    def dispatch(instance):
        assert instance.net_load_mw == pytest.approx(record["net_load_mw"])
        assert instance.config.branch_limit_scale == .9
        return DispatchDecision(active_power_mw=original["generation_mw"])

    plugin = ModuleType("custom_power_solver")
    plugin.dispatch = dispatch
    monkeypatch.setitem(sys.modules, plugin.__name__, plugin)
    saved = ["reference", "--dataset", str(directory), "--split", "train", "--sample", "0"]
    assert main(saved) == 0
    builtin = json.loads(capsys.readouterr().out)
    assert main([*saved, "--solver", "custom_power_solver:dispatch"]) == 0
    alternative = json.loads(capsys.readouterr().out)
    assert alternative.keys() == builtin.keys() == original.keys()
    for key in ("model_id", "configuration_id", "input_id"):
        assert alternative[key] == builtin[key] == original[key]
    assert alternative["objective"] == pytest.approx(original["objective"])


def test_basic_cli_never_imports_ml_stack():
    result = subprocess.run([sys.executable, "-c",
                             "import powersim.cli, sys; assert 'torch' not in sys.modules; assert 'streamlit' not in sys.modules"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_dashboard_defaults_and_dc_results():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=15).run()
    assert not app.exception
    assert app.sidebar.selectbox(key="physics").value == "dc"
    assert app.sidebar.selectbox(key="benchmark").value == "case30"
    assert app.sidebar.selectbox[2].value == "numerical"
    assert app.sidebar.selectbox[3].disabled
    app.button[0].click().run()
    assert not app.exception and not app.warning
    assert app.session_state["result"]["bus_voltage_pu"] is None
    app.sidebar.selectbox(key="physics").select("ac").run()
    assert not app.exception
    app.button[0].click().run()
    assert app.session_state["result"]["physics_mode"] == "ac"
    assert app.session_state["result"]["bus_voltage_pu"] is not None
    app.sidebar.selectbox(key="benchmark").select("germany30").run()
    assert app.sidebar.selectbox(key="physics").options == ["dc"]
    assert "result" not in app.session_state
    app.button[0].click().run()
    assert not app.exception and not app.warning
    assert len(app.get("plotly_chart")) >= 2  # solution graph and separate demo map
    assert app.session_state["result"]["physics_mode"] == "dc"
    load = next(widget for widget in app.sidebar.number_input if widget.label == "Load (MW)")
    load.set_value(load.value + 1).run()
    assert "result" not in app.session_state
