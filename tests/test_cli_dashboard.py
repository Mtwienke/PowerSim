import json
import subprocess
import sys
from pathlib import Path

from powersim.cli import main


def test_reference_cli(capsys):
    assert main(["reference"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["physics_mode"] == "dc" and result["operation"] == "opf"


def test_basic_cli_never_imports_ml_stack():
    result = subprocess.run([sys.executable, "-c",
                             "import powersim.cli, sys; assert 'torch' not in sys.modules; assert 'streamlit' not in sys.modules"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_dashboard_defaults_and_dc_results():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=15).run()
    assert not app.exception
    assert app.sidebar.selectbox[0].value == "dc"
    assert app.sidebar.selectbox[1].value == "case30"
    assert app.sidebar.selectbox[2].value == "numerical"
    assert app.sidebar.selectbox[3].disabled
    app.button[0].click().run()
    assert not app.exception and not app.warning
    assert app.session_state["result"]["bus_voltage_pu"] is None
    app.sidebar.selectbox[0].select("ac").run()
    assert not app.exception
    app.button[0].click().run()
    assert app.session_state["result"]["physics_mode"] == "ac"
    assert app.session_state["result"]["bus_voltage_pu"] is not None
