import streamlit as st
import plotly.express as px
from powersim.simulator import make_case, evaluate
from controllers.my_dispatch import MyDispatchPolicy
from powersim.geography import load_mapping, plot_map
from pathlib import Path
from powersim.config import SimulationConfig
from powersim.solver import solve_reference
from powersim.network import build_dispatch_inputs, nodal_loads
from copy import deepcopy
from dataclasses import asdict

st.set_page_config(page_title="PowerSim Research", layout="wide")
st.title("PowerSim · reproducible OPF research")
st.caption("DC paper-inspired experiments; optional AC reference/evaluation. Research use only.")
st.info("Simulation outputs are synthetic research results, not live utility measurements. Geographic proximity does not indicate service area or outage risk.")
st.sidebar.header("Case and controls")
physics = st.sidebar.selectbox("Physics mode", ["dc", "ac"])
benchmark = st.sidebar.selectbox("Benchmark", ["case30", "case14"])
limit_scale = st.sidebar.number_input("Branch limit scale", min_value=.01, max_value=10.0, value=1.0)
model_path = st.sidebar.text_input("Trained DC model path", value="")
decoder_path = st.sidebar.text_input("Calibrated decoder path", value="")
base_config = SimulationConfig(physics, benchmark, branch_limit_scale=limit_scale)
net = make_case(base_config)
learned = None
learned_message = "Unavailable — provide trained model + calibrated decoder"
if physics == "dc" and model_path and decoder_path:
    try:
        from powersim.ml import load_model, load_decoder
        from powersim.hybrid import HybridPredictor
        model, checkpoint = load_model(model_path)
        decoder, metadata = load_decoder(decoder_path, model_path, model)
        context = build_dispatch_inputs(net, base_config)["context"]
        if (checkpoint["model_id"] != context.network_fingerprint or
                tuple(checkpoint["bus_ids"]) != context.bus_ids or
                tuple(checkpoint["generator_ids"]) != context.controllable_assets):
            raise ValueError("Artifact network/limits mismatch; use its benchmark and branch limit scale")
        learned = HybridPredictor(model, decoder, metadata, checkpoint["generator_ids"])
        learned_message = "Ready — calibrated DC hybrid artifacts loaded"
    except Exception as exc:
        st.sidebar.warning(f"Learned inference unavailable: {exc}")
elif physics == "ac":
    learned_message = "Unavailable — DC-trained method is not an AC method"
method = st.sidebar.selectbox("Solution method", ["numerical", "heuristic"] + (["learned"] if learned else []))
st.sidebar.selectbox("Learned inference", [learned_message], disabled=True)
st.sidebar.caption("Train/calibrate using the CLI. Match the source benchmark and limits. Decoder acceptance is not a feasibility certificate.")
config = SimulationConfig(physics, benchmark, method, limit_scale)
if len(net.gen):
    idx = st.sidebar.selectbox("Generator", list(net.gen.index))
    net.gen.at[idx, "p_mw"] = st.sidebar.number_input("Generator setpoint (MW)", value=float(net.gen.at[idx,"p_mw"]))
if len(net.load):
    idx = st.sidebar.selectbox("Load", list(net.load.index))
    net.load.at[idx,"p_mw"] = st.sidebar.number_input("Load (MW)", value=float(net.load.at[idx,"p_mw"]))
if st.button("Solve reference OPF" if method == "numerical" else f"Evaluate {method}", type="primary"):
    if method == "numerical":
        result = solve_reference(net, config).to_dict()
    elif method == "heuristic":
        result = evaluate(net, MyDispatchPolicy(), config=config)
    else:
        from powersim.contracts import DispatchDecision
        from powersim.solver import evaluate_decision
        hybrid = learned.predict(nodal_loads(net), lambda: solve_reference(deepcopy(net), config))
        result = {**asdict(hybrid), "physics_mode": "dc", "bus_voltage_pu": None}
        for label, values in (("raw_evaluation", hybrid.raw_prediction_mw), ("chosen_evaluation", hybrid.generation_mw)):
            if values is not None:
                decision = DispatchDecision(active_power_mw=dict(zip(learned.generator_ids, values)))
                result[label] = evaluate_decision(deepcopy(net), decision, config).to_dict()
    st.session_state["result"] = result
    st.session_state["result_config"] = (physics, benchmark, method, limit_scale, model_path, decoder_path)
if st.session_state.get("result_config") != (physics, benchmark, method, limit_scale, model_path, decoder_path):
    st.session_state.pop("result", None)
if "result" in st.session_state:
    result = st.session_state["result"]
    st.json(result)
    if result.get("bus_voltage_pu") is not None:
        df = __import__("pandas").DataFrame({"bus":list(result["bus_voltage_pu"]),"voltage_pu":list(result["bus_voltage_pu"].values())})
        st.plotly_chart(px.bar(df, x="bus", y="voltage_pu", color="voltage_pu", title="Bus voltage magnitude (pu)"), use_container_width=True)

st.subheader("Optional geographic view")
st.caption("Separate mapping layer. The demonstration coordinates are synthetic and illustrative. OpenStreetMap basemap requires internet; attribution: © OpenStreetMap contributors. Mapping data license/provenance is shown below.")
geo_path = Path(__file__).parent / "examples/geography/synthetic_feeder.geojson"
try:
    geo = load_mapping(geo_path)
    st.caption(f"Geometry: {geo['metadata']['geometry_kind']} · Provenance: {geo['metadata']['provenance']} · License: {geo['metadata']['license']}")
    metric = st.selectbox("Map metric", ["voltage deviation", "voltage magnitude"])
    st.plotly_chart(plot_map(geo, st.session_state.get("result"), metric), use_container_width=True)
except Exception as exc:
    st.warning(f"Geographic view unavailable: {exc}. Electrical simulation remains usable.")
st.caption("No-basemap fallback: asset locations and electrical values remain available in the result JSON and voltage/branch charts above.")
