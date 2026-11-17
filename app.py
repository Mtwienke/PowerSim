import streamlit as st
import plotly.express as px
from powersim.simulator import make_case, evaluate
from controllers.my_dispatch import MyDispatchPolicy
from powersim.geography import load_mapping, plot_map
from pathlib import Path
from powersim.config import SimulationConfig
from powersim.solver import solve_reference

st.set_page_config(page_title="PowerSim Research", layout="wide")
st.title("PowerSim · reproducible OPF research")
st.caption("DC paper-inspired experiments; optional AC reference/evaluation. Research use only.")
st.info("Simulation outputs are synthetic research results, not live utility measurements. Geographic proximity does not indicate service area or outage risk.")
st.sidebar.header("Case and controls")
physics = st.sidebar.selectbox("Physics mode", ["dc", "ac"])
benchmark = st.sidebar.selectbox("Benchmark", ["case30", "case14"])
method = st.sidebar.selectbox("Solution method", ["numerical", "heuristic"])
st.sidebar.selectbox("Learned inference", ["Unavailable — train and evaluate via CLI"], disabled=True)
st.sidebar.caption("Learned inference is DC-only and disabled here until a trained artifact is explicitly integrated.")
config = SimulationConfig(physics, benchmark, method)
net = make_case(config)
if len(net.gen):
    idx = st.sidebar.selectbox("Generator", list(net.gen.index))
    net.gen.at[idx, "p_mw"] = st.sidebar.number_input("Generator setpoint (MW)", value=float(net.gen.at[idx,"p_mw"]))
if len(net.load):
    idx = st.sidebar.selectbox("Load", list(net.load.index))
    net.load.at[idx,"p_mw"] = st.sidebar.number_input("Load (MW)", value=float(net.load.at[idx,"p_mw"]))
if st.button("Solve reference OPF" if method == "numerical" else "Evaluate heuristic with PF", type="primary"):
    result = solve_reference(net, config).to_dict() if method == "numerical" else evaluate(net, MyDispatchPolicy(), config=config)
    st.session_state["result"] = result
    st.session_state["result_config"] = (physics, benchmark, method)
if st.session_state.get("result_config") != (physics, benchmark, method):
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
