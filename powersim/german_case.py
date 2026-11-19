"""Load the bundled German transmission equivalent (all buses on a 380 kV base)."""
import json
import math
from importlib.resources import files

import pandapower as pp


def germany_case_data():
    return json.loads(files("powersim").joinpath("benchmarks/germany30.json").read_text(encoding="utf-8"))


def make_germany30():
    data = germany_case_data()
    net = pp.create_empty_network(name="Germany-30 regional transmission equivalent", f_hz=50, sn_mva=100)
    voltage = data["base_voltage_kv"]
    for bus in data["buses"]:
        idx = pp.create_bus(net, voltage, index=bus["index"], name=bus["name"],
                            geodata=(bus["longitude"], bus["latitude"]), min_vm_pu=.9, max_vm_pu=1.1)
        pp.create_load(net, idx, bus["gross_load_mw"], name="Assumed regional demand", controllable=False)
        for kind in ("wind", "pv"):
            power = bus[f"{kind}_mw"]
            if power:
                pp.create_sgen(net, idx, power, name=f"Assumed regional {kind.upper()}",
                               type=kind, controllable=False)
    for line in data["corridors"]:
        pp.create_line_from_parameters(
            net, line["from_bus"], line["to_bus"], length_km=1, r_ohm_per_km=0,
            x_ohm_per_km=line["x_ohm_at_base"], c_nf_per_km=0,  # length=1 stores total equivalent X
            max_i_ka=line["rating_mva"] / (math.sqrt(3) * voltage),
            name=f"Regional corridor {line['from_bus']:02d}–{line['to_bus']:02d}", max_loading_percent=100.)
    for source in data["sources"]:
        bus, pmax = source["bus"], source["max_mw"]
        if source["is_slack"]:
            kind = "ext_grid"
            index = pp.create_ext_grid(net, bus, name="Domestic equivalent slack (not an import)",
                                       min_p_mw=0., max_p_mw=pmax, min_q_mvar=-pmax, max_q_mvar=pmax)
        else:
            kind = "gen"
            index = pp.create_gen(net, bus, p_mw=0., name="Assumed conventional generation", vm_pu=1.,
                                  min_p_mw=0., max_p_mw=pmax, min_q_mvar=-pmax, max_q_mvar=pmax)
        pp.create_poly_cost(net, index, kind, cp1_eur_per_mw=source["linear_cost"],
                            cp2_eur_per_mw2=source["quadratic_cost"])
    net["source_provenance"] = data["provenance"]
    return net
