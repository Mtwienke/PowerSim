import json
from pathlib import Path

import pytest

from powersim.output_generation.geography import load_mapping, plot_map


EXAMPLE = Path(__file__).parents[1] / "examples/geography/synthetic_feeder.geojson"


def test_bundled_geography_mapping_is_valid() -> None:
    mapping = load_mapping(EXAMPLE)

    assert mapping["type"] == "FeatureCollection"
    assert mapping["metadata"]["crs"] == "EPSG:4326"
    assert mapping["features"]


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        ({"type": "Point", "coordinates": []}, "too few Point coordinates"),
        ({"type": "LineString", "coordinates": [[1.0, 2.0]]}, "too few LineString coordinates"),
    ],
)
def test_mapping_rejects_insufficient_geometry_coordinates(tmp_path, geometry, message) -> None:
    mapping = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    mapping["features"] = [{
        "type": "Feature",
        "properties": {"asset_type": "bus", "asset_id": "bus:test"},
        "geometry": geometry,
    }]
    path = tmp_path / "bad-mapping.geojson"
    path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_mapping(path)


def test_mapping_rejects_non_feature_collection(tmp_path) -> None:
    mapping = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    mapping["type"] = "Feature"
    path = tmp_path / "bad-mapping.geojson"
    path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(ValueError, match="FeatureCollection"):
        load_mapping(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("center_lat", "north"),
        ("center_lat", 91.0),
        ("center_lon", -181.0),
        ("center_lon", float("inf")),
        ("center_lat", True),
    ],
)
def test_mapping_rejects_invalid_optional_map_center(tmp_path, field, value) -> None:
    mapping = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    mapping["metadata"][field] = value
    path = tmp_path / "bad-center.geojson"
    path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_mapping(path)


def test_map_renders_mapped_assets_as_unknown_without_results() -> None:
    figure = plot_map(load_mapping(EXAMPLE), None)

    assert figure.data
    assert len(figure.data) == len(load_mapping(EXAMPLE)["features"])


def test_map_renders_evaluated_bus_voltage_and_line_loading() -> None:
    mapping = load_mapping(EXAMPLE)
    result = {
        "status": "evaluated",
        "bus_voltage_pu": {0: 1.05},
        "line_loading_percent": {0: 120.0},
    }

    figure = plot_map(mapping, result)

    bus_trace = next(trace for trace in figure.data if trace.name == "bus" and "bus:0" in trace.text[0])
    line_trace = next(trace for trace in figure.data if trace.name == "line" and "line:0" in trace.text[0])
    assert "1.0500 pu" in bus_trace.text[0]
    assert "0.0500 pu" in bus_trace.text[0]
    assert "120.0%" in line_trace.text[0]
    assert line_trace.line.color == "#c62828"
