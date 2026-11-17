"""Optional GeoJSON mapping layer. Never participates in electrical simulation."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

if TYPE_CHECKING:
    import plotly.graph_objects as go


class GeoMetadata(TypedDict):
    schema_version: str
    crs: str
    geometry_kind: str
    provenance: str
    license: str
    center_lat: NotRequired[float]
    center_lon: NotRequired[float]


class GeoProperties(TypedDict):
    asset_type: str
    asset_id: str
    electrical_index: NotRequired[int]
    limit: NotRequired[str]


class GeoPointGeometry(TypedDict):
    type: Literal["Point"]
    coordinates: list[float]


class GeoLineStringGeometry(TypedDict):
    type: Literal["LineString"]
    coordinates: list[list[float]]


GeoGeometry = GeoPointGeometry | GeoLineStringGeometry
MapMetric = Literal["voltage deviation", "voltage magnitude"]


class GeoFeature(TypedDict):
    type: Literal["Feature"]
    properties: GeoProperties
    geometry: GeoGeometry


class GeoMapping(TypedDict):
    type: Literal["FeatureCollection"]
    metadata: GeoMetadata
    features: list[GeoFeature]


class MapResult(TypedDict, total=False):
    status: str
    bus_voltage_pu: dict[int, float]
    line_loading_percent: dict[int, float]

SUPPORTED = frozenset({"bus", "line", "trafo", "load", "gen"})


def _validate_metadata(metadata: object) -> GeoMetadata:
    if not isinstance(metadata, dict):
        raise ValueError("Mapping metadata must be an object")
    meta = metadata
    if meta.get("schema_version") != "1.0" or meta.get("crs") != "EPSG:4326":
        raise ValueError("Mapping requires schema_version 1.0 and explicit EPSG:4326 coordinates")
    for name in ("provenance", "geometry_kind", "license"):
        _validate_metadata_text(meta, name)
    for name, minimum, maximum in (("center_lat", -90, 90), ("center_lon", -180, 180)):
        _validate_optional_coordinate(meta, name, minimum, maximum)
    return meta


def _validate_metadata_text(metadata: dict[str, Any], name: str) -> None:
    value = metadata.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Mapping metadata {name} must be a non-empty string")


def _validate_optional_coordinate(
    metadata: dict[str, Any], name: str, minimum: int, maximum: int
) -> None:
    if name not in metadata:
        return
    value = metadata[name]
    valid_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not valid_number or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"Mapping metadata {name} must be a finite EPSG:4326 coordinate")


def _validate_feature_properties(properties: object, feature_index: int) -> None:
    if not isinstance(properties, dict):
        raise ValueError(f"Feature {feature_index} properties must be an object")
    asset_type = properties.get("asset_type")
    if (
        not isinstance(asset_type, str)
        or asset_type not in SUPPORTED
        or not isinstance(properties.get("asset_id"), str)
        or not properties["asset_id"].strip()
    ):
        raise ValueError("Each feature needs a supported asset_type and stable asset_id")
    index = properties.get("electrical_index")
    if index is not None and (isinstance(index, bool) or not isinstance(index, int) or index < 0):
        raise ValueError(f"Feature {feature_index} electrical_index must be a non-negative integer")
    limit = properties.get("limit")
    if limit is not None and not isinstance(limit, str):
        raise ValueError(f"Feature {feature_index} limit must be a string when supplied")


def _validate_coordinates(geometry_type: str, coordinates: object, feature_index: int) -> None:
    if not isinstance(coordinates, list):
        raise ValueError(f"Feature {feature_index} coordinates must be an array")
    if geometry_type == "Point" and len(coordinates) < 2:
        raise ValueError(f"Feature {feature_index} has too few Point coordinates")
    positions = [coordinates] if geometry_type == "Point" else coordinates
    if geometry_type == "LineString" and len(positions) < 2:
        raise ValueError(f"Feature {feature_index} has too few LineString coordinates")
    for position in positions:
        _validate_position(position, feature_index)


def _validate_position(position: object, feature_index: int) -> None:
    if not isinstance(position, list) or len(position) < 2:
        raise ValueError(f"Feature {feature_index} coordinates must contain finite longitude and latitude")
    longitude, latitude = position[:2]
    values_are_finite = all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in (longitude, latitude)
    )
    if not values_are_finite:
        raise ValueError(f"Feature {feature_index} coordinates must contain finite longitude and latitude")
    if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
        raise ValueError(f"Feature {feature_index} coordinates are outside EPSG:4326 bounds")


def _validate_feature(feature: object, feature_index: int) -> None:
    if not isinstance(feature, dict) or feature.get("type") != "Feature":
        raise ValueError(f"Feature {feature_index} must be a GeoJSON Feature")
    _validate_feature_properties(feature.get("properties"), feature_index)
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError(f"Feature {feature_index} geometry must be an object")
    geometry_type = geometry.get("type")
    if not isinstance(geometry_type, str) or geometry_type not in {"Point", "LineString"}:
        raise ValueError(f"Unsupported GeoJSON geometry type: {geometry_type!r}")
    _validate_coordinates(geometry_type, geometry.get("coordinates"), feature_index)


def load_mapping(path: str | Path) -> GeoMapping:
    """Load a mapping, raising I/O/JSON errors or ValueError for an invalid schema."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("type") != "FeatureCollection":
        raise ValueError("Mapping must be a GeoJSON FeatureCollection")
    metadata = _validate_metadata(doc.get("metadata", {}))
    features = doc.get("features")
    if not isinstance(features, list):
        raise ValueError("FeatureCollection must contain a features array")
    for feature_index, feature in enumerate(features):
        _validate_feature(feature, feature_index)
    doc["metadata"] = metadata
    return doc

def _point_trace(
    go: Any,
    feature: GeoFeature,
    result: MapResult | None,
    metric: MapMetric,
    provenance: str,
) -> go.Scattermap:
    props = feature["properties"]
    longitude, latitude = feature["geometry"]["coordinates"][:2]
    asset_type = props["asset_type"]
    measurements = (result or {}).get("bus_voltage_pu", {})
    raw = measurements.get(props.get("electrical_index", -1)) if asset_type == "bus" else None
    if raw is None or (result or {}).get("status") != "evaluated":
        value, label = None, "Unknown (no successful mapped result)"
    elif metric == "voltage deviation":
        value = abs(float(raw) - 1.0)
        label = f"{float(raw):.4f} pu; deviation {value:.4f} pu"
    elif metric == "voltage magnitude":
        value, label = float(raw), f"{float(raw):.4f} pu"
    else:
        raise ValueError(f"Unsupported map metric: {metric!r}")
    hover = (f"{asset_type} {props['asset_id']}<br>{label}<br>"
             f"status: {(result or {}).get('status', 'no result')}<br>source: {provenance}")
    default_color = {"bus": "#44546a", "load": "#d95f02", "gen": "#1b9e77", "trafo": "#8c564b", "line": "#65758b"}[asset_type]
    marker_color: str | float = default_color if value is None else value
    return go.Scattermap(
        lon=[longitude], lat=[latitude], mode="markers",
        marker={"size": 13, "color": marker_color, "colorscale": "RdYlGn_r",
                "cmin": 0, "cmax": 0.1, "showscale": value is not None},
        name=asset_type, text=[hover], hoverinfo="text",
    )


def _line_trace(go: Any, feature: GeoFeature, result: MapResult | None, provenance: str):
    props = feature["properties"]
    coordinates = feature["geometry"]["coordinates"]
    longitudes, latitudes = zip(*[position[:2] for position in coordinates])
    measurements = (result or {}).get("line_loading_percent", {})
    index = props.get("electrical_index")
    raw = measurements.get(index) if index is not None and (result or {}).get("status") == "evaluated" else None
    color = "#888888" if raw is None else ("#c62828" if float(raw) > 100 else "#2878b5")
    loading = "Unknown" if raw is None else f"{float(raw):.1f}%"
    hover = (f"{props['asset_type']} {props['asset_id']}<br>loading: {loading}<br>"
             f"limit: {props.get('limit', 'not supplied')}<br>provenance: {provenance}")
    return go.Scattermap(
        lon=longitudes, lat=latitudes, mode="lines",
        line={"width": 4, "color": color}, name=props["asset_type"],
        text=[hover] * len(longitudes), hoverinfo="text",
    )


def plot_map(
    mapping: GeoMapping,
    result: MapResult | None,
    metric: MapMetric = "voltage deviation",
) -> go.Figure:
    if metric not in {"voltage deviation", "voltage magnitude"}:
        raise ValueError(f"Unsupported map metric: {metric!r}")
    import plotly.graph_objects as go

    metadata = mapping["metadata"]
    figure = go.Figure()
    for feature in mapping["features"]:
        geometry = feature["geometry"]
        if geometry["type"] == "Point":
            figure.add_trace(_point_trace(go, feature, result, metric, metadata["provenance"]))
        else:
            figure.add_trace(_line_trace(go, feature, result, metadata["provenance"]))
    figure.update_layout(
        map={"style": "open-street-map",
             "center": {"lat": metadata.get("center_lat", 37.5), "lon": metadata.get("center_lon", -122.2)},
             "zoom": 11},
        margin={"l": 0, "r": 0, "t": 10, "b": 0}, legend={"orientation": "h"},
    )
    return figure
