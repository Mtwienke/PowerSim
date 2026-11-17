"""Versioned JSON-friendly public policy interface; all power units are MW/MVAr."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

@dataclass(frozen=True)
class Generator:
    asset_id: str
    p_min_mw: float
    p_max_mw: float
    marginal_cost_per_mwh: float = 1.0
    bus_id: str = ""
    source_type: str = "gen"
    cost_quadratic: float = 0.0
    cost_constant: float = 0.0
    q_min_mvar: float = -1e9
    q_max_mvar: float = 1e9

@dataclass(frozen=True)
class DispatchContext:
    schema_version: str
    network_fingerprint: str
    units: tuple[tuple[str, str], ...]
    controllable_assets: tuple[str, ...]
    assumptions: tuple[str, ...] = ()
    bus_ids: tuple[str, ...] = ()
    physics_mode: str = "dc"
    configuration_id: str = ""

@dataclass(frozen=True)
class DispatchObservation:
    schema_version: str
    context_fingerprint: str
    timestamp: str
    step_id: int
    interval_hours: float
    total_demand_mw: float
    renewable_available_mw: float
    generators: tuple[Generator, ...]
    bus_ids: tuple[str, ...] = ()
    net_load_mw: tuple[float, ...] = ()
    scenario_id: str = ""
    physics_mode: str = "dc"
    configuration_id: str = ""

@dataclass(frozen=True)
class DispatchDecision:
    schema_version: str = "1.0"
    active_power_mw: Mapping[str, float] = field(default_factory=dict)
    voltage_targets_pu: Mapping[str, float] = field(default_factory=dict)
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_power_mw", MappingProxyType(dict(self.active_power_mw)))
        object.__setattr__(self, "voltage_targets_pu", MappingProxyType(dict(self.voltage_targets_pu)))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

@dataclass(frozen=True)
class DispatchFailure:
    status: str
    diagnostic: str

class DispatchPolicy(Protocol):
    def reset(self, context: DispatchContext) -> None: ...
    def decide(self, observation: DispatchObservation) -> DispatchDecision | DispatchFailure: ...
