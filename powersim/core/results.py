"""Shared result types for OPF solvers and power-flow audits."""
from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class BranchResult:
    asset_id: str
    p_from_mw: float
    p_to_mw: float
    limit_mw: float | None
    loading_percent: float
    limit_percent: float
    active: bool
    violated: bool


@dataclass(frozen=True)
class SolveResult:
    status: Literal["ok", "failed", "invalid"]
    converged: bool
    physics_mode: str
    operation: str
    solver_seconds: float
    diagnostics: tuple[str, ...] = ()
    generation_mw: dict[str, float] = field(default_factory=dict)
    objective: float | None = None
    branches: tuple[BranchResult, ...] = ()
    active_lower: tuple[bool, ...] = ()
    active_upper: tuple[bool, ...] = ()
    proposed_generation_mw: dict[str, float] | None = None
    raw_balance_residual_mw: float | None = None
    slack_adjustment_mw: dict[str, float] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    bus_voltage_pu: dict[str, float] | None = None
    reactive_generation_mvar: dict[str, float] | None = None
    active_reactive_lower: tuple[bool, ...] | None = None
    active_reactive_upper: tuple[bool, ...] | None = None
    active_voltage_lower: tuple[bool, ...] | None = None
    active_voltage_upper: tuple[bool, ...] | None = None
    model_id: str | None = None
    configuration_id: str | None = None
    input_id: str | None = None
    generator_ids: tuple[str, ...] = ()
    bus_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)
