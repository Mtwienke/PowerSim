"""Validated, orthogonal experiment choices."""
from dataclasses import dataclass
from typing import Literal

PhysicsMode = Literal["dc", "ac"]
SolutionMethod = Literal["numerical", "heuristic", "learned"]
Benchmark = Literal["case30", "case14", "germany30"]
BENCHMARKS = ("case30", "case14", "germany30")


@dataclass(frozen=True)
class SimulationConfig:
    physics_mode: PhysicsMode = "dc"
    benchmark: Benchmark = "case30"
    method: SolutionMethod = "numerical"
    branch_limit_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.physics_mode not in ("dc", "ac"):
            raise ValueError("physics_mode must be dc or ac")
        if self.benchmark not in BENCHMARKS:
            raise ValueError(f"benchmark must be one of {', '.join(BENCHMARKS)}")
        if self.benchmark == "germany30" and self.physics_mode != "dc":
            raise ValueError("Germany-30 is a reduced DC-only transmission equivalent")
        if self.method not in ("numerical", "heuristic", "learned"):
            raise ValueError("method must be numerical, heuristic or learned")
        if not 0 < self.branch_limit_scale <= 10:
            raise ValueError("branch_limit_scale must be in (0, 10]")
        if self.method == "learned" and self.physics_mode != "dc":
            raise ValueError("The learned paper-inspired method is DC only")
