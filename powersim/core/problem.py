"""Reusable OPF instances and the interface for alternative dispatch solvers."""
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from typing import Protocol

from pandapower import pandapowerNet

from .config import SimulationConfig
from .contracts import DispatchContext, DispatchDecision, DispatchFailure, Generator
from ..instance_generation.network import build_dispatch_inputs, nodal_loads


@dataclass(frozen=True, eq=False)
class OPFInstance:
    """Snapshot a configured network and its current operating inputs."""

    net: InitVar[pandapowerNet]
    config: SimulationConfig = field(default_factory=SimulationConfig)
    context: DispatchContext = field(init=False)
    generators: tuple[Generator, ...] = field(init=False)
    net_load_mw: tuple[float, ...] = field(init=False)
    _network: pandapowerNet = field(init=False, repr=False)

    def __post_init__(self, net: pandapowerNet) -> None:
        network = deepcopy(net)
        inputs = build_dispatch_inputs(network, self.config)
        object.__setattr__(self, "_network", network)
        object.__setattr__(self, "context", inputs["context"])
        object.__setattr__(self, "generators", inputs["generators"])
        object.__setattr__(self, "net_load_mw", nodal_loads(network))

    def copy_network(self) -> pandapowerNet:
        """Give a solver a private copy of the topology, limits and P/Q inputs."""
        return deepcopy(self._network)


class OPFSolver(Protocol):
    """Compute an OPF dispatch for the supplied instance."""

    def __call__(self, instance: OPFInstance, /) -> DispatchDecision | DispatchFailure: ...
