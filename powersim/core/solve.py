"""Run the built-in OPF backend or audit a custom solver's dispatch."""
from dataclasses import replace
import time

from pandapower import pandapowerNet

from ..config import SimulationConfig
from ..contracts import DispatchDecision, DispatchFailure
from ..network import build_dispatch_inputs
from ..problem import OPFInstance, OPFSolver
from ..solvers.numerical.pandapower import run_opf
from .power_flow import collect_result, evaluate_decision
from .results import SolveResult


def solve_reference(net: pandapowerNet | OPFInstance, config: SimulationConfig | None = None, *,
                    solver: OPFSolver | None = None) -> SolveResult:
    """Solve an instance with pandapower or audit an alternative solver's dispatch.

    An instance supplies its own configuration and is never changed. Passing a
    raw network to the default solver retains pandapower's in-place results.
    """
    started = time.perf_counter()
    instance = net if isinstance(net, OPFInstance) else None
    config = config or (instance.config if instance is not None else SimulationConfig())
    try:
        if instance is not None and config != instance.config:
            raise ValueError("Configuration must match the OPF instance")
        if solver is None:
            if instance is not None:
                net = instance.copy_network()
            inputs = build_dispatch_inputs(net, config)
            run_opf(net, config.physics_mode)
            result = collect_result(net, config, "opf", time.perf_counter()-started, inputs)
        else:
            instance = instance if instance is not None else OPFInstance(net, config)
            decision = solver(instance)
            if isinstance(decision, DispatchFailure):
                return SolveResult("failed", False, config.physics_mode, "opf", time.perf_counter()-started,
                                   (f"{decision.status}: {decision.diagnostic}",))
            if not isinstance(decision, DispatchDecision):
                raise TypeError("Solver must return DispatchDecision or DispatchFailure")
            if not all(isinstance(note, str) for note in decision.diagnostics):
                raise TypeError("Solver diagnostics must be strings")
            audit = evaluate_decision(instance.copy_network(), decision, config)
            result = replace(audit, operation="opf", solver_seconds=time.perf_counter()-started,
                             diagnostics=(*audit.diagnostics, *decision.diagnostics))
        if result.status == "ok" and not result.metrics["feasible"]:
            return replace(result, status="invalid",
                           diagnostics=("Converged OPF failed independent label validation", *result.diagnostics))
        return result
    except Exception as exc:
        return SolveResult("failed", False, config.physics_mode, "opf", time.perf_counter()-started,
                           (f"{type(exc).__name__}: {exc}",))
