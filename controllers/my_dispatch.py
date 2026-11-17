"""The intended single-file starting point for custom dispatch logic."""
from powersim.contracts import (
    DispatchContext,
    DispatchDecision,
    DispatchFailure,
    DispatchObservation,
)


def choose_dispatch(observation: DispatchObservation) -> DispatchDecision | DispatchFailure:
    """Renewables first, then conventional generators in stable merit order."""
    # v2 observations are already NET of fixed renewables.
    demand = sum(observation.net_load_mw) if observation.net_load_mw else observation.total_demand_mw - observation.renewable_available_mw
    targets = {gen.asset_id: gen.p_min_mw for gen in observation.generators}
    residual = demand - sum(targets.values())
    if residual < -1e-6:
        return DispatchFailure("minimum_generation", "Demand is below total minimum generation")
    for gen in sorted(observation.generators, key=lambda g: (g.marginal_cost_per_mwh, g.asset_id)):
        addition = min(gen.p_max_mw - gen.p_min_mw, max(0.0, residual))
        targets[gen.asset_id] += addition
        residual -= addition
    if residual > 1e-6:
        return DispatchFailure("insufficient_capacity", f"Unserved active demand: {residual:.3f} MW")
    return DispatchDecision(
        active_power_mw=targets,
        diagnostics=("Renewables-first merit-order heuristic; not an OPF.",),
    )


class MyDispatchPolicy:
    def __init__(self) -> None:
        self.context: DispatchContext | None = None

    def reset(self, context: DispatchContext) -> None:
        self.context = context

    def decide(self, observation: DispatchObservation) -> DispatchDecision | DispatchFailure:
        if self.context is None:
            return DispatchFailure("policy_not_initialized", "reset must be called before decide")
        if observation.context_fingerprint != self.context.network_fingerprint:
            return DispatchFailure("context_mismatch", "Observation does not match the active network context")
        observed_assets = tuple(sorted(generator.asset_id for generator in observation.generators))
        if observed_assets != tuple(sorted(self.context.controllable_assets)):
            return DispatchFailure("asset_mismatch", "Observation generators do not match the active controllable assets")
        return choose_dispatch(observation)
