"""Compatibility imports for policy evaluation and benchmark setup."""
from .core.power_flow import evaluate_decision
from .core.simulator import evaluate
from .core.solve import solve_reference
from .instance_generation.network import build_dispatch_inputs, make_case, network_fingerprint, observation

__all__ = ["build_dispatch_inputs", "evaluate", "evaluate_decision", "make_case", "network_fingerprint",
           "observation", "solve_reference"]
