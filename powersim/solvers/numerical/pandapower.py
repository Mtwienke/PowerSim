"""Run pandapower's native OPF and retain its solved network tables."""
import pandapower as pp

from ...core.config import PhysicsMode


def run_opf(net: pp.pandapowerNet, physics_mode: PhysicsMode) -> None:
    (pp.rundcopp if physics_mode == "dc" else pp.runopp)(net, numba=False)
    if not net.OPF_converged:
        raise ValueError("OPF did not converge")
