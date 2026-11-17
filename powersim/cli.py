"""Headless research commands. ML imports are intentionally lazy."""
import argparse
import json

from .config import SimulationConfig
from .data import DatasetConfig, generate_dataset
from .network import make_case
from .solver import solve_reference


def main(argv=None):
    parser = argparse.ArgumentParser(prog="powersim")
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("reference", help="Solve and validate numerical OPF")
    solve.add_argument("--physics", choices=["dc", "ac"], default="dc")
    solve.add_argument("--benchmark", choices=["case30", "case14"], default="case30")
    data = commands.add_parser("dataset", help="Generate seeded DC-OPF labels")
    data.add_argument("directory")
    data.add_argument("--preset", choices=["smoke", "paper"], default="smoke")
    data.add_argument("--seed", type=int, default=2026)
    data.add_argument("--benchmark", choices=["case30", "case14"], default="case30")
    data.add_argument("--branch-limit-scale", type=float, default=.9)
    data.add_argument("--config", help="JSON DatasetConfig overrides, including renewable conversion parameters")
    args = parser.parse_args(argv)
    if args.command == "reference":
        config = SimulationConfig(args.physics, args.benchmark)
        result = solve_reference(make_case(config), config)
        print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
        return 0 if result.status == "ok" else 1
    if args.command == "dataset":
        values = {"seed": args.seed, "simulation": SimulationConfig(benchmark=args.benchmark,
                                                                   branch_limit_scale=args.branch_limit_scale)}
        if args.config:
            from pathlib import Path
            from .data import RenewableConfig
            overrides = json.loads(Path(args.config).read_text())
            if "simulation" in overrides:
                overrides["simulation"] = SimulationConfig(**overrides["simulation"])
            if "renewable" in overrides:
                overrides["renewable"] = RenewableConfig(**overrides["renewable"])
            values.update(overrides)
        manifest = generate_dataset(args.directory, DatasetConfig.preset(args.preset, **values))
        print(json.dumps(manifest["splits"], indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
