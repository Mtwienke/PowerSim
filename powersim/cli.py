"""Headless research commands. ML imports are intentionally lazy."""
import argparse
from importlib import import_module
import json
from pathlib import Path

from .config import BENCHMARKS, SimulationConfig
from .data import DatasetConfig, generate_dataset, load_dataset_instance
from .network import make_case
from .solver import solve_reference


def _hidden_widths(value):
    try:
        widths = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated positive hidden widths") from exc
    if not widths or min(widths) < 1:
        raise argparse.ArgumentTypeError("Hidden widths must be positive")
    return widths


def _learning_rates(value):
    try:
        return tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated learning rates") from exc


def _solver_callable(value):
    module, _, name = value.partition(":")
    if not module or not name:
        raise argparse.ArgumentTypeError("Use MODULE:CALLABLE for an alternative solver")
    try:
        solver = getattr(import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(f"Cannot import solver {value!r}: {exc}") from exc
    if not callable(solver):
        raise argparse.ArgumentTypeError(f"Solver {value!r} must be callable")
    return solver


def main(argv=None):
    parser = argparse.ArgumentParser(prog="powersim")
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("reference", help="Solve and validate numerical OPF")
    solve.add_argument("--physics", choices=["dc", "ac"], help="Physics for a bundled case (default: dc)")
    solve.add_argument("--benchmark", choices=BENCHMARKS, help="Bundled case (default: case30)")
    solve.add_argument("--branch-limit-scale", type=float, help="Branch limit multiplier (default: 1)")
    solve.add_argument("--solver", type=_solver_callable, metavar="MODULE:CALLABLE",
                       help="Alternative OPF dispatch solver (default: pandapower)")
    solve.add_argument("--dataset", type=Path, help="Solve a saved attempt using the dataset's network/configuration")
    solve.add_argument("--split", help="Saved split (default: iid_test)")
    solve.add_argument("--sample", type=int, help="Original zero-based attempt number (default: 0)")
    data = commands.add_parser("dataset", help="Generate seeded DC-OPF labels")
    data.add_argument("directory")
    data.add_argument("--preset", choices=["smoke", "paper"], default="smoke")
    data.add_argument("--seed", type=int, default=2026)
    data.add_argument("--benchmark", choices=BENCHMARKS, default="case30")
    data.add_argument("--branch-limit-scale", type=float, default=.9)
    data.add_argument("--workers", type=int, default=1, help="Independent OPF processes (default: 1)")
    data.add_argument("--config", help="JSON DatasetConfig overrides, including renewable conversion parameters")
    train = commands.add_parser("train", help="Train baseline or configurable physics-guided DC network")
    train.add_argument("dataset")
    train.add_argument("output")
    train.add_argument("--method", choices=["baseline", "physics-guided", "M1", "M2", "M3-example", "M4", "M5"], default="baseline")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--hidden", type=_hidden_widths, default=(200, 200, 200))
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--activation", choices=["tanh", "relu", "silu"], default="tanh")
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--learning-rate", type=float, default=.001)
    train.add_argument("--checkpoint-selection", choices=["best_validation", "last"], default="best_validation")
    train.add_argument("--initialize-from", help="Compatible model checkpoint; weights only, with fresh Adam state")
    train.add_argument("--loss-weights", help='JSON overrides, e.g. \'{"L1":0,"L5":0.1}\'')
    train.add_argument("--config", help="JSON TrainConfig overrides (learning rate, perturbation, etc.)")
    decoder = commands.add_parser("decoder", help="Train frozen-network decoder and calibrate on validation only")
    decoder.add_argument("dataset")
    decoder.add_argument("model")
    decoder.add_argument("output")
    decoder.add_argument("--epochs", type=int, default=100)
    decoder.add_argument("--batch-size", type=int, default=64)
    decoder.add_argument("--hidden", type=_hidden_widths, default=(128,), help="One or more hidden widths, e.g. 128,128")
    decoder.add_argument("--learning-rate", type=float, default=.001)
    decoder.add_argument("--quantile", type=float, default=.95)
    decoder.add_argument("--error-units", choices=["normalized", "mw2", "mae_mw", "rmse_mw", "l1_mw", "linf_mw"], default="normalized")
    decoder.add_argument("--threshold", type=float, help="Explicit fixed threshold in the chosen score units (not an accuracy guarantee)")
    decoder.add_argument("--seed", type=int, default=2026)
    decoder.add_argument("--config", help="JSON decoder overrides, including activation, feature layer and optimizer numerics")
    calibration = commands.add_parser("calibrate", help="Recalibrate a decoder threshold on validation only")
    calibration.add_argument("dataset")
    calibration.add_argument("model")
    calibration.add_argument("decoder")
    calibration.add_argument("output")
    calibration.add_argument("--quantile", type=float, default=.95)
    calibration.add_argument("--error-units", choices=["normalized", "mw2", "mae_mw", "rmse_mw", "l1_mw", "linf_mw"])
    calibration.add_argument("--threshold", type=float, help="Explicit fixed threshold instead of a validation quantile")
    pilot = commands.add_parser("pilot", help="Shared-M1, warm-started Gao training pilot on an existing DC dataset")
    pilot.add_argument("dataset")
    pilot.add_argument("output")
    pilot.add_argument("--pretrain-epochs", type=int, default=1000)
    pilot.add_argument("--fine-tune-epochs", type=int, default=200)
    pilot.add_argument("--learning-rates", type=_learning_rates, default=(.01,), help="Add 0.001 as an explicit stability control")
    pilot.add_argument("--hidden", type=_hidden_widths, default=(200, 200, 200))
    pilot.add_argument("--activation", choices=["tanh", "relu", "silu"], default="tanh")
    pilot.add_argument("--batch-size", type=int, default=64)
    pilot.add_argument("--seed", type=int, default=2027)
    pilot.add_argument("--checkpoint-selection", choices=["last", "best_validation"], default="last")
    pilot.add_argument("--perturb-std-normalized", type=float, default=.1)
    execution = pilot.add_mutually_exclusive_group()
    execution.add_argument("--plan", action="store_true", help="Validate and print the plan without writing or training")
    execution.add_argument("--resume", action="store_true", help="Verify and reuse completed stages; preserve interrupted attempts")
    evaluation = commands.add_parser("evaluate", help="Evaluate raw learned and optional decoder/OPF hybrid")
    evaluation.add_argument("dataset")
    evaluation.add_argument("model")
    evaluation.add_argument("output")
    evaluation.add_argument("--split", default="iid_test")
    evaluation.add_argument("--decoder")
    evaluation.add_argument("--workers", type=int, default=1, help="Independent scenario processes (default: 1)")
    evaluation.add_argument("--prediction-format", choices=["jsonl", "jsonl.gz"], default="jsonl")
    evaluation.add_argument("--decoder-norm-sensitivity", action="store_true")
    view = commands.add_parser("plot-solution", help="Export a solved grid graph as standalone interactive HTML")
    view.add_argument("output", type=Path)
    view.add_argument("--benchmark", choices=BENCHMARKS, default="case30")
    view.add_argument("--physics", choices=["dc", "ac"], default="dc")
    view.add_argument("--branch-limit-scale", type=float, default=1.)
    view.add_argument("--dataset", type=Path, help="Replay the saved reference instead of solving a new case")
    view.add_argument("--split", default="iid_test")
    view.add_argument("--sample", type=int, default=0, help="Original zero-based attempt number (not filtered row)")
    infer = commands.add_parser("infer", help="Online DC hybrid prediction; rejected cases logged separately")
    infer.add_argument("dataset", help="Source manifest/configuration (not held-out inputs)")
    infer.add_argument("model")
    infer.add_argument("decoder")
    infer.add_argument("input", help="JSON file containing the ordered net_load_mw array")
    infer.add_argument("--production-log", required=True)
    args = parser.parse_args(argv)
    if args.command == "reference":
        choices = {"physics_mode": args.physics, "benchmark": args.benchmark,
                   "branch_limit_scale": args.branch_limit_scale}
        overrides = {name: value for name, value in choices.items() if value is not None}
        if args.dataset is not None:
            if overrides:
                parser.error("--dataset uses its saved physics, benchmark and branch limits")
            instance = load_dataset_instance(args.dataset, args.split or "iid_test", args.sample or 0)
            result = solve_reference(instance, solver=args.solver)
        else:
            if args.split is not None or args.sample is not None:
                parser.error("--split and --sample require --dataset")
            config = SimulationConfig(**overrides)
            result = solve_reference(make_case(config), config, solver=args.solver)
        print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
        return 0 if result.status == "ok" else 1
    if args.command == "dataset":
        values = {"seed": args.seed, "simulation": SimulationConfig(benchmark=args.benchmark,
                                                                   branch_limit_scale=args.branch_limit_scale)}
        if args.config:
            from .data import RenewableConfig
            overrides = json.loads(Path(args.config).read_text())
            if "simulation" in overrides:
                overrides["simulation"] = SimulationConfig(**overrides["simulation"])
            if "renewable" in overrides:
                overrides["renewable"] = RenewableConfig(**overrides["renewable"])
            values.update(overrides)
        manifest = generate_dataset(args.directory, DatasetConfig.preset(args.preset, **values), workers=args.workers)
        print(json.dumps(manifest["splits"], indent=2))
        return 0
    if args.command == "train":
        from .ml import METHOD_WEIGHTS, TrainConfig, train_model
        weights = dict(METHOD_WEIGHTS[args.method])
        if args.loss_weights:
            weights.update(json.loads(args.loss_weights))
        values = {"hidden": args.hidden, "activation": args.activation,
                  "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed, "loss_weights": weights,
                  "learning_rate": args.learning_rate, "checkpoint_selection": args.checkpoint_selection}
        if args.config:
            values.update(json.loads(Path(args.config).read_text()))
        report = train_model(args.dataset, args.output, TrainConfig(**values), initialize_from=args.initialize_from)
        print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))
        return 0
    if args.command == "decoder":
        from .ml import train_decoder
        values = {"epochs": args.epochs, "batch_size": args.batch_size, "hidden": args.hidden,
                  "quantile": args.quantile, "units": args.error_units, "seed": args.seed,
                  "threshold": args.threshold, "learning_rate": args.learning_rate}
        if args.config:
            values.update(json.loads(Path(args.config).read_text()))
        report = train_decoder(args.dataset, args.model, args.output, **values)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "pilot":
        from .pilot import PilotConfig, pilot_plan, run_pilot
        config = PilotConfig(seed=args.seed, pretrain_epochs=args.pretrain_epochs,
                             fine_tune_epochs=args.fine_tune_epochs, learning_rates=args.learning_rates,
                             hidden=args.hidden, activation=args.activation, batch_size=args.batch_size,
                             checkpoint_selection=args.checkpoint_selection,
                             perturb_std_normalized=args.perturb_std_normalized)
        report = (pilot_plan(args.dataset, config) if args.plan else
                  run_pilot(args.dataset, args.output, config, resume=args.resume))
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "evaluate":
        from .experiments import evaluate_model
        report = evaluate_model(args.dataset, args.model, args.output, split=args.split,
                                 decoder_path=args.decoder, workers=args.workers,
                                 prediction_format=args.prediction_format,
                                 decoder_norm_sensitivity=args.decoder_norm_sensitivity)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "plot-solution":
        from .solution_view import dataset_solution, solution_figure
        if args.output.exists():
            parser.error(f"Output already exists: {args.output}")
        if args.dataset:
            net, result, demand, renewables = dataset_solution(args.dataset, args.split, args.sample)
            label = f"{net.name or 'Saved reference'} · {args.split}, attempt {args.sample}"
        else:
            config = SimulationConfig(args.physics, args.benchmark, branch_limit_scale=args.branch_limit_scale)
            net = make_case(config)
            result = solve_reference(net, config).to_dict()
            demand = renewables = label = None
        figure = solution_figure(net, result, demand=demand, renewables=renewables, label=label)
        with args.output.open("x", encoding="utf-8") as output:
            figure.write_html(output, include_plotlyjs=True)
        print(f"{result['status']}: {args.output}")
        return 0 if result["status"] == "ok" else 1
    if args.command == "calibrate":
        from .ml import recalibrate_decoder
        report = recalibrate_decoder(args.dataset, args.model, args.decoder, args.output,
                                      quantile=args.quantile, units=args.error_units, threshold=args.threshold)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "infer":
        from dataclasses import asdict
        from .data import load_dataset
        from .hybrid import HybridPredictor
        from .ml import load_decoder, load_model
        from .network import apply_nodal_loads
        manifest, _ = load_dataset(args.dataset)
        if Path(args.production_log).resolve().is_relative_to(Path(args.dataset).resolve()):
            raise ValueError("Production rejection log must be separate from dataset/held-out artifacts")
        model, checkpoint = load_model(args.model)
        if checkpoint["model_id"] != manifest["model_id"]:
            raise ValueError("Model/configuration mismatch")
        if manifest["configuration"]["simulation"]["physics_mode"] != "dc":
            raise ValueError("DC-trained inference cannot be used as an AC method")
        decoder, meta = load_decoder(args.decoder, args.model, model)
        values = json.loads(Path(args.input).read_text())
        def fallback():
            config = SimulationConfig(**manifest["configuration"]["simulation"])
            net = make_case(config)
            apply_nodal_loads(net, values)
            return solve_reference(net, config)
        result = HybridPredictor(model, decoder, meta, manifest["generator_ids"]).predict(
            values, fallback, production_log=args.production_log)
        print(json.dumps(asdict(result), indent=2, allow_nan=False))
        return 1 if result.status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
