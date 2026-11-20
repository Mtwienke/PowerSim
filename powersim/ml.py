"""Optional CPU learning stack. All Jacobian penalties are in physical MW/MW."""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import time

import numpy as np
try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError("Training/inference requires: pip install -e '.[ml]'") from exc

from .data import load_dataset, package_versions, write_json
from .metrics import accuracy
from .network import digest


def tensor(value):
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def artifact_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_identity(manifest):
    return digest({"configuration": manifest["configuration"], "model_id": manifest["model_id"],
                    "bus_ids": manifest["bus_ids"], "generator_ids": manifest["generator_ids"]})


METHOD_WEIGHTS = {
    "baseline": {"L0": 1.0},
    "physics-guided": {f"L{i}": 1.0 for i in range(6)},
    "M1": {"L0": 1.0},
    "M2": {"L0": 1.0, "L1": 1.0},
    "M4": {f"L{i}": 1.0 for i in range(5)},
    "M5": {f"L{i}": 1.0 for i in range(6)},
    # Author's visible branch-penalty comparator; not full external DeepOPF.
    "M3-example": {"L0": 1.0, "branch_penalty": 1.0},
}


def training_data_hash(arrays):
    """Bind initialization to train/validation data, never to held-out values."""
    checksum = hashlib.sha256()
    for name in sorted(arrays):
        if not (name.startswith(("train_", "validation_")) or
                name in ("x_mean", "x_scale", "y_mean", "y_scale", "m_g", "m_d", "branch_constant",
                         "branch_limits_mw")):
            continue
        value = np.ascontiguousarray(arrays[name])
        checksum.update(f"{name}:{value.dtype}:{value.shape}".encode())
        checksum.update(value.tobytes())
    return checksum.hexdigest()


def state_dict_hash(state):
    checksum = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        checksum.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        checksum.update(value.numpy().tobytes())
    return checksum.hexdigest()


class OPFNetwork(nn.Module):
    def __init__(self, x_mean, x_scale, y_mean, y_scale, hidden=(200, 200, 200), activation="tanh"):
        super().__init__()
        if len(hidden) < 2 or any(h < 1 for h in hidden):
            raise ValueError("At least two positive hidden widths required for decoder features")
        activations = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}
        if activation not in activations:
            raise ValueError("activation must be tanh, relu or silu")
        self.hidden = tuple(hidden)
        self.activation_name = activation
        self.activation = activations[activation]()
        for name, values in (("x_mean", x_mean), ("x_scale", x_scale), ("y_mean", y_mean), ("y_scale", y_scale)):
            self.register_buffer(name, tensor(values))
        if not torch.all(self.x_scale > 0) or not torch.all(self.y_scale > 0):
            raise ValueError("Normalization scales must be positive")
        sizes = [len(x_mean), *hidden]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.output = nn.Linear(hidden[-1], len(y_mean))

    def normalized_forward(self, z):
        for layer in self.layers:
            z = self.activation(layer(z))
        return self.output(z)

    def forward(self, x):
        return self.normalized_forward((x-self.x_mean)/self.x_scale)*self.y_scale+self.y_mean

    def features(self, x, layer="penultimate_hidden"):
        """Legacy second hidden layer, or explicitly selected last hidden layer."""
        if layer not in ("penultimate_hidden", "last_hidden"):
            raise ValueError("Unknown decoder feature layer")
        z = (x-self.x_mean)/self.x_scale
        selected = self.layers[:-1] if layer == "penultimate_hidden" else self.layers
        for hidden in selected:
            z = self.activation(hidden(z))
        return z


def physical_jacobian(model, x, create_graph=True):
    """Independent samples only (no batch normalization): shape batch x G x D.

    Differentiating model(raw_MW) includes dY/dX = scale_y * dZy/dZx / scale_x.
    create_graph=True is essential: penalties differentiate through these J's.
    """
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)
    y = model(x)
    return torch.stack([torch.autograd.grad(y[:, g].sum(), x, create_graph=create_graph,
                        retain_graph=True)[0] for g in range(y.shape[1])], dim=1)


def residual_loss(values, norm="mse"):
    if norm not in ("mse", "mae"):
        raise ValueError("Residual norm must be mse or mae")
    return values.square() if norm == "mse" else values.abs()


def masked_residual_loss(values, mask, norm="mse", reduction="active"):
    if reduction not in ("active", "full"):
        raise ValueError("Mask reduction must be active or full")
    mask = mask.to(dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(values)
    total = (residual_loss(values, norm)*expanded).sum()
    return total/expanded.sum().clamp_min(1) if reduction == "active" else total/max(values.numel(), 1)


def masked_mean_square(values, mask):
    return masked_residual_loss(values, mask)


def jacobian_losses(j, reference_j, reference_valid, active_generator, active_branch, m_g, m_d,
                    *, pattern_norm="mse", reduction="active"):
    return {
        "L1": masked_residual_loss(j-reference_j, reference_valid, "mse", reduction),
        "L2": residual_loss(j.sum(dim=1)-1, pattern_norm).mean(),
        "L3": masked_residual_loss(j, active_generator, pattern_norm, reduction),
        "L4": masked_residual_loss(torch.einsum("bg,ngd->nbd", m_g, j)-m_d, active_branch,
                                   pattern_norm, reduction),
    }


@dataclass(frozen=True)
class TrainConfig:
    hidden: tuple[int, ...] = (200, 200, 200)
    activation: str = "tanh"
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = .001
    seed: int = 2026
    perturb_std_normalized: float = .1
    loss_weights: dict[str, float] = field(default_factory=lambda: {"L0": 1.0})
    checkpoint_selection: str = "best_validation"
    normalization: str = "standard"
    power_base_mw: float = 100.
    prediction_loss: str = "mse"
    prediction_units: str = "mw"
    pattern_norm: str = "mse"
    mask_reduction: str = "active"
    sensitivity_policy: str = "available"
    branch_subset: tuple[str, ...] = ()
    initializer: str = "pytorch"
    adam_variant: str = "pytorch"
    adam_epsilon: float = 1e-8
    learning_rate_decay: float = 0.

    def __post_init__(self):
        if (not all(isinstance(v, int) and v > 0 for v in (self.epochs, self.batch_size))
                or not np.isfinite([self.learning_rate, self.perturb_std_normalized]).all()
                or self.learning_rate <= 0 or self.perturb_std_normalized < 0):
            raise ValueError("Invalid training settings")
        if len(self.hidden) < 2 or not all(isinstance(h, int) and h > 0 for h in self.hidden):
            raise ValueError("At least two positive integer hidden widths required")
        if self.activation not in ("tanh", "relu", "silu"):
            raise ValueError("Unsupported activation")
        if set(self.loss_weights) - {*(f"L{i}" for i in range(6)), "branch_penalty"}:
            raise ValueError("Unknown loss term")
        if not any(v > 0 for v in self.loss_weights.values()) or any(not np.isfinite(v) or v < 0 for v in self.loss_weights.values()):
            raise ValueError("Loss weights must be finite, nonnegative, with at least one positive")
        if self.checkpoint_selection not in ("best_validation", "last"):
            raise ValueError("checkpoint_selection must be best_validation or last")
        if self.normalization not in ("standard", "per_unit") or self.prediction_units not in ("mw", "model"):
            raise ValueError("Unknown normalization or prediction units")
        if self.prediction_loss not in ("mse", "mae") or self.pattern_norm not in ("mse", "mae"):
            raise ValueError("Loss norms must be mse or mae")
        if self.mask_reduction not in ("active", "full") or self.sensitivity_policy not in ("available", "example"):
            raise ValueError("Unknown mask reduction or sensitivity policy")
        if self.initializer not in ("pytorch", "glorot_uniform") or self.adam_variant not in ("pytorch", "keras_v2"):
            raise ValueError("Unknown initializer or Adam variant")
        if (not np.isfinite([self.power_base_mw, self.adam_epsilon, self.learning_rate_decay]).all()
                or min(self.power_base_mw, self.adam_epsilon) <= 0 or self.learning_rate_decay < 0):
            raise ValueError("Invalid power base or optimizer numerics")
        if len(set(self.branch_subset)) != len(self.branch_subset) or any(not isinstance(b, str) for b in self.branch_subset):
            raise ValueError("branch_subset must contain distinct branch IDs")


def initialize_layers(module, initializer):
    if initializer == "glorot_uniform":
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)


def make_optimizer(parameters, *, learning_rate, variant="pytorch", epsilon=1e-8, decay=0.):
    if variant == "keras_v2":
        from .optim import KerasV2Adam
        return KerasV2Adam(parameters, lr=learning_rate, epsilon=epsilon, decay=decay)
    if variant != "pytorch":
        raise ValueError("Unknown Adam variant")
    return torch.optim.Adam(parameters, lr=learning_rate, eps=epsilon)


def supervised_loss(prediction, target, model, config):
    error = prediction-target
    if config.prediction_units == "model":
        error = error/model.y_scale
    return residual_loss(error, config.prediction_loss).mean()


def branch_flow_penalty(prediction, x, m_g, m_d, constant, limits):
    valid = torch.isfinite(limits) & (limits > 0) & (limits < 1e8)
    if not valid.any():
        return prediction.sum()*0
    flow = prediction @ m_g[valid].T - x @ m_d[valid].T + constant[valid]
    return torch.relu((flow/limits[valid]).square()-1).mean()


def train_model(dataset, output, config: TrainConfig | None = None, *, initialize_from=None):
    config = config or TrainConfig()
    manifest, a = load_dataset(dataset)
    if manifest["configuration"]["simulation"]["physics_mode"] != "dc":
        raise ValueError("Learned methods are DC only")
    if not len(a["validation_x"]):
        raise ValueError("Nonempty validation split required for model selection")
    training_input_ids = [digest(row.tolist()) for row in a["train_x"]]
    validation_input_ids = [digest(row.tolist()) for row in a["validation_x"]]
    if set(training_input_ids) & set(validation_input_ids):
        raise ValueError("Training and validation contain duplicate nodal inputs")
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    normalization = ([a["x_mean"], a["x_scale"], a["y_mean"], a["y_scale"]]
                     if config.normalization == "standard" else
                     [np.zeros_like(a["x_mean"]), np.full_like(a["x_scale"], config.power_base_mw),
                      np.zeros_like(a["y_mean"]), np.full_like(a["y_scale"], config.power_base_mw)])
    model = OPFNetwork(*normalization, config.hidden, config.activation)
    initialize_layers(model, config.initializer)
    if set(config.branch_subset)-set(manifest["branch_ids"]):
        raise ValueError("Unknown branch_subset IDs")
    if config.loss_weights.get("branch_penalty", 0) > 0 and "branch_limits_mw" not in a:
        raise ValueError("Branch-penalty comparator requires recorded branch_limits_mw")
    data_hash = training_data_hash(a)
    initialization = {"kind": "random", "seed": config.seed,
                      "optimizer_state": "Fresh Adam; no optimizer state is inherited."}
    if initialize_from is not None:
        parent = torch.load(initialize_from, map_location="cpu", weights_only=True)
        if (parent["dataset_id"] != dataset_identity(manifest) or parent["model_id"] != manifest["model_id"]
                or parent["bus_ids"] != manifest["bus_ids"] or parent["generator_ids"] != manifest["generator_ids"]
                or parent["physics_mode"] != "dc"):
            raise ValueError("Warm-start checkpoint dataset/model/ordering mismatch")
        if (tuple(parent["configuration"]["hidden"]) != tuple(config.hidden) or
                parent["configuration"]["activation"] != config.activation):
            raise ValueError("Warm-start checkpoint architecture mismatch")
        if parent["configuration"].get("normalization", "standard") != config.normalization:
            raise ValueError("Warm-start checkpoint normalization mismatch")
        if (parent["training_scenario_ids"] != a["train_scenario_ids"].tolist() or
                parent["validation_scenario_ids"] != a["validation_scenario_ids"].tolist() or
                parent["training_input_ids"] != training_input_ids or parent["validation_input_ids"] != validation_input_ids):
            raise ValueError("Warm-start checkpoint training/validation inputs mismatch")
        if parent.get("training_data_sha256", data_hash) != data_hash:
            raise ValueError("Warm-start checkpoint training/validation data mismatch")
        for name in ("x_mean", "x_scale", "y_mean", "y_scale"):
            if not torch.equal(parent["state_dict"][name], getattr(model, name)):
                raise ValueError("Warm-start checkpoint normalization mismatch")
        parent_state_hash = state_dict_hash(parent["state_dict"])
        if parent.get("state_dict_sha256", parent_state_hash) != parent_state_hash:
            raise ValueError("Warm-start checkpoint state checksum mismatch")
        model.load_state_dict(parent["state_dict"])
        initialization.update(
            kind="checkpoint", checkpoint_path=str(Path(initialize_from).resolve()),
            checkpoint_sha256=artifact_hash(initialize_from),
            parent_epoch=parent.get("epoch"), parent_selection=parent.get("checkpoint_selection"),
            training_data_hash_verified="training_data_sha256" in parent)
    initialization["state_dict_sha256"] = state_dict_hash(model.state_dict())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    optimizer = make_optimizer(model.parameters(), learning_rate=config.learning_rate,
                               variant=config.adam_variant, epsilon=config.adam_epsilon,
                               decay=config.learning_rate_decay)
    # Intentionally address only TRAIN and VALIDATION, never held-out labels.
    x, y = tensor(a["train_x"]), tensor(a["train_y"])
    reference_j = tensor(a["train_jacobian"])
    valid_j = tensor(a["train_jacobian_valid"]).bool()
    if config.sensitivity_policy == "example":
        from .gao_audit import example_sensitivity_mask
        valid_j &= torch.as_tensor(example_sensitivity_mask(a["train_jacobian"]))
    active_g = tensor(a["train_lower"] | a["train_upper"]).bool()
    active_b = tensor(a["train_active_branch"]).bool()
    mg, md = tensor(a["m_g"]), tensor(a["m_d"])
    selected = ([manifest["branch_ids"].index(b) for b in config.branch_subset] if config.branch_subset
                else list(range(len(manifest["branch_ids"]))))
    pattern_mg, pattern_md, active_b = mg[selected], md[selected], active_b[:, selected]
    validation_x, validation_y = tensor(a["validation_x"]), tensor(a["validation_y"])
    with torch.no_grad():
        initial_validation = accuracy(model(validation_x).numpy(), a["validation_y"])
    # Keep batch order identical across ablations; perturbations cannot advance
    # the shuffle RNG merely because L5 was enabled.
    shuffle_rng = torch.Generator().manual_seed(config.seed)
    perturb_rng = torch.Generator().manual_seed(config.seed)
    history, best, best_state = [], float("inf"), None
    best_epoch = None
    updates = 0
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        totals = {f"L{i}": 0.0 for i in range(6)}
        if config.loss_weights.get("branch_penalty", 0) > 0:
            totals["branch_penalty"] = 0.
        for indices in torch.randperm(len(x), generator=shuffle_rng).split(config.batch_size):
            bx = x[indices].detach().requires_grad_(True)
            prediction = model(bx)
            terms = {"L0": supervised_loss(prediction, y[indices], model, config)}
            if any(config.loss_weights.get(f"L{i}", 0) > 0 for i in range(1, 5)):
                j = physical_jacobian(model, bx)
                terms.update(jacobian_losses(j, reference_j[indices], valid_j[indices],
                                             active_g[indices], active_b[indices], pattern_mg, pattern_md,
                                             pattern_norm=config.pattern_norm, reduction=config.mask_reduction))
            if config.loss_weights.get("L5", 0) > 0:
                noise = torch.randn(bx.shape, generator=perturb_rng)
                perturbed = (bx.detach()+noise*model.x_scale*config.perturb_std_normalized).requires_grad_(True)
                jp = physical_jacobian(model, perturbed)
                # No local L3/L4 masks are extrapolated to unlabeled points.
                terms["L5"] = residual_loss(jp.sum(dim=1)-1, config.pattern_norm).mean()
            if config.loss_weights.get("branch_penalty", 0) > 0:
                terms["branch_penalty"] = branch_flow_penalty(
                    prediction, bx, mg, md, tensor(a["branch_constant"]), tensor(a["branch_limits_mw"]))
            loss = sum(weight*terms[key] for key, weight in config.loss_weights.items() if weight > 0)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            if config.adam_variant == "pytorch" and config.learning_rate_decay:
                for group in optimizer.param_groups:
                    group["lr"] = config.learning_rate/(1+config.learning_rate_decay*updates)
            optimizer.step()
            updates += 1
            for key, value in terms.items():
                totals[key] += float(value.detach())*len(indices)/len(x)
        model.eval()
        with torch.no_grad():
            prediction = model(validation_x)
            validation = float((prediction-validation_y).square().mean())
            validation_accuracy = accuracy(prediction.numpy(), a["validation_y"])
        if not np.isfinite(validation):
            raise ValueError("Non-finite validation loss")
        history.append({"epoch": epoch+1, **totals, "validation_mse_mw2": validation,
                        "learning_rate_last": optimizer.param_groups[0]["lr"],
                        "validation_generator_accuracy_lt_1mw": validation_accuracy["generator_accuracy_lt_1mw"],
                        "validation_joint_accuracy_lt_1mw": validation_accuracy["joint_accuracy_lt_1mw"]})
        if validation < best:
            best, best_state, best_epoch = validation, deepcopy(model.state_dict()), epoch+1
    last_state = deepcopy(model.state_dict())
    checkpoint = {"configuration": asdict(config),
                   "model_id": manifest["model_id"], "dataset_id": dataset_identity(manifest),
                   "bus_ids": manifest["bus_ids"], "generator_ids": manifest["generator_ids"],
                   "physics_mode": "dc", "versions": package_versions(),
                   "training_data_sha256": data_hash, "initialization": initialization,
                   "training_scenario_ids": a["train_scenario_ids"].tolist(),
                   "validation_scenario_ids": a["validation_scenario_ids"].tolist(),
                   "training_input_ids": training_input_ids, "validation_input_ids": validation_input_ids}
    selected_state, selected_epoch = ((best_state, best_epoch) if config.checkpoint_selection == "best_validation"
                                      else (last_state, config.epochs))
    checkpoints = {}
    for name, state, epoch, selection in (
            ("best.pt", best_state, best_epoch, "best_validation"),
            ("last.pt", last_state, config.epochs, "last"),
            ("model.pt", selected_state, selected_epoch, config.checkpoint_selection)):
        torch.save({**checkpoint, "state_dict": state, "epoch": epoch,
                    "checkpoint_selection": selection, "state_dict_sha256": state_dict_hash(state)}, output / name)
        checkpoints[name] = {"sha256": artifact_hash(output / name), "epoch": epoch, "selection": selection}
    model.load_state_dict(selected_state)
    with torch.no_grad():
        training_accuracy = accuracy(model(x).numpy(), a["train_y"])
        validation_accuracy = accuracy(model(validation_x).numpy(), a["validation_y"])
    report = {"configuration": asdict(config), "training_seconds": time.perf_counter()-started,
               "selection_split": "validation" if config.checkpoint_selection == "best_validation" else None,
               "selected_epoch": selected_epoch, "best_epoch": best_epoch,
               "best_validation_mse_mw2": best, "last_validation_mse_mw2": history[-1]["validation_mse_mw2"],
               "initialization": initialization, "training_data_sha256": data_hash,
               "initial_validation_accuracy": initial_validation, "training_accuracy": training_accuracy,
               "validation_accuracy": validation_accuracy, "checkpoints": checkpoints,
               "rng_policy": "Independent deterministic initialization, shuffle and perturbation streams.",
               "history": history, "versions": package_versions(),
                "sensitivity_learning": {
                    "policy": config.sensitivity_policy, "available": int(np.sum(a["train_jacobian_valid"])),
                    "kept_for_L1": int(valid_j.sum()), "used_in_objective": config.loss_weights.get("L1", 0) > 0},
                "loss_units": {
                    "L0": ("MW" if config.prediction_units == "mw" else "model power units") +
                          ("^2" if config.prediction_loss == "mse" else ""),
                    "L1": "(MW/MW)^2",
                    **{f"L{i}": "(MW/MW)^2" if config.pattern_norm == "mse" else "MW/MW" for i in range(2, 6)},
                    **({"branch_penalty": "dimensionless"} if "branch_penalty" in config.loss_weights else {})}}
    write_json(output / "training.json", report)
    return report


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    cfg = checkpoint["configuration"]
    model = OPFNetwork(state["x_mean"], state["x_scale"], state["y_mean"], state["y_scale"],
                       cfg["hidden"], cfg["activation"])
    model.load_state_dict(state)
    return model.eval(), checkpoint


class Decoder(nn.Module):
    def __init__(self, feature_width, input_width, hidden=128, activation="tanh",
                 feature_layer="penultimate_hidden"):
        super().__init__()
        self.hidden = (hidden,) if isinstance(hidden, int) else tuple(hidden)
        if not self.hidden or any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in self.hidden):
            raise ValueError("Decoder requires positive integer hidden widths")
        if activation not in ("tanh", "relu", "silu") or feature_layer not in ("penultimate_hidden", "last_hidden"):
            raise ValueError("Unknown decoder activation or feature layer")
        self.activation_name, self.feature_layer = activation, feature_layer
        activation_class = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
        layers = []
        width = feature_width
        for next_width in self.hidden:
            layers.extend([nn.Linear(width, next_width), activation_class()])
            width = next_width
        layers.append(nn.Linear(width, input_width))
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        return self.layers(features)


RECONSTRUCTION_SCORES = {
    "normalized": ("normalized^2", "Mean squared normalized-input reconstruction error"),
    "mw2": ("MW^2", "Mean squared physical-input reconstruction error"),
    "mae_mw": ("MW", "Mean absolute physical-input reconstruction error"),
    "rmse_mw": ("MW", "Root mean squared physical-input reconstruction error"),
    "l1_mw": ("MW", "Summed absolute physical-input reconstruction error"),
    "linf_mw": ("MW", "Maximum absolute physical-input reconstruction error"),
}


def decoder_features(model, x, feature_layer="penultimate_hidden"):
    # Preserve legacy callers/test doubles with a single-argument features method.
    return model.features(x) if feature_layer == "penultimate_hidden" else model.features(x, feature_layer)


def decoder_feature_width(model, feature_layer):
    if feature_layer not in ("penultimate_hidden", "last_hidden"):
        raise ValueError("Unknown decoder feature layer")
    return model.hidden[-2] if feature_layer == "penultimate_hidden" else model.hidden[-1]


def reconstruction_errors(model, decoder, x, units="normalized"):
    error = decoder(decoder_features(model, x, getattr(decoder, "feature_layer", "penultimate_hidden"))) - (x-model.x_mean)/model.x_scale
    if units not in RECONSTRUCTION_SCORES:
        raise ValueError(f"Unknown reconstruction score: {units}")
    if units != "normalized":
        error = error*model.x_scale
    if units == "mae_mw":
        return error.abs().mean(dim=-1)
    if units == "rmse_mw":
        return error.square().mean(dim=-1).sqrt()
    if units == "l1_mw":
        return error.abs().sum(dim=-1)
    if units == "linf_mw":
        return error.abs().amax(dim=-1)
    return error.square().mean(dim=-1)


def calibrate_threshold(errors, quantile=.95):
    values = np.asarray(errors)
    if not 0 < quantile <= 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite nonempty validation errors and quantile in (0,1] required")
    return float(np.quantile(values, quantile, method="higher"))


def _decoder_calibration(errors, units, quantile, threshold):
    if units not in RECONSTRUCTION_SCORES:
        raise ValueError(f"Unknown reconstruction score: {units}")
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
        raise ValueError("Fixed threshold must be finite and nonnegative")
    calibrated = calibrate_threshold(errors, quantile)
    score_units, aggregation = RECONSTRUCTION_SCORES[units]
    return {
        "threshold": calibrated if threshold is None else float(threshold),
        "quantile": quantile if threshold is None else None,
        "calibration_method": "validation_quantile" if threshold is None else "explicit_fixed_threshold",
        "error_units": units, "score_units": score_units,
        "aggregation": aggregation + " across all ordered input buses",
        "validation_thresholds": [calibrate_threshold(errors, q) for q in (.05, .1, .25, .5, .75, .9, .95, .99, 1.)],
        "warning": "Reconstruction is not an accuracy/feasibility certificate. The paper's MW error norm is unspecified.",
    }


def train_decoder(dataset, model_path, output, *, epochs=100, batch_size=64, hidden=128,
                   learning_rate=.001, quantile=.95, units="normalized", seed=2026, threshold=None,
                   activation="tanh", feature_layer="penultimate_hidden", initializer="pytorch",
                   adam_variant="pytorch", adam_epsilon=1e-8, learning_rate_decay=0.,
                   checkpoint_selection="best_validation"):
    if (min(epochs, batch_size) < 1 or not np.isfinite(learning_rate) or learning_rate <= 0
            or units not in RECONSTRUCTION_SCORES):
        raise ValueError("Invalid decoder settings")
    if (initializer not in ("pytorch", "glorot_uniform") or adam_variant not in ("pytorch", "keras_v2")
            or checkpoint_selection not in ("best_validation", "last")
            or not np.isfinite([adam_epsilon, learning_rate_decay]).all()
            or adam_epsilon <= 0 or learning_rate_decay < 0):
        raise ValueError("Invalid decoder numerical settings")
    # Validate before creating artifacts, including explicit fixed thresholds.
    _decoder_calibration([0.], units, quantile, threshold)
    manifest, a = load_dataset(dataset)
    model, checkpoint = load_model(model_path)
    if dataset_identity(manifest) != checkpoint["dataset_id"]:
        raise ValueError("Decoder must use the OPF model's training/validation dataset")
    if not len(a["validation_x"]):
        raise ValueError("Decoder calibration requires validation inputs")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for p in model.parameters():
        p.requires_grad_(False)
    decoder = Decoder(decoder_feature_width(model, feature_layer), len(model.x_mean), hidden, activation, feature_layer)
    initialize_layers(decoder, initializer)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    optimizer = make_optimizer(decoder.parameters(), learning_rate=learning_rate, variant=adam_variant,
                               epsilon=adam_epsilon, decay=learning_rate_decay)
    train_x, val_x = tensor(a["train_x"]), tensor(a["validation_x"])
    with torch.no_grad():
        features = decoder_features(model, train_x, feature_layer)
    best, best_state, history = float("inf"), None, []
    started = time.perf_counter()
    best_epoch, updates = None, 0
    for epoch in range(epochs):
        decoder.train()
        for indices in torch.randperm(len(train_x)).split(batch_size):
            reconstructed = decoder(features[indices])
            loss = (reconstructed-(train_x[indices]-model.x_mean)/model.x_scale).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if adam_variant == "pytorch" and learning_rate_decay:
                # Set the rate for the next update; step zero used initial rate.
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate/(1+learning_rate_decay*(updates+1))
            updates += 1
        decoder.eval()
        with torch.no_grad():
            error = float(reconstruction_errors(model, decoder, val_x, units).mean())
        history.append({"epoch": epoch+1, "validation_error": error})
        if error < best:
            best, best_state = error, deepcopy(decoder.state_dict())
            best_epoch = epoch+1
    last_state = deepcopy(decoder.state_dict())
    selected_state, selected_epoch = ((best_state, best_epoch) if checkpoint_selection == "best_validation"
                                      else (last_state, epochs))
    decoder.load_state_dict(selected_state)
    with torch.no_grad():
        errors = reconstruction_errors(model, decoder, val_x, units).numpy()
    metadata = {"hidden": decoder.hidden[0] if len(decoder.hidden) == 1 else list(decoder.hidden),
                 "hidden_layers": list(decoder.hidden), "epochs": epochs, "batch_size": batch_size,
                  "learning_rate": learning_rate, "seed": seed,
                  "activation": activation, "initializer": initializer, "adam_variant": adam_variant,
                  "adam_epsilon": adam_epsilon, "learning_rate_decay": learning_rate_decay,
                  "checkpoint_selection": checkpoint_selection, "selected_epoch": selected_epoch,
                  "best_epoch": best_epoch, "best_validation_error": best,
                 **_decoder_calibration(errors, units, quantile, threshold),
                 "calibration_split": "validation", "opf_frozen": True,
                 "training_loss": "Mean squared normalized-input reconstruction error",
                 "feature_layer_name": feature_layer,
                 "feature_layer": ("penultimate hidden layer (hidden[-2])" if feature_layer == "penultimate_hidden"
                                   else "last hidden layer (hidden[-1])"),
                "model_hash": artifact_hash(model_path), "dataset_id": checkpoint["dataset_id"],
                "validation_scenario_ids": a["validation_scenario_ids"].tolist(),
                "versions": package_versions(), "training_seconds": time.perf_counter()-started}
    checkpoints = {}
    for name, state, epoch in (("decoder.pt", selected_state, selected_epoch),
                               ("decoder-best.pt", best_state, best_epoch),
                               ("decoder-last.pt", last_state, epochs)):
        decoder.load_state_dict(state)
        with torch.no_grad():
            state_errors = reconstruction_errors(model, decoder, val_x, units).numpy()
        state_metadata = {**metadata, "selected_epoch": epoch,
                          "checkpoint_selection": (checkpoint_selection if name == "decoder.pt" else
                                                   "best_validation" if name == "decoder-best.pt" else "last"),
                          **_decoder_calibration(state_errors, units, quantile, threshold)}
        torch.save({"state_dict": state, "metadata": state_metadata}, output / name)
        checkpoints[name] = {"sha256": artifact_hash(output/name), "epoch": epoch}
    write_json(output / "decoder.json", {**metadata, "history": history, "checkpoints": checkpoints})
    return metadata


def load_decoder(path, model_path, model):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    meta = checkpoint["metadata"]
    if meta["model_hash"] != artifact_hash(model_path):
        raise ValueError("Decoder belongs to a different OPF checkpoint")
    feature_layer = meta.get("feature_layer_name", "penultimate_hidden")
    decoder = Decoder(decoder_feature_width(model, feature_layer), len(model.x_mean),
                      meta.get("hidden_layers", meta["hidden"]), meta.get("activation", "tanh"), feature_layer)
    decoder.load_state_dict(checkpoint["state_dict"])
    return decoder.eval(), meta


def recalibrate_decoder(dataset, model_path, decoder_path, output, *, quantile=.95, units=None, threshold=None):
    """Explicit threshold-only operation; never fit on test or production data."""
    manifest, arrays = load_dataset(dataset)
    model, checkpoint = load_model(model_path)
    decoder, metadata = load_decoder(decoder_path, model_path, model)
    if dataset_identity(manifest) != checkpoint["dataset_id"]:
        raise ValueError("Calibration must use the model's validation dataset")
    validation_x = tensor(arrays["validation_x"])
    units = units or metadata["error_units"]
    with torch.no_grad():
        errors = reconstruction_errors(model, decoder, validation_x, units).numpy()
    calibration = _decoder_calibration(errors, units, quantile, threshold)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {**metadata, **calibration,
                "calibration_split": "validation",
                "validation_scenario_ids": arrays["validation_scenario_ids"].tolist()}
    torch.save({"state_dict": decoder.state_dict(), "metadata": metadata}, output / "decoder.pt")
    write_json(output / "decoder.json", metadata)
    return metadata
