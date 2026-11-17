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
from .network import digest


def tensor(value):
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def artifact_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_identity(manifest):
    return digest({"configuration": manifest["configuration"], "model_id": manifest["model_id"],
                   "bus_ids": manifest["bus_ids"], "generator_ids": manifest["generator_ids"]})


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

    def features(self, x):
        """Penultimate HIDDEN layer: hidden[-2], not the output or last hidden."""
        z = (x-self.x_mean)/self.x_scale
        for layer in self.layers[:-1]:
            z = self.activation(layer(z))
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


def masked_mean_square(values, mask):
    mask = mask.to(dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(values)
    return (values.square()*expanded).sum()/expanded.sum().clamp_min(1)


def jacobian_losses(j, reference_j, reference_valid, active_generator, active_branch, m_g, m_d):
    return {
        "L1": masked_mean_square(j-reference_j, reference_valid),
        "L2": (j.sum(dim=1)-1).square().mean(),
        "L3": masked_mean_square(j, active_generator),
        "L4": masked_mean_square(torch.einsum("bg,ngd->nbd", m_g, j)-m_d, active_branch),
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

    def __post_init__(self):
        if (not all(isinstance(v, int) and v > 0 for v in (self.epochs, self.batch_size))
                or not np.isfinite([self.learning_rate, self.perturb_std_normalized]).all()
                or self.learning_rate <= 0 or self.perturb_std_normalized < 0):
            raise ValueError("Invalid training settings")
        if len(self.hidden) < 2 or not all(isinstance(h, int) and h > 0 for h in self.hidden):
            raise ValueError("At least two positive integer hidden widths required")
        if self.activation not in ("tanh", "relu", "silu"):
            raise ValueError("Unsupported activation")
        if set(self.loss_weights) - {f"L{i}" for i in range(6)}:
            raise ValueError("Unknown loss term")
        if not any(v > 0 for v in self.loss_weights.values()) or any(not np.isfinite(v) or v < 0 for v in self.loss_weights.values()):
            raise ValueError("Loss weights must be finite, nonnegative, with at least one positive")


def train_model(dataset, output, config: TrainConfig | None = None):
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
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = OPFNetwork(a["x_mean"], a["x_scale"], a["y_mean"], a["y_scale"], config.hidden, config.activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # Intentionally address only TRAIN and VALIDATION, never held-out labels.
    x, y = tensor(a["train_x"]), tensor(a["train_y"])
    reference_j = tensor(a["train_jacobian"])
    valid_j = tensor(a["train_jacobian_valid"]).bool()
    active_g = tensor(a["train_lower"] | a["train_upper"]).bool()
    active_b = tensor(a["train_active_branch"]).bool()
    mg, md = tensor(a["m_g"]), tensor(a["m_d"])
    history, best, best_state = [], float("inf"), None
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        totals = {f"L{i}": 0.0 for i in range(6)}
        for indices in torch.randperm(len(x)).split(config.batch_size):
            bx = x[indices].detach().requires_grad_(True)
            terms = {"L0": (model(bx)-y[indices]).square().mean()}
            if any(config.loss_weights.get(f"L{i}", 0) > 0 for i in range(1, 5)):
                j = physical_jacobian(model, bx)
                terms.update(jacobian_losses(j, reference_j[indices], valid_j[indices],
                                             active_g[indices], active_b[indices], mg, md))
            if config.loss_weights.get("L5", 0) > 0:
                perturbed = (bx.detach()+torch.randn_like(bx)*model.x_scale*config.perturb_std_normalized).requires_grad_(True)
                jp = physical_jacobian(model, perturbed)
                # No local L3/L4 masks are extrapolated to unlabeled points.
                terms["L5"] = (jp.sum(dim=1)-1).square().mean()
            loss = sum(weight*terms[key] for key, weight in config.loss_weights.items() if weight > 0)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for key, value in terms.items():
                totals[key] += float(value.detach())*len(indices)/len(x)
        model.eval()
        with torch.no_grad():
            validation = float((model(tensor(a["validation_x"]))-tensor(a["validation_y"])).square().mean())
        history.append({"epoch": epoch+1, **totals, "validation_mse_mw2": validation})
        if validation < best:
            best, best_state = validation, deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    checkpoint = {"state_dict": model.state_dict(), "configuration": asdict(config),
                  "model_id": manifest["model_id"], "dataset_id": dataset_identity(manifest),
                  "bus_ids": manifest["bus_ids"], "generator_ids": manifest["generator_ids"],
                  "physics_mode": "dc", "versions": package_versions(),
                  "training_scenario_ids": a["train_scenario_ids"].tolist(),
                  "validation_scenario_ids": a["validation_scenario_ids"].tolist(),
                  "training_input_ids": training_input_ids, "validation_input_ids": validation_input_ids}
    torch.save(checkpoint, output / "model.pt")
    report = {"configuration": asdict(config), "training_seconds": time.perf_counter()-started,
              "selection_split": "validation", "best_validation_mse_mw2": best,
              "history": history, "versions": package_versions(),
              "loss_units": {"L0": "MW^2", **{f"L{i}": "(MW/MW)^2" for i in range(1, 6)}}}
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
    def __init__(self, feature_width, input_width, hidden=128):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(feature_width, hidden), nn.Tanh(), nn.Linear(hidden, input_width))

    def forward(self, features):
        return self.layers(features)


def reconstruction_errors(model, decoder, x, units="normalized"):
    error = decoder(model.features(x)) - (x-model.x_mean)/model.x_scale
    if units == "mw2":
        error = error*model.x_scale
    elif units != "normalized":
        raise ValueError("Decoder error units must be normalized or mw2")
    return error.square().mean(dim=-1)


def calibrate_threshold(errors, quantile=.95):
    values = np.asarray(errors)
    if not 0 < quantile <= 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite nonempty validation errors and quantile in (0,1] required")
    return float(np.quantile(values, quantile, method="higher"))


def train_decoder(dataset, model_path, output, *, epochs=100, batch_size=64, hidden=128,
                  learning_rate=.001, quantile=.95, units="normalized", seed=2026):
    if min(epochs, batch_size, hidden) < 1 or learning_rate <= 0:
        raise ValueError("Invalid decoder settings")
    manifest, a = load_dataset(dataset)
    model, checkpoint = load_model(model_path)
    if dataset_identity(manifest) != checkpoint["dataset_id"]:
        raise ValueError("Decoder must use the OPF model's training/validation dataset")
    if not len(a["validation_x"]):
        raise ValueError("Decoder calibration requires validation inputs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for p in model.parameters():
        p.requires_grad_(False)
    decoder = Decoder(model.hidden[-2], len(model.x_mean), hidden)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    train_x, val_x = tensor(a["train_x"]), tensor(a["validation_x"])
    with torch.no_grad():
        features = model.features(train_x)
    best, best_state, history = float("inf"), None, []
    started = time.perf_counter()
    for epoch in range(epochs):
        decoder.train()
        for indices in torch.randperm(len(train_x)).split(batch_size):
            reconstructed = decoder(features[indices])
            loss = (reconstructed-(train_x[indices]-model.x_mean)/model.x_scale).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        decoder.eval()
        with torch.no_grad():
            error = float(reconstruction_errors(model, decoder, val_x, units).mean())
        history.append({"epoch": epoch+1, "validation_error": error})
        if error < best:
            best, best_state = error, deepcopy(decoder.state_dict())
    decoder.load_state_dict(best_state)
    with torch.no_grad():
        errors = reconstruction_errors(model, decoder, val_x, units).numpy()
    threshold = calibrate_threshold(errors, quantile)
    metadata = {"hidden": hidden, "epochs": epochs, "batch_size": batch_size,
                "learning_rate": learning_rate, "seed": seed, "quantile": quantile,
                "threshold": threshold, "error_units": units,
                "aggregation": "mean squared reconstruction error across all ordered input buses",
                "calibration_split": "validation", "opf_frozen": True,
                "feature_layer": "penultimate hidden layer (hidden[-2])",
                "model_hash": artifact_hash(model_path), "dataset_id": checkpoint["dataset_id"],
                "validation_scenario_ids": a["validation_scenario_ids"].tolist(),
                "versions": package_versions(), "training_seconds": time.perf_counter()-started}
    torch.save({"state_dict": decoder.state_dict(), "metadata": metadata}, output / "decoder.pt")
    write_json(output / "decoder.json", {**metadata, "history": history})
    return metadata


def load_decoder(path, model_path, model):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    meta = checkpoint["metadata"]
    if meta["model_hash"] != artifact_hash(model_path):
        raise ValueError("Decoder belongs to a different OPF checkpoint")
    decoder = Decoder(model.hidden[-2], len(model.x_mean), meta["hidden"])
    decoder.load_state_dict(checkpoint["state_dict"])
    return decoder.eval(), meta


def recalibrate_decoder(dataset, model_path, decoder_path, output, *, quantile=.95, units=None):
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
    threshold = calibrate_threshold(errors, quantile)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {**metadata, "quantile": quantile, "threshold": threshold, "error_units": units,
                "calibration_split": "validation",
                "validation_scenario_ids": arrays["validation_scenario_ids"].tolist()}
    torch.save({"state_dict": decoder.state_dict(), "metadata": metadata}, output / "decoder.pt")
    write_json(output / "decoder.json", metadata)
    return metadata
