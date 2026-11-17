"""Empirical decoder selection. Rejection invokes OPF, never hidden repair."""
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from .ml import reconstruction_errors, tensor
from .network import digest


@dataclass(frozen=True)
class HybridResult:
    status: str
    accepted: bool
    raw_prediction_mw: tuple[float, ...] | None
    generation_mw: tuple[float, ...] | None
    reconstruction_error: float | None
    elapsed_seconds: float
    fallback_seconds: float
    diagnostics: tuple[str, ...] = ()


class HybridPredictor:
    def __init__(self, model, decoder, metadata, generator_ids):
        self.model, self.decoder = model.eval(), decoder.eval()
        self.metadata, self.generator_ids = metadata, tuple(generator_ids)
        if len(self.generator_ids) != len(model.y_mean) or len(set(self.generator_ids)) != len(self.generator_ids):
            raise ValueError("Hybrid source coverage/ordering mismatch")
        if not np.isfinite(metadata["threshold"]) or metadata["threshold"] < 0:
            raise ValueError("Decoder threshold must be finite and nonnegative")

    def predict(self, net_load_mw, fallback, *, production_log=None):
        """fallback is called without arguments and MUST solve this same input.

        Held-out evaluation never supplies production_log. Retraining is a
        separate, explicit operation; this function never updates either model.
        """
        started = time.perf_counter()
        x = np.asarray(net_load_mw, dtype=float)
        if x.shape != (len(self.model.x_mean),) or not np.isfinite(x).all():
            return HybridResult("failed", False, None, None, None, time.perf_counter()-started, 0,
                                ("Invalid nodal input coverage/values",))
        with torch.no_grad():
            tx = tensor(x[None])
            prediction = self.model(tx)[0].numpy()
            error = float(reconstruction_errors(self.model, self.decoder, tx, self.metadata["error_units"])[0])
        finite = bool(np.isfinite(prediction).all() and np.isfinite(error))
        raw = tuple(float(p) for p in prediction) if np.isfinite(prediction).all() else None
        if finite and error <= self.metadata["threshold"]:
            return HybridResult("learned", True, raw, raw, error, time.perf_counter()-started, 0)
        fallback_started = time.perf_counter()
        try:
            result = fallback()
            if (result.status != "ok" or not result.converged or not result.metrics.get("feasible")
                    or set(result.generation_mw) != set(self.generator_ids)
                    or not np.isfinite(list(result.generation_mw.values())).all()):
                diagnostics = result.diagnostics or ("Fallback did not return a complete validated feasible solution",)
                generation, status = None, "failed"
            else:
                generation = tuple(result.generation_mw[key] for key in self.generator_ids)
                status, diagnostics = "fallback", ()
        except Exception as exc:
            generation, status, diagnostics = None, "failed", (f"{type(exc).__name__}: {exc}",)
        fallback_seconds = time.perf_counter()-fallback_started
        if production_log is not None:
            path = Path(production_log)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as log:
                log.write(json.dumps({"scenario_id": digest(x.tolist()), "net_load_mw": x.tolist(),
                                      "raw_prediction_mw": raw, "reconstruction_error": error if np.isfinite(error) else None,
                                      "model_hash": self.metadata.get("model_hash"),
                                      "generator_ids": self.generator_ids, "error_units": self.metadata["error_units"],
                                      "threshold": self.metadata["threshold"], "fallback_status": status,
                                      "diagnostics": diagnostics}, allow_nan=False) + "\n")
        return HybridResult(status, False, raw, generation, error if np.isfinite(error) else None,
                            time.perf_counter()-started, fallback_seconds, tuple(diagnostics))
