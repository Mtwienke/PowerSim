"""Explicit Keras-v2 Adam equations for the opt-in reconstruction protocol.

This is a mathematical reimplementation, not vendored TensorFlow source or a
claim of cross-framework bitwise identity. Historical PyTorch Adam is unchanged.
"""
import math

import torch


class KerasV2Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=.001, betas=(.9, .999), epsilon=1e-7, decay=2e-6):
        if (not all(math.isfinite(v) for v in (lr, *betas, epsilon, decay))
                or lr <= 0 or epsilon <= 0 or decay < 0 or any(not 0 <= b < 1 for b in betas)):
            raise ValueError("Invalid Keras-v2 Adam settings")
        super().__init__(params, dict(lr=lr, initial_lr=lr, betas=betas, epsilon=epsilon, decay=decay))
        self.iterations = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        step = self.iterations+1
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            rate = group["initial_lr"]/(1+group["decay"]*self.iterations)
            group["lr"] = rate
            alpha = rate*math.sqrt(1-beta2**step)/(1-beta1**step)
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise ValueError("Reconstruction Keras-v2 Adam supports dense gradients only")
                state = self.state[parameter]
                if not state:
                    state["m"] = torch.zeros_like(parameter)
                    state["v"] = torch.zeros_like(parameter)
                state["m"].mul_(beta1).add_(gradient, alpha=1-beta1)
                state["v"].mul_(beta2).addcmul_(gradient, gradient, value=1-beta2)
                # epsilon-hat is outside the uncorrected sqrt(v), not sqrt(vhat).
                parameter.addcdiv_(state["m"], state["v"].sqrt().add_(group["epsilon"]), value=-alpha)
        self.iterations += 1
        return loss

    def state_dict(self):
        return {**super().state_dict(), "iterations": self.iterations}

    def load_state_dict(self, state_dict):
        values = dict(state_dict)
        iterations = values.pop("iterations")
        super().load_state_dict(values)
        self.iterations = iterations
