"""Sensitivity-label conventions from the Gao public examples."""
import numpy as np


def finite_jacobians(values):
    values = np.asarray(values, dtype=float)
    if (values.ndim != 3 or min(values.shape[1:]) < 1
            or not np.isfinite(values).all()):
        raise ValueError("Finite sample x generator x input Jacobians required")
    return values


def example_sensitivity_mask(jacobians):
    """Exact signed-max/nonzero-sum predicate on finite supplied Jacobians.

    Not max(abs(J)), not clipping, and not a supervised-sample exclusion rule.
    """
    j = finite_jacobians(jacobians)
    return (j.sum(axis=(1, 2)) != 0) & (j.max(axis=(1, 2)) < 10)
