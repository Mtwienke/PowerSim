"""Dispatch accuracy with explicit source/scenario denominators; no ML dependency."""
import numpy as np


ACCURACY_DEFINITIONS = {
    "generator_accuracy_lt_1mw": "Individual finite source errors < 1 MW / all requested source outputs.",
    "joint_accuracy_lt_1mw": "Complete scenarios with every source error < 1 MW / requested scenarios.",
    "success_rate_1mw": "Legacy joint-scenario criterion: every source error <= 1 MW.",
    "mae_mw": "Mean over complete finite prediction/reference rows; unavailable rows are not zero errors.",
    "paper_comparison": "Generator-wise aggregation is inferred from Gao et al. Table V; Eq. (26) leaves vector reduction unspecified.",
}


def accuracy(prediction, reference, *, denominator=None):
    """Nonfinite outputs count as failures, never as dropped accuracy denominators.

    Keep the historical inclusive joint criterion while exposing strict '< 1 MW'
    metrics matching the inequality in the paper. Errors retain legacy behavior:
    MAE/RMSE include only complete finite rows.
    """
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if reference.shape == prediction.shape == (0,):
        prediction, reference = prediction.reshape(0, 0), reference.reshape(0, 0)
    if reference.ndim != 2 or prediction.shape != reference.shape:
        raise ValueError("Prediction and reference must have matching scenario-by-source shapes")
    n, ng = reference.shape
    if n and not ng:
        raise ValueError("At least one source is required")
    count = n if denominator is None else denominator
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < n:
        raise ValueError("Accuracy denominator must be an integer >= evaluated scenarios")
    with np.errstate(invalid="ignore"):
        error = prediction-reference
    finite_outputs = np.isfinite(error)
    complete = finite_outputs.all(axis=1)
    finite = error[complete]
    strict = finite_outputs & (np.abs(error) < 1)
    inclusive = finite_outputs & (np.abs(error) <= 1)
    scalar_hits = int(strict.sum())
    joint_hits = int((complete & strict.all(axis=1)).sum())
    legacy_hits = int((complete & inclusive.all(axis=1)).sum())
    scalar_count = int(count*ng)
    return {
        "mae_mw": float(np.abs(finite).mean()) if finite.size else None,
        "rmse_mw": float(np.sqrt(np.mean(finite**2))) if finite.size else None,
        "maximum_error_mw": float(np.abs(finite).max()) if finite.size else None,
        "success_rate_1mw": legacy_hits/count if count else None,
        "joint_success_rate_1mw": legacy_hits/count if count else None,
        "accuracy_samples": int(complete.sum()),
        "evaluated_scenarios": n,
        "sources_per_scenario": ng,
        "generator_accuracy_lt_1mw": scalar_hits/scalar_count if scalar_count else None,
        "joint_accuracy_lt_1mw": joint_hits/count if count else None,
        "generator_accuracy_count": scalar_hits,
        "joint_accuracy_count": joint_hits,
        "generator_accuracy_denominator": scalar_count,
        "joint_accuracy_denominator": int(count),
        "generator_finite_outputs": int(finite_outputs.sum()),
        "per_generator_accuracy_lt_1mw": (strict.sum(axis=0)/count).tolist() if count else [None]*ng,
    }


def selection_curve(scores, prediction, reference, thresholds):
    """Describe held-out risk/coverage at thresholds fixed elsewhere (validation).

    No threshold is selected here. The reject-all row has undefined conditional
    accuracy, rather than a misleading 100%. Source counts are not independent
    statistical trials; independent scenario counts are included in each row.
    """
    scores = np.asarray(scores)
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if scores.shape != (len(reference),):
        raise ValueError("One reconstruction score is required per scenario")
    values = sorted(set(float(t) for t in thresholds))
    if any(not np.isfinite(t) or t < 0 for t in values):
        raise ValueError("Selection thresholds must be finite and nonnegative")
    rows = []
    for threshold in [None, *values]:
        accepted = (np.isfinite(scores) & (scores <= threshold) if threshold is not None
                    else np.zeros(len(scores), dtype=bool))
        rows.append({
            "threshold": threshold,
            "accepted_scenarios": int(accepted.sum()),
            "neural_coverage": float(accepted.mean()) if len(accepted) else None,
            "accepted_accuracy": accuracy(prediction[accepted], reference[accepted]),
        })
    return rows
