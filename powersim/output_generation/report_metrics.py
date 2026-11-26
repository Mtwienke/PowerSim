"""Vectorized, physical-unit metrics for the frozen DCOPF report."""
from dataclasses import dataclass

import numpy as np

from .metrics import accuracy


# Solver-validation tolerances, not operational security criteria.
BALANCE_MW = 1e-4
BOUND_MW = 1e-3
REFERENCE_COST_EPS = 1e-9


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def errors(prediction, reference):
    prediction, reference = np.asarray(prediction, float), np.asarray(reference, float)
    if prediction.shape != reference.shape or reference.ndim != 2 or not reference.shape[1]:
        raise ValueError("Expected matching sample × source matrices")
    with np.errstate(invalid="ignore"):
        result = np.abs(prediction - reference)
    return np.where(np.isfinite(result), result, np.nan)


def accuracy_summary(prediction, reference, *, requested):
    error = errors(prediction, reference)
    complete = np.isfinite(error).all(axis=1)
    # Historical reductions and compatibility fields.
    summary = accuracy(np.asarray(prediction, float), np.asarray(reference, float), denominator=requested)
    summary.pop("per_generator_accuracy_lt_1mw")
    result = {
        **{key: value for key, value in summary.items() if "accuracy" not in key and "success_rate" not in key},
        "legacy_joint_le_1mw": fraction((complete & (error <= 1).all(axis=1)).sum(), len(error)),
    }
    for t, suffix in ((.5, "0p5mw"), (1., "1mw"), (2., "2mw")):
        hits = error < t
        quantities = {"element": hits.sum(), "max": (complete & hits.all(axis=1)).sum(),
                      "mean": (complete & (error.mean(axis=1) < t)).sum()}
        for kind, count in quantities.items():
            width = error.shape[1] if kind == "element" else 1
            key = f"{kind}_lt_{suffix}"
            result[key + "_hits"] = int(count)
            result[key] = fraction(count, len(error)*width)
            result[key + "_all_attempts"] = fraction(count, requested*width)
    finite = error[complete]
    for q, name in ((.5, "median"), (.95, "p95"), (.99, "p99")):
        result[f"{name}_absolute_error_mw"] = float(np.quantile(finite, q)) if finite.size else None
    result["complete_error_rows"] = int(complete.sum())
    return result


@dataclass(frozen=True)
class Physics:
    """All matrices use ordered MW arrays, including slack in the proposal."""
    m_g: np.ndarray
    m_d: np.ndarray
    constant: np.ndarray
    shunt_mw: float
    gen_min: np.ndarray
    gen_max: np.ndarray
    branch_min: np.ndarray
    branch_max: np.ndarray
    cp0: np.ndarray
    cp1: np.ndarray
    cp2: np.ndarray

    def evaluate(self, generation, x, reference_cost):
        generation, x, reference_cost = map(lambda a: np.asarray(a, float), (generation, x, reference_cost))
        if (generation.ndim != 2 or x.ndim != 2 or generation.shape != (len(x), len(self.gen_min))
                or x.shape[1] != self.m_d.shape[1] or reference_cost.shape != (len(x),)):
            raise ValueError("Dispatch, load or cost coverage mismatch")
        complete = np.isfinite(generation).all(axis=1) & np.isfinite(x).all(axis=1)
        generation = np.where(complete[:, None], generation, np.nan)
        # The project PTDF fixes a reference bus: unbalanced proposals get a
        # *diagnostic* balancing injection there, never a repaired dispatch.
        flow = generation @ self.m_g.T - x @ self.m_d.T + self.constant
        balance = generation.sum(axis=1) - x.sum(axis=1) - self.shunt_mw
        low = np.maximum(self.gen_min - generation, 0)
        high = np.maximum(generation - self.gen_max, 0)
        line = np.maximum.reduce((flow - self.branch_max, self.branch_min - flow, np.zeros_like(flow)))
        balance_ok = complete & (np.abs(balance) <= BALANCE_MW)
        gen_ok = complete & ~(np.maximum(low, high) > BOUND_MW).any(axis=1)
        line_ok = complete & ~(line > BOUND_MW).any(axis=1)
        cost = (self.cp0 + generation * (self.cp1 + generation*self.cp2)).sum(axis=1)
        delta = cost - reference_cost
        base = np.abs(reference_cost)
        relative = np.divide(delta, base, out=np.full(len(x), np.nan), where=base > REFERENCE_COST_EPS)
        return {
            "evaluated": complete, "flow_mw": flow, "balance_residual_mw": balance,
            "gen_lower_violation_mw": low, "gen_upper_violation_mw": high, "line_violation_mw": line,
            "balance_pass": balance_ok, "generator_pass": gen_ok, "line_pass": line_ok,
            "feasible": balance_ok & gen_ok & line_ok,
            "predicted_cost_eur_h": cost, "reference_cost_eur_h": reference_cost,
            "cost_difference_eur_h": delta, "relative_cost_difference": relative,
        }


def violations_summary(excess, prefix, n, *, tolerance=BOUND_MW):
    events = excess > tolerance
    values = excess[events]
    return {
        f"{prefix}_violation_samples": int(events.any(axis=1).sum()),
        f"{prefix}_violation_rate": fraction(events.any(axis=1).sum(), n),
        f"{prefix}_violation_events": int(events.sum()),
        f"{prefix}_distinct_violating_assets": int(events.any(axis=0).sum()),
        f"{prefix}_mean_violation_mw": float(values.mean()) if values.size else None,
        f"{prefix}_maximum_violation_mw": float(values.max()) if values.size else (0. if n else None),
        f"{prefix}_aggregate_violation_mw": float(values.sum()) if n else None,
    }


def physics_summary(data):
    mask = data["evaluated"]
    checked = int(mask.sum())
    absolute = np.abs(data["balance_residual_mw"][mask])
    info = {"physics_evaluated": checked, "physics_missing": int(len(mask)-checked)}
    for key in ("balance_pass", "generator_pass", "line_pass", "feasible"):
        info[key + "_count"] = int(data[key].sum())
        info[key + "_rate"] = fraction(data[key].sum(), len(mask))
    for t, name in ((BALANCE_MW, "numerical"), (.01, "0p01mw"), (.1, "0p1mw"), (1., "1mw")):
        info[f"balance_exceeds_{name}_rate"] = fraction((absolute > t).sum(), checked)
    for key, reducer in (("mean", np.mean), ("max", np.max), ("p95", lambda a: np.quantile(a, .95))):
        info[f"balance_absolute_{key}_mw"] = float(reducer(absolute)) if checked else None
    low, high = data["gen_lower_violation_mw"], data["gen_upper_violation_mw"]
    for prefix, matrix in (
        ("generator", np.maximum(low, high)), ("generator_lower", low), ("generator_upper", high),
        ("line", data["line_violation_mw"]), ("line_limit_plus_1mw", np.maximum(data["line_violation_mw"]-1, 0)),
    ):
        info.update(violations_summary(matrix, prefix, checked, tolerance=0. if prefix.endswith("1mw") else BOUND_MW))
    for label, subset in (("feasible", data["feasible"]), ("infeasible", mask & ~data["feasible"])):
        for k in ("cost_difference_eur_h", "relative_cost_difference"):
            v = data[k][subset & np.isfinite(data[k])]
            info[f"{label}_{k}_count"] = len(v)
            for name, reducer in (("mean", np.mean), ("median", np.median), ("max", np.max), ("min", np.min)):
                info[f"{label}_{k}_{name}"] = float(reducer(v)) if v.size else None
    return info


def gate_mask(scores, complete, threshold):
    return np.isfinite(scores) & complete & (scores <= threshold) if threshold is not None else np.zeros_like(complete)


def gate_classification(accepted, reliable, known):
    """Positive class = unreliable; positive decision = reject."""
    accepted, reliable, known = map(lambda a: np.asarray(a, bool), (accepted, reliable, known))
    if accepted.ndim != 1 or accepted.shape != reliable.shape or accepted.shape != known.shape:
        raise ValueError("Expected aligned classification vectors")
    tp = int((~accepted & ~reliable & known).sum())
    fn = int((accepted & ~reliable & known).sum())
    fp = int((~accepted & reliable & known).sum())
    tn = int((accepted & reliable & known).sum())
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn, "known": tp+fn+fp+tn,
        "unreliable_prevalence": fraction(tp+fn, tp+fn+fp+tn),
        "false_acceptance_rate": fraction(fn, tp+fn), "false_rejection_rate": fraction(fp, fp+tn),
        "precision": fraction(tp, tp+fp), "recall": fraction(tp, tp+fn),
    }


def ranking_metrics(scores, unreliable, known):
    """AUROC (trapezoid) and AUPRC (non-interpolated average precision); ties grouped."""
    mask = known & np.isfinite(scores)
    y = unreliable[mask]
    n, npos = len(y), int(y.sum())
    result = {"ranking_n": n, "ranking_positive": npos, "ranking_prevalence": fraction(npos, n),
              "auroc": None, "auprc_ap": None}
    if 0 < npos < n:
        s = scores[mask]
        idx = np.argsort(-s, kind="stable")
        last = np.r_[np.flatnonzero(np.diff(s[idx]) != 0), n-1]
        tp = np.cumsum(y[idx])[last]
        tpr = np.r_[0., tp/npos]
        fpr = np.r_[0., (last+1-tp)/(n-npos)]
        result["auroc"] = float(np.sum(np.diff(fpr)*(tpr[:-1] + tpr[1:])/2))
        result["auprc_ap"] = float(np.sum(np.diff(tpr)*tp/(last+1)))
    return result


def change(value, baseline):
    """Return absolute percentage-point and relative percent change."""
    if value is None or baseline is None:
        return None, None
    return 100*(value-baseline), 100*(value-baseline)/baseline if baseline else None


def speedup(numerical_seconds, method_seconds, numerical_workload, method_workload):
    if numerical_workload != method_workload:
        raise ValueError("Speedup requires identical workload IDs")
    if not np.isfinite([numerical_seconds, method_seconds]).all() or min(numerical_seconds, method_seconds) <= 0:
        raise ValueError("Speedup requires finite positive measured durations")
    return numerical_seconds/method_seconds
