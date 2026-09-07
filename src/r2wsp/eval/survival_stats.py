from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class LogRankResult:
    p_value: float | None
    chi_square: float | None
    threshold: float | None
    group_high: int
    group_low: int
    notes: list[str]


def _as_1d_float_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64).reshape(-1)
    return arr


def logrank_p_value_from_risk(
    event_times: Iterable[float] | np.ndarray,
    censorships: Iterable[float] | np.ndarray,
    risk_scores: Iterable[float] | np.ndarray,
) -> LogRankResult:
    times = _as_1d_float_array(event_times)
    cens = _as_1d_float_array(censorships)
    risk = _as_1d_float_array(risk_scores)

    notes: list[str] = []
    if not (len(times) == len(cens) == len(risk)):
        raise ValueError("event_times, censorships, and risk_scores must have the same length")
    if len(times) == 0:
        return LogRankResult(None, None, None, 0, 0, ["empty"])

    valid = np.isfinite(times) & np.isfinite(cens) & np.isfinite(risk)
    if not np.all(valid):
        notes.append(f"dropped_invalid={int((~valid).sum())}")
    times = times[valid]
    cens = cens[valid]
    risk = risk[valid]
    if len(times) < 2:
        notes.append("too_few_samples")
        return LogRankResult(None, None, None, 0, 0, notes)

    threshold = float(np.median(risk))
    high = risk >= threshold
    if np.all(high) or np.all(~high):
        high = risk > threshold
    n_high = int(high.sum())
    n_low = int((~high).sum())
    if n_high == 0 or n_low == 0:
        notes.append("degenerate_risk_split")
        return LogRankResult(None, None, threshold, n_high, n_low, notes)

    event_mask = cens == 0
    event_times_unique = np.unique(times[event_mask])
    if len(event_times_unique) == 0:
        notes.append("no_events")
        return LogRankResult(None, None, threshold, n_high, n_low, notes)

    obs_minus_exp = 0.0
    variance = 0.0
    for t in event_times_unique:
        at_risk = times >= t
        at_risk_high = at_risk & high
        at_risk_low = at_risk & (~high)

        n1 = int(at_risk_high.sum())
        n0 = int(at_risk_low.sum())
        n = n1 + n0
        if n <= 1:
            continue

        is_event_at_t = (times == t) & event_mask
        d1 = int((is_event_at_t & high).sum())
        d0 = int((is_event_at_t & (~high)).sum())
        d = d1 + d0
        if d == 0:
            continue

        expected_high = d * (n1 / n)
        if n > 1:
            variance_t = (n1 * n0 * d * (n - d)) / ((n * n) * (n - 1))
        else:
            variance_t = 0.0
        obs_minus_exp += d1 - expected_high
        variance += variance_t

    if variance <= 0:
        notes.append("non_positive_variance")
        return LogRankResult(None, None, threshold, n_high, n_low, notes)

    chi_square = float((obs_minus_exp * obs_minus_exp) / variance)
    p_value = float(math.erfc(math.sqrt(chi_square / 2.0)))
    return LogRankResult(p_value, chi_square, threshold, n_high, n_low, notes)
