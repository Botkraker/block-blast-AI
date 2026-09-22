"""Summary statistics, bootstrap confidence intervals and agent comparison."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from blockblast.evaluation.evaluate import EvalResult


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    metric: str
    mean_a: float
    mean_b: float
    ratio: float
    diff_ci: tuple[float, float]


def summarize(values: Sequence[float]) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        raise ValueError("no values")
    return {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_resamples, v.size))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _diff_ci(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Unpaired bootstrap CI of mean(a) − mean(b) (conservative even with shared seeds)."""
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, a.size, size=(n_resamples, a.size))].mean(axis=1)
    mb = b[rng.integers(0, b.size, size=(n_resamples, b.size))].mean(axis=1)
    lo, hi = np.percentile(ma - mb, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def compare(a: EvalResult, b: EvalResult, metric: str = "score") -> ComparisonResult:
    va = np.array([getattr(e, metric) for e in a.episodes], dtype=np.float64)
    vb = np.array([getattr(e, metric) for e in b.episodes], dtype=np.float64)
    mean_a, mean_b = float(va.mean()), float(vb.mean())
    ratio = mean_a / mean_b if mean_b else float("inf")
    return ComparisonResult(metric, mean_a, mean_b, ratio, _diff_ci(va, vb))
