from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blockblast.agents import GreedyAgent, RandomAgent
from blockblast.evaluation import (
    EpisodeStats,
    EvalResult,
    bootstrap_ci,
    compare,
    evaluate_agent,
    summarize,
)
from blockblast.utils.seeding import spawn_seeds


def result(name: str, scores: list[int]) -> EvalResult:
    return EvalResult(
        name, tuple(EpisodeStats(i, s, s // 10, s, 0, 0) for i, s in enumerate(scores))
    )


def test_summarize() -> None:
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s["mean"] == 2.5 and s["min"] == 1.0 and s["max"] == 4.0 and s["median"] == 2.5
    with pytest.raises(ValueError):
        summarize([])


def test_bootstrap_ci_contains_mean_and_is_seeded() -> None:
    v = np.random.default_rng(0).normal(10, 2, 500).tolist()
    lo, hi = bootstrap_ci(v, n_resamples=2000)
    assert lo < float(np.mean(v)) < hi and hi - lo < 1.0
    assert bootstrap_ci(v, n_resamples=2000) == (lo, hi)


def test_compare() -> None:
    c = compare(result("a", [200] * 50 + [220] * 50), result("b", [100] * 50 + [110] * 50))
    assert c.ratio == pytest.approx(2.0) and c.diff_ci[0] > 0
    zero = compare(result("a", [1, 2]), result("b", [0, 0]))
    assert zero.ratio == float("inf")


def test_eval_json_round_trip_and_recording(tmp_path: Path) -> None:
    r = evaluate_agent(RandomAgent(0), range(1_000_000, 1_000_004), {}, tmp_path / "rep", 2)
    assert len(list((tmp_path / "rep").glob("*.json"))) == 2
    r.to_json(tmp_path / "e.json")
    assert EvalResult.from_json(tmp_path / "e.json") == r
    s = r.summary()
    assert s["n_episodes"] == 4 and s["score_ci_lo"] <= s["score_mean"] <= s["score_ci_hi"]


def test_greedy_beats_random_on_small_sample() -> None:
    seeds = range(1_000_000, 1_000_030)
    g = evaluate_agent(GreedyAgent(), seeds, {})
    r = evaluate_agent(RandomAgent(0), seeds, {})
    assert compare(g, r).ratio > 1.0


def test_spawn_seeds() -> None:
    a = spawn_seeds(0, 5)
    assert a == spawn_seeds(0, 5) and len(set(a)) == 5
