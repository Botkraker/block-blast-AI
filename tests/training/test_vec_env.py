"""ParallelVecEnv must behave exactly like DummyVecEnv, without torch in the workers."""

from __future__ import annotations

import numpy as np
import pytest
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported

from blockblast.training.vec_env import ParallelVecEnv, env_kwargs_from_config, make_vec_env
from blockblast.utils.config import EnvConfig, RewardSection

KW = env_kwargs_from_config(EnvConfig(max_steps=40), RewardSection())


@pytest.fixture(scope="module")
def pair():  # type: ignore[no-untyped-def]
    par = make_vec_env(KW, n_envs=6, seed=11, use_subproc=True, n_workers=3)
    ref = make_vec_env(KW, n_envs=6, seed=11, use_subproc=False)
    yield par, ref
    par.close()
    ref.close()


def test_matches_dummy_vec_env_step_by_step(pair) -> None:  # type: ignore[no-untyped-def]
    par, ref = pair
    np.testing.assert_array_equal(par.reset(), ref.reset())
    rng = np.random.default_rng(0)
    dones_seen = 0
    for _ in range(120):  # max_steps=40 forces truncations as well as game overs
        masks = get_action_masks(ref)
        np.testing.assert_array_equal(get_action_masks(par), masks)
        actions = np.array([rng.choice(np.flatnonzero(m)) for m in masks])
        o1, r1, d1, i1 = par.step(actions)
        o2, r2, d2, i2 = ref.step(actions)
        np.testing.assert_array_equal(o1, o2)
        np.testing.assert_allclose(r1, r2)
        np.testing.assert_array_equal(d1, d2)
        for a, b in zip(i1, i2, strict=True):
            assert a["score"] == b["score"]
            ea, eb = dict(a.get("episode", {})), dict(b.get("episode", {}))
            ea.pop("t", None), eb.pop("t", None)  # wall-clock time, legitimately differs
            assert ea == eb
            if "terminal_observation" in b:
                np.testing.assert_array_equal(a["terminal_observation"], b["terminal_observation"])
                assert a["TimeLimit.truncated"] == b["TimeLimit.truncated"]
        dones_seen += int(d2.sum())
    assert dones_seen > 0


def test_masking_and_attributes(pair) -> None:  # type: ignore[no-untyped-def]
    par, _ = pair
    assert is_masking_supported(par)
    assert par.get_attr("max_steps") == [40] * 6
    assert par.get_attr("max_steps", indices=[4, 1]) == [40, 40]
    par.set_attr("max_steps", 41, indices=2)
    assert par.get_attr("max_steps") == [40, 40, 41, 40, 40, 40]
    moves = par.env_method("render", indices=[0])
    assert moves == [None]
    assert par.env_is_wrapped(object) == [False] * 6  # type: ignore[arg-type]


def test_workers_do_not_import_torch() -> None:
    vec = ParallelVecEnv(KW, n_envs=4, n_workers=2)
    try:
        assert vec.workers_import_torch() == [False, False]
    finally:
        vec.close()
        vec.close()  # idempotent


def test_single_worker_falls_back_to_dummy() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec = make_vec_env(KW, n_envs=4, seed=0, use_subproc=True, n_workers=1)
    assert isinstance(vec.venv, DummyVecEnv)
    vec.close()
