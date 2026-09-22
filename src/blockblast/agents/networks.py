"""Policy networks: plain CNN extractor and the afterstate policy (default)."""

from __future__ import annotations

from typing import Any, Final

import gymnasium
import numpy as np
import numpy.typing as npt
import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn


class BlockBlastCNN(BaseFeaturesExtractor):
    """3× Conv3x3(64, pad=1)+ReLU → flatten → Linear(features_dim)+ReLU.

    Padding keeps the full 8x8 resolution so each unit can see exact piece/hole alignment.
    """

    def __init__(self, observation_space: gymnasium.spaces.Box, features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        channels, height, width = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(nn.Linear(64 * height * width, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))  # type: ignore[no-any-return]


def _shift_matrix() -> torch.Tensor:
    """(64 src, 64 pos · 64 tgt) 0/1 matrix: piece cell ``src`` lands on ``tgt`` at ``pos``."""
    t = torch.zeros(64, 64, 64)
    for si in range(8):
        for sj in range(8):
            for r in range(8):
                for c in range(8):
                    if si + r < 8 and sj + c < 8:
                        t[si * 8 + sj, r * 8 + c, (si + r) * 8 + sj + c] = 1.0
    return t.reshape(64, 64 * 64)


_OTHER_SLOTS = torch.tensor([[1, 2], [0, 2], [0, 1]])

ILLEGAL_LOGIT: Final[float] = -1e8  # same value MaskableCategorical writes for masked actions


class AfterstateExtractor(BaseFeaturesExtractor):
    """Scores all 192 actions by the exact board each one produces.

    For every (slot, row, col) the resulting board is computed in torch (place, then clear
    full rows/columns simultaneously), mirroring ``engine`` rules. A shared MLP scores
    ``(afterstate, remaining hand, lines, new combo, Δscore)``; the logit adds
    ``alpha · prior`` so the initial policy leans toward a greedy rule: ``prior="score"``
    uses Δscore/10 (most points), ``prior="board"`` uses −filled cells/8 (emptiest board,
    for the ``safe`` reward). Illegal actions produce meaningless logits and are removed by
    the action mask.

    Output: ``[192 action logits | value_dim CNN features of the current state]``.
    """

    shift: torch.Tensor
    shift_t: torch.Tensor
    in_board: torch.Tensor
    other_slots: torch.Tensor

    def __init__(
        self,
        observation_space: gymnasium.spaces.Box,
        hidden: int = 128,
        value_dim: int = 256,
        prior: str = "score",
    ) -> None:
        super().__init__(observation_space, 192 + value_dim)
        if prior not in ("score", "board"):
            raise ValueError(f"unknown prior {prior!r}")
        self.prior = prior
        shift = _shift_matrix()
        self.register_buffer("shift", shift, persistent=False)
        # (tgt, src·pos) and (src, pos) views of the same map, for the legality counts
        shift_t = shift.reshape(64, 64, 64).permute(2, 0, 1).reshape(64, 64 * 64).contiguous()
        self.register_buffer("shift_t", shift_t, persistent=False)
        self.register_buffer("in_board", shift.reshape(64, 64, 64).sum(-1), persistent=False)
        self.register_buffer("other_slots", _OTHER_SLOTS.clone(), persistent=False)
        self.after_in = nn.Linear(64, hidden)
        self.rest_in = nn.Linear(128, hidden, bias=False)  # the 2 remaining hand planes
        self.scalar_in = nn.Linear(3, hidden, bias=False)  # lines, new combo, Δscore/10
        self.hidden = nn.Linear(hidden, hidden)
        self.head_out = nn.Linear(hidden, 1)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.value_cnn = BlockBlastCNN(observation_space, value_dim)

    def _placed(self, obs: torch.Tensor) -> torch.Tensor:
        """(b, 192, 64) piece cells of every action; cells falling off the board are dropped."""
        b = obs.shape[0]
        # One (3b, 64) @ (64, 4096) GEMM: ~2x faster than the batched (b, 3, 64) @ (64, 4096)
        # broadcast, and exact either way (every output is a sum of 0/1 terms, at most one 1).
        return (obs[:, 1:4].reshape(b * 3, 64) @ self.shift).reshape(b, 192, 64)

    @staticmethod
    def _resolve(
        board: torch.Tensor, placed: torch.Tensor, n_cells: torch.Tensor, combo: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Place, clear full rows/columns simultaneously, score. Any leading dims ``(...)``.

        ``board``/``placed``: ``(..., 64)``; ``n_cells``/``combo``: ``(...)``.
        """
        lead = placed.shape[:-1]
        occ = torch.clamp(board + placed, max=1.0).reshape(*lead, 8, 8)
        rows = occ.amin(dim=-1)  # (..., 8), 1 = full row
        cols = occ.amin(dim=-2)
        cleared = torch.maximum(rows.unsqueeze(-1), cols.unsqueeze(-2))
        after = (occ * (1.0 - cleared)).reshape(*lead, 64)
        n_lines = rows.sum(-1) + cols.sum(-1)
        delta = (n_cells + 10.0 * n_lines * (1.0 + combo)) / 10.0
        new_combo = torch.where(n_lines > 0, torch.clamp(combo + 1.0, max=8.0), 0.0)
        return after, n_lines, delta, new_combo

    def _prior(self, after: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return delta if self.prior == "score" else after.sum(-1) / -8.0

    def afterstates(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per action: (board after clears (b,192,64), n_lines, Δscore/10, new combo)."""
        b = obs.shape[0]
        board = obs[:, 0].reshape(b, 1, 64)
        combo = obs[:, 4, 0, 0].unsqueeze(1) * 8.0  # streak, capped at 8 like the observation
        n_cells = obs[:, 1:4].reshape(b, 3, 64).sum(-1).repeat_interleave(64, dim=1)
        return self._resolve(board, self._placed(obs), n_cells, combo)

    def legal(self, obs: torch.Tensor) -> torch.Tensor:
        """(b, 192) bool legality, derived from the observation (equals the env action mask).

        In bounds (no piece cell dropped by the shift), no overlap with the board, and the
        slot holds a piece. Counted with two small GEMMs rather than reductions over the
        (b, 192, 64) placements; exact, since the counts are small integers.
        """
        b = obs.shape[0]
        planes = obs[:, 1:4].reshape(b, 3, 64)
        # cover[b, src, pos] = board cell under piece cell ``src`` when anchored at ``pos``
        cover = (obs[:, 0].reshape(b, 64) @ self.shift_t).reshape(b, 64, 64)
        overlap = torch.bmm(planes, cover).reshape(b, 192)
        kept = (planes @ self.in_board).reshape(b, 192)  # piece cells that stay on the board
        n_cells = planes.sum(-1).repeat_interleave(64, dim=1)
        return (kept == n_cells) & (overlap == 0) & (n_cells > 0)

    def action_logits(self, obs: torch.Tensor, compact: bool = True) -> torch.Tensor:
        """(b, 192) logits. Illegal actions get ``ILLEGAL_LOGIT``.

        ``compact=True`` runs the MLP on legal actions only (~20 % of 192 in real games).
        The action mask replaces illegal logits and zeroes their gradients anyway, so the
        masked distribution and every parameter gradient are the same as scoring all 192.
        It needs one host sync (``nonzero``); ``compact=False`` scores all 192 with static
        shapes, which is what the CUDA-graph rollout path captures.
        """
        if not compact:
            return self._dense_logits(obs)
        b = obs.shape[0]
        planes = obs[:, 1:4].reshape(b, 3, 64)
        board = obs[:, 0].reshape(b, 1, 64)
        placed = self._placed(obs)
        n_cells = planes.sum(-1)  # (b, 3)
        idx = self.legal(obs).reshape(-1).nonzero().squeeze(1)  # flat b·192 + action
        bi, slot = idx // 192, (idx % 192) // 64
        after, n_lines, delta, new_combo = self._resolve(
            board.reshape(b, 64)[bi],
            placed.reshape(b * 192, 64)[idx],
            n_cells[bi, slot],
            obs[:, 4, 0, 0][bi] * 8.0,
        )
        rest = planes[:, self.other_slots].reshape(b, 3, 128)  # hand after using each slot
        rest_h = self.rest_in(rest).reshape(b * 3, -1)[bi * 3 + slot]
        scalars = torch.stack([n_lines / 4.0, new_combo / 8.0, delta], dim=-1)
        h = torch.relu(self.after_in(after) + rest_h + self.scalar_in(scalars))
        h = torch.relu(self.hidden(h))
        scores = self.head_out(h).squeeze(-1) + self.alpha * self._prior(after, delta)
        logits = obs.new_full((b * 192,), ILLEGAL_LOGIT).index_put((idx,), scores)
        return logits.reshape(b, 192)

    def _dense_logits(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        board = obs[:, 0].reshape(b, 1, 64)
        planes = obs[:, 1:4].reshape(b, 3, 64)
        placed = self._placed(obs)
        n_cells = planes.sum(-1).repeat_interleave(64, dim=1)
        combo = obs[:, 4, 0, 0].unsqueeze(1) * 8.0
        after, n_lines, delta, new_combo = self._resolve(board, placed, n_cells, combo)
        rest = planes[:, self.other_slots].reshape(b, 3, 128)
        rest_h = self.rest_in(rest).repeat_interleave(64, dim=1)  # (b, 192, hidden)
        scalars = torch.stack([n_lines / 4.0, new_combo / 8.0, delta], dim=-1)
        h = torch.relu(self.after_in(after) + rest_h + self.scalar_in(scalars))
        h = torch.relu(self.hidden(h))
        scores = self.head_out(h).squeeze(-1) + self.alpha * self._prior(after, delta)
        return torch.where(self.legal(obs), scores, torch.full_like(scores, ILLEGAL_LOGIT))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.action_logits(observations), self.value_cnn(observations)], dim=1)


class _SplitLatent(nn.Module):
    """Routes the first 192 features (logits) to the actor, the rest through a critic MLP."""

    def __init__(self, feature_dim: int, vf_arch: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = feature_dim - 192
        for width in vf_arch:
            layers += [nn.Linear(last, width), nn.ReLU()]
            last = width
        self.critic = nn.Sequential(*layers)
        self.latent_dim_pi = 192
        self.latent_dim_vf = last

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, :192]

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.critic(features[:, 192:])  # type: ignore[no-any-return]


class AfterstatePolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy whose logits come straight from ``AfterstateExtractor``."""

    use_cuda_graphs: bool = True  # set False on an instance to debug the eager rollout path

    def __init__(self, *args: Any, vf_arch: list[int] | None = None, **kwargs: Any) -> None:
        self._vf_arch = vf_arch if vf_arch is not None else [256]
        self._graphs: dict[Any, Any] = {}
        kwargs.setdefault("features_extractor_class", AfterstateExtractor)
        kwargs.setdefault("normalize_images", False)
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        params = super()._get_constructor_parameters()
        params["vf_arch"] = self._vf_arch
        return params

    def logits_and_values(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Action logits (b, 192) and values (b, 1), without the (b, 448) feature concat.

        Under ``no_grad`` on CUDA (rollout collection) this replays a CUDA graph captured
        per batch size: a rollout step is ~140 tiny kernels, and on Windows launching them
        one by one costs ~4x the GPU time they take.
        """
        obs = obs.float()
        if self.use_cuda_graphs and obs.is_cuda and not torch.is_grad_enabled():
            return self._graphed(obs)
        return self._logits_and_values(obs, compact=True)

    def _logits_and_values(
        self, obs: torch.Tensor, compact: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ext = self.features_extractor
        assert isinstance(ext, AfterstateExtractor)
        mlp = self.mlp_extractor
        assert isinstance(mlp, _SplitLatent)
        values: torch.Tensor = self.value_net(mlp.critic(ext.value_cnn(obs)))
        return ext.action_logits(obs, compact=compact), values

    def _graphed(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Keyed on parameter storage too: the graph reads weights by address, so a
        # re-allocation (e.g. ``.to(device)``, which moves every parameter) must trigger a new
        # capture. In-place optimizer updates and ``load_state_dict`` keep the addresses.
        key = (tuple(obs.shape), next(self.parameters()).data_ptr())
        entry = self._graphs.get(key)
        if entry is None:
            self._graphs.clear()  # old captures reference stale parameters
            static_in = obs.clone()
            side = torch.cuda.Stream(obs.device)  # type: ignore[no-untyped-call]
            side.wait_stream(torch.cuda.current_stream(obs.device))
            with torch.cuda.stream(side):
                for _ in range(2):  # warm-up allocations/cuBLAS handles outside the capture
                    self._logits_and_values(static_in, compact=False)
            torch.cuda.current_stream(obs.device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = self._logits_and_values(static_in, compact=False)
            entry = self._graphs[key] = (graph, static_in, static_out)
        graph, static_in, (logits, values) = entry
        static_in.copy_(obs)
        graph.replay()
        return logits.clone(), values.clone()

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_masks: npt.NDArray[np.bool_] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same contract as ``MaskableActorCriticPolicy.forward`` (action, value, log-prob)."""
        logits, values = self.logits_and_values(obs)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        shape = self.action_space.shape or ()  # () for Discrete
        return actions.reshape((-1, *shape)), values, log_prob

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        """Critic only: skips the actor's afterstate scoring the base class would run."""
        assert isinstance(obs, torch.Tensor)
        ext = self.features_extractor
        assert isinstance(ext, AfterstateExtractor)
        mlp = self.mlp_extractor
        assert isinstance(mlp, _SplitLatent)
        return self.value_net(mlp.critic(ext.value_cnn(obs.float())))  # type: ignore[no-any-return]

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = _SplitLatent(self.features_dim, self._vf_arch)  # type: ignore[assignment]

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = nn.Identity()  # logits are already per action
        extractor = self.features_extractor
        assert isinstance(extractor, AfterstateExtractor)
        nn.init.orthogonal_(extractor.head_out.weight, gain=0.01)  # start near alpha·Δscore
        nn.init.zeros_(extractor.head_out.bias)
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
