"""Policy networks: plain CNN extractor and the afterstate policy (default)."""

from __future__ import annotations

from typing import Any

import gymnasium
import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
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


class AfterstateExtractor(BaseFeaturesExtractor):
    """Scores all 192 actions by the exact board each one produces.

    For every (slot, row, col) the resulting board is computed in torch (place, then clear
    full rows/columns simultaneously), mirroring ``engine`` rules. A shared MLP scores
    ``(afterstate, remaining hand, lines, new combo, Δscore)``; the logit adds
    ``alpha · Δscore/10`` so the initial policy leans toward the greedy baseline. Illegal
    actions produce meaningless logits and are removed by the action mask.

    Output: ``[192 action logits | value_dim CNN features of the current state]``.
    """

    shift: torch.Tensor
    other_slots: torch.Tensor

    def __init__(
        self, observation_space: gymnasium.spaces.Box, hidden: int = 128, value_dim: int = 256
    ) -> None:
        super().__init__(observation_space, 192 + value_dim)
        self.register_buffer("shift", _shift_matrix(), persistent=False)
        self.register_buffer("other_slots", _OTHER_SLOTS.clone(), persistent=False)
        self.after_in = nn.Linear(64, hidden)
        self.rest_in = nn.Linear(128, hidden, bias=False)  # the 2 remaining hand planes
        self.scalar_in = nn.Linear(3, hidden, bias=False)  # lines, new combo, Δscore/10
        self.hidden = nn.Linear(hidden, hidden)
        self.head_out = nn.Linear(hidden, 1)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.value_cnn = BlockBlastCNN(observation_space, value_dim)

    def afterstates(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per action: (board after clears (b,192,64), n_lines, Δscore/10, new combo)."""
        b = obs.shape[0]
        board = obs[:, 0].reshape(b, 1, 64)
        planes = obs[:, 1:4].reshape(b, 3, 64)
        combo = obs[:, 4, 0, 0] * 8.0  # streak, capped at 8 like the observation
        placed = (planes @ self.shift).reshape(b, 192, 64)
        occ = torch.clamp(board + placed, max=1.0).reshape(b, 192, 8, 8)
        rows = occ.amin(dim=-1)  # (b, 192, 8), 1 = full row
        cols = occ.amin(dim=-2)
        cleared = torch.maximum(rows.unsqueeze(-1), cols.unsqueeze(-2))
        after = (occ * (1.0 - cleared)).reshape(b, 192, 64)
        n_lines = rows.sum(-1) + cols.sum(-1)
        n_cells = planes.sum(-1).repeat_interleave(64, dim=1)
        delta = (n_cells + 10.0 * n_lines * (1.0 + combo.unsqueeze(1))) / 10.0
        new_combo = torch.where(n_lines > 0, torch.clamp(combo + 1.0, max=8.0).unsqueeze(1), 0.0)
        return after, n_lines, delta, new_combo

    def action_logits(self, obs: torch.Tensor) -> torch.Tensor:
        after, n_lines, delta, new_combo = self.afterstates(obs)
        b = obs.shape[0]
        planes = obs[:, 1:4].reshape(b, 3, 64)
        rest = planes[:, self.other_slots].reshape(b, 3, 128)  # hand after using each slot
        rest_h = self.rest_in(rest).repeat_interleave(64, dim=1)  # (b, 192, hidden)
        scalars = torch.stack([n_lines / 4.0, new_combo / 8.0, delta], dim=-1)
        h = torch.relu(self.after_in(after) + rest_h + self.scalar_in(scalars))
        h = torch.relu(self.hidden(h))
        logits: torch.Tensor = self.head_out(h).squeeze(-1) + self.alpha * delta
        return logits

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

    def __init__(self, *args: Any, vf_arch: list[int] | None = None, **kwargs: Any) -> None:
        self._vf_arch = vf_arch if vf_arch is not None else [256]
        kwargs.setdefault("features_extractor_class", AfterstateExtractor)
        kwargs.setdefault("normalize_images", False)
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        params = super()._get_constructor_parameters()
        params["vf_arch"] = self._vf_arch
        return params

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
