"""One-hidden-layer MLP policy: board -> logits over all (from, to) action pairs.

Invalid actions are masked to -inf before the softmax, so probability mass is
spread over valid actions only.

    net = MLPPolicyNet(size=8)
    logits = net(obs, mask)                 # masked logits, shape (B, N**4)
    dist = net.distribution(obs, mask)      # torch Categorical over valid actions

    policy = MLPPolicy(size=8)              # usable as env opponent / in the UI
    python -m checkers.ui --policy checkers.mlp:MLPPolicy --obs-mode board \\
        --policy-kwargs '{"checkpoint": "mlp.pt"}'
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

from .encoding import num_actions
from .policies import Policy


class MLPPolicyNet(nn.Module):
    def __init__(self, size: int = 8, hidden: int = 256, in_features: Optional[int] = None):
        super().__init__()
        self.size = size
        self.hidden = hidden
        # default input: the (N, N) canonical board (obs_mode="board"); pass
        # in_features=5*N*N to feed obs_mode="planes" instead
        self.in_features = in_features or size * size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions(size)),
        )

    def forward(self, obs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """obs: (B, N, N) or (N, N); mask: bool (B, N**4) or (N**4,). Returns masked logits."""
        obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.numel() == self.in_features:
            obs = obs.unsqueeze(0)  # single unbatched observation
        logits = self.net(obs)
        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device).reshape(logits.shape)
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits

    def distribution(self, obs, mask) -> torch.distributions.Categorical:
        """Softmax over valid actions only."""
        return torch.distributions.Categorical(logits=self.forward(obs, mask))


class MLPPolicy(Policy):
    """Wraps ``MLPPolicyNet`` as a ``Policy`` (opponent, UI, evaluation)."""

    def __init__(
        self,
        size: int = 8,
        hidden: int = 256,
        checkpoint: Optional[str] = None,
        deterministic: bool = False,
        seed: Optional[int] = None,
        net: Optional[MLPPolicyNet] = None,
    ):
        if checkpoint is not None:
            net = load_mlp(checkpoint)
        self.net = net or MLPPolicyNet(size=size, hidden=hidden)
        self.net.eval()
        self.deterministic = deterministic
        self.generator = torch.Generator()
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.generator.manual_seed(seed)

    @torch.no_grad()
    def act(self, obs: np.ndarray, action_mask: np.ndarray, info=None) -> int:
        logits = self.net(obs, action_mask)[0]
        if self.deterministic:
            return int(logits.argmax())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1, generator=self.generator))


def save_mlp(net: MLPPolicyNet, path: str) -> None:
    torch.save(
        {"size": net.size, "hidden": net.hidden, "in_features": net.in_features, "state_dict": net.state_dict()},
        path,
    )


def load_mlp(path: str) -> MLPPolicyNet:
    ckpt = torch.load(path, map_location="cpu")
    net = MLPPolicyNet(size=ckpt["size"], hidden=ckpt["hidden"], in_features=ckpt["in_features"])
    net.load_state_dict(ckpt["state_dict"])
    return net
