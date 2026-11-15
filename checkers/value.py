"""One-hidden-layer MLP value function: board -> V(s), the expected return from s.

    net = ValueNet(size=8)
    v = net(obs)          # (B,) for a batch of (B, N, N) boards, (1,) for one board
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ValueNet(nn.Module):
    def __init__(self, size: int = 8, hidden: int = 256, in_features: Optional[int] = None):
        super().__init__()
        self.size = size
        self.hidden = hidden
        # same input convention as MLPPolicyNet: the (N, N) canonical board by default
        self.in_features = in_features or size * size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.numel() == self.in_features:
            obs = obs.unsqueeze(0)  # single unbatched observation
        return self.net(obs).squeeze(-1)


def save_value(net: ValueNet, path: str) -> None:
    torch.save(
        {"size": net.size, "hidden": net.hidden, "in_features": net.in_features, "state_dict": net.state_dict()},
        path,
    )


def load_value(path: str) -> ValueNet:
    ckpt = torch.load(path, map_location="cpu")
    net = ValueNet(size=ckpt["size"], hidden=ckpt["hidden"], in_features=ckpt["in_features"])
    net.load_state_dict(ckpt["state_dict"])
    return net
