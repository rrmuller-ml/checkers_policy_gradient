"""Depth-limited tree search guided by a policy net and a value net.

At every node the policy picks its top-``width`` moves; the tree goes
``depth`` plies deep (alternating sides), so depth=3, width=2 gives 2**3 = 8
leaves. Leaves are scored with the value net, terminal positions with the real
outcome (+1 / -1 / 0). Backup: the searching player takes the max, the
opponent takes the min (``opponent="min"``) or the mean (``opponent="mean"``).

This is a deterministic, policy-pruned minimax search, not sampled MCTS.
"""

from __future__ import annotations

import torch

from .encoding import action_mask, action_to_squares, observe
from .game import CheckersGame
from .mlp import MLPPolicyNet
from .policies import Policy
from .value import ValueNet


class SearchPolicy(Policy):
    def __init__(
        self,
        policy_net: MLPPolicyNet,
        value_net: ValueNet,
        depth: int = 3,
        width: int = 2,
        opponent: str = "min",
    ):
        if opponent not in ("min", "mean"):
            raise ValueError("opponent must be 'min' or 'mean'")
        self.policy_net = policy_net.eval()
        self.value_net = value_net.eval()
        self.depth = depth
        self.width = width
        self.opponent = opponent

    @torch.no_grad()
    def act(self, obs, action_mask_, info) -> int:
        game, root = info["game"], info["player"]
        best_action, best_value = None, -float("inf")
        for action in self._top_actions(game):
            value = self._search(self._child(game, action), root, self.depth - 1)
            if value > best_value:
                best_action, best_value = action, value
        return best_action

    def _top_actions(self, game: CheckersGame) -> list[int]:
        player = game.current_player
        mask = action_mask(game, player)
        logits = self.policy_net(observe(game, player, "board"), mask)[0]
        k = min(self.width, int(mask.sum()))
        return logits.topk(k).indices.tolist()

    def _child(self, game: CheckersGame, action: int) -> CheckersGame:
        child = game.copy()
        child.step(action_to_squares(action, game.current_player, game.size))
        return child

    def _search(self, game: CheckersGame, root: int, depth: int) -> float:
        """Value of ``game`` for ``root``."""
        if game.is_over:
            return 0.0 if game.winner == 0 else (1.0 if game.winner == root else -1.0)
        if depth == 0:
            # V is from the view of the player to move; negate when that's the opponent
            v = self.value_net(observe(game, game.current_player, "board")).item()
            return v if game.current_player == root else -v
        values = [self._search(self._child(game, a), root, depth - 1) for a in self._top_actions(game)]
        if game.current_player == root:
            return max(values)
        return min(values) if self.opponent == "min" else sum(values) / len(values)


def load_search_policy(
    policy: str = "checkpoints/mlp_vs_random_400it.pt",
    value: str = "checkpoints/value_vs_random.pt",
    depth: int = 3,
    width: int = 2,
    opponent: str = "min",
) -> SearchPolicy:
    """Factory for ``load_policy`` / the UI: ``--policy checkers.search:load_search_policy``."""
    from .mlp import load_mlp
    from .value import load_value

    return SearchPolicy(load_mlp(policy), load_value(value), depth, width, opponent)
