"""Policy interface used for opponents, agents and the UI.

A policy maps ``(obs, action_mask, info) -> action`` where everything is in the
canonical frame of the player to move (see ``encoding.py``). ``info`` contains
``player``, ``legal_actions``, ``legal_moves`` (canonical (from, to) pairs) and
``game`` (an absolute-frame copy of the game, for search-based policies).

Any of these can be used where a policy is expected:
  * a ``Policy`` subclass instance
  * a plain callable ``f(obs, action_mask, info) -> int``
  * a trained model, wrapped e.g. as
        FunctionPolicy(lambda obs, mask, info: model.predict(obs, action_masks=mask)[0])
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .encoding import move_to_action


class Policy:
    def act(self, obs: np.ndarray, action_mask: np.ndarray, info: dict) -> int:
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None) -> None:
        """Called at the start of every episode."""

    def __call__(self, obs, action_mask, info) -> int:
        return self.act(obs, action_mask, info)


class FunctionPolicy(Policy):
    def __init__(self, fn: Callable[[np.ndarray, np.ndarray, dict], int]):
        self.fn = fn

    def act(self, obs, action_mask, info) -> int:
        return int(self.fn(obs, action_mask, info))


def as_policy(obj: Any) -> Policy:
    if isinstance(obj, Policy):
        return obj
    if callable(obj):
        return FunctionPolicy(obj)
    raise TypeError(f"cannot use {obj!r} as a policy")


class RandomPolicy(Policy):
    """Uniformly random over legal actions."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, obs, action_mask, info) -> int:
        legal = np.flatnonzero(action_mask)
        if len(legal) == 0:
            raise ValueError("no legal actions")
        return int(self.rng.choice(legal))


class GreedyPolicy(RandomPolicy):
    """Baseline: prefer capturing kings, then men, then promoting; random tie-break."""

    def act(self, obs, action_mask, info) -> int:
        game, player = info["game"], info["player"]
        promo_row = game.promotion_row(player)
        best, best_score = [], -1
        for m in game.legal_moves():
            score = 0
            if m.captured is not None:
                score += 10 * abs(int(game.board[m.captured]))
            if abs(int(game.board[m.frm])) == 1 and m.to[0] == promo_row:
                score += 5
            if score > best_score:
                best, best_score = [m], score
            elif score == best_score:
                best.append(m)
        move = best[self.rng.integers(len(best))]
        return move_to_action(move, player, game.size)


def load_policy(spec: str, **kwargs) -> Policy:
    """Load a policy from ``"package.module:Name"`` or ``"path/to/file.py:Name"``.

    If ``Name`` is a class (or factory function with no required args) it is
    called with ``kwargs``; otherwise the object itself is used.
    Shortcuts: ``"random"``, ``"greedy"``.
    """
    shortcuts = {"random": RandomPolicy, "greedy": GreedyPolicy}
    if spec.lower() in shortcuts:
        return shortcuts[spec.lower()](**kwargs)
    target, _, attr = spec.rpartition(":")
    if not target:
        raise ValueError("policy spec must look like 'module:Name' or 'file.py:Name'")
    if target.endswith(".py"):
        path = Path(target).resolve()
        mod_spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    obj = getattr(module, attr)
    if inspect.isclass(obj) or (inspect.isfunction(obj) and _is_factory(obj)):
        obj = obj(**kwargs)
    return as_policy(obj)


def _is_factory(fn) -> bool:
    """A function with no required positional args is treated as a policy factory."""
    params = inspect.signature(fn).parameters.values()
    return not any(p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params)
