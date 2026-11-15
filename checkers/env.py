"""Gymnasium environment: a learning agent plays checkers against an opponent policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .encoding import (
    OBS_MODES,
    action_mask,
    action_to_squares,
    legal_actions,
    move_to_action,
    num_actions,
    observe,
    policy_inputs,
)
from .game import KING, PLAYER_1, PLAYER_2, CheckersGame, Move, MoveResult
from .policies import Policy, RandomPolicy, as_policy


@dataclass
class RewardConfig:
    """All rewards are from the agent's point of view."""

    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    capture: float = 0.0  # per opponent man captured by the agent (kings count king_multiplier x)
    captured: float = 0.0  # per agent man lost (use a negative value to penalise)
    promotion: float = 0.0  # agent man promoted to king
    opponent_promotion: float = 0.0  # opponent man promoted (use a negative value to penalise)
    king_multiplier: float = 2.0
    step: float = 0.0  # added on every agent step
    illegal: float = -1.0  # only used with illegal_action="penalize"


class CheckersEnv(gym.Env):
    """Single-agent checkers.

    The agent controls ``agent_player`` (1 = bottom, moves first; -1 = top;
    "random" = pick each episode). After each agent move the ``opponent``
    policy replies, so every ``step`` returns a state where it's the agent's
    turn (or the game is over).

    Observation / action are in the agent's canonical frame (see encoding.py):
      * observation: (5, N, N) float planes, or (N, N) int8 board with obs_mode="board"
      * action: Discrete(N**4), action = from_idx * N*N + to_idx, idx = row*N + col
      * valid actions: ``info["action_mask"]``, ``env.action_masks()`` or ``env.valid_actions()``
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        size: int = 8,
        opponent: Union[Policy, Any, None] = None,
        agent_player: Union[int, str] = PLAYER_1,
        obs_mode: str = "planes",
        rewards: Optional[RewardConfig] = None,
        illegal_action: str = "raise",
        max_episode_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        **game_kwargs,
    ):
        if obs_mode not in OBS_MODES:
            raise ValueError(f"obs_mode must be one of {OBS_MODES}")
        if illegal_action not in ("raise", "penalize"):
            raise ValueError("illegal_action must be 'raise' or 'penalize'")
        if agent_player not in (PLAYER_1, PLAYER_2, "random"):
            raise ValueError("agent_player must be 1, -1 or 'random'")
        self.size = size
        self.game = CheckersGame(size=size, **game_kwargs)
        self.opponent = as_policy(opponent) if opponent is not None else RandomPolicy()
        self.agent_player_cfg = agent_player
        self.agent_player = PLAYER_1 if agent_player == "random" else agent_player
        self.obs_mode = obs_mode
        self.rewards = rewards or RewardConfig()
        self.illegal_action = illegal_action
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.steps = 0

        self.action_space = spaces.Discrete(num_actions(size))
        if obs_mode == "planes":
            self.observation_space = spaces.Box(0.0, 1.0, shape=(5, size, size), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(-KING, KING, shape=(size, size), dtype=np.int8)

    # -------------------------------------------------------------- gym API

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.opponent.reset(seed=None if seed is None else seed + 1)
        if self.agent_player_cfg == "random":
            self.agent_player = int(self.np_random.choice([PLAYER_1, PLAYER_2]))
        self.game.reset()
        self.steps = 0
        reward = 0.0
        if self.game.current_player != self.agent_player:
            reward = self._opponent_turn()
        info = self._info()
        info["opponent_reward"] = reward  # reward already incurred by the opponent's opening move (normally 0)
        if self.render_mode == "human":
            self.render()
        return self._obs(), info

    def step(self, action: int):
        if self.game.is_over:
            raise RuntimeError("episode is over, call reset()")
        self.steps += 1
        frm, to = action_to_squares(int(action), self.agent_player, self.size)
        if self.game.find_move(frm, to) is None:
            if self.illegal_action == "raise":
                raise ValueError(f"illegal action {action} ({frm} -> {to}); legal: {self.valid_actions()}")
            info = self._info()
            info["illegal_action"] = True
            return self._obs(), self.rewards.illegal, True, False, info

        reward = self.rewards.step + self._event_reward(self.game.step((frm, to)))
        if not self.game.is_over:
            reward += self._opponent_turn()
        if self.game.is_over:
            reward += self._outcome_reward()

        terminated = self.game.is_over
        truncated = not terminated and self.max_episode_steps is not None and self.steps >= self.max_episode_steps
        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, self._info()

    def render(self):
        text = str(self.game)
        if self.render_mode == "human":
            print(text + "\n")
            return None
        return text

    # ------------------------------------------------------------ helpers

    def valid_actions(self) -> list[int]:
        return legal_actions(self.game, self.agent_player)

    def action_masks(self) -> np.ndarray:
        """Boolean mask over the action space (sb3-contrib MaskablePPO convention)."""
        return action_mask(self.game, self.agent_player)

    def action_from_squares(self, frm, to) -> int:
        """Agent action id from *absolute* board squares."""
        return move_to_action(Move(tuple(frm), tuple(to)), self.agent_player, self.size)

    def _opponent_turn(self) -> float:
        obs, mask, info = policy_inputs(self.game, self.obs_mode)
        action = int(self.opponent.act(obs, mask, info))
        if not mask[action]:
            raise ValueError(f"opponent policy chose illegal action {action}")
        frm, to = action_to_squares(action, self.game.current_player, self.size)
        return self._event_reward(self.game.step((frm, to)))

    def _event_reward(self, result: MoveResult) -> float:
        r = self.rewards
        weight = r.king_multiplier if result.captured_piece == KING else 1.0
        reward = 0.0
        if result.player == self.agent_player:
            if result.captured_piece:
                reward += r.capture * weight
            if result.promoted:
                reward += r.promotion
        else:
            if result.captured_piece:
                reward += r.captured * weight
            if result.promoted:
                reward += r.opponent_promotion
        return reward

    def _outcome_reward(self) -> float:
        if self.game.winner == self.agent_player:
            return self.rewards.win
        if self.game.winner == -self.agent_player:
            return self.rewards.loss
        return self.rewards.draw

    def _obs(self) -> np.ndarray:
        return observe(self.game, self.agent_player, self.obs_mode)

    def _info(self) -> dict:
        mask = self.action_masks()
        return {
            "action_mask": mask,
            "valid_actions": np.flatnonzero(mask).tolist(),
            "agent_player": self.agent_player,
            "board": self.game.board.copy(),  # absolute frame
            "last_move": self.game.last_move,  # absolute frame
            "winner": self.game.winner,
            "plies": self.game.plies,
        }
