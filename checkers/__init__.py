from gymnasium.envs.registration import register

from .encoding import (
    action_mask,
    action_to_squares,
    decode_action,
    encode_action,
    move_to_action,
    observe,
    policy_inputs,
)
from .env import CheckersEnv, RewardConfig
from .game import PLAYER_1, PLAYER_2, CheckersGame, Move
from .policies import FunctionPolicy, GreedyPolicy, Policy, RandomPolicy, as_policy, load_policy

register(id="Checkers-v0", entry_point="checkers.env:CheckersEnv")

__all__ = [
    "CheckersEnv",
    "CheckersGame",
    "FunctionPolicy",
    "GreedyPolicy",
    "Move",
    "PLAYER_1",
    "PLAYER_2",
    "Policy",
    "RandomPolicy",
    "RewardConfig",
    "action_mask",
    "action_to_squares",
    "as_policy",
    "decode_action",
    "encode_action",
    "load_policy",
    "move_to_action",
    "observe",
    "policy_inputs",
]
