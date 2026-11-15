"""State / action encodings shared by the env, the policies and the UI.

Everything a policy sees is in the *canonical* frame of the player to move:
the board is rotated 180 degrees for player -1, so a policy always plays
"upwards" with its own pieces positive. The same policy can therefore play
either side (handy for self-play).

Action encoding ("from x to"):
    square index  = row * N + col
    action id     = from_index * N*N + to_index      (action space size N**4)
Rotating by 180 degrees maps square index i -> N*N - 1 - i.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .game import KING, MAN, PLAYER_1, CheckersGame, Move, Square

OBS_MODES = ("planes", "board")


def num_actions(size: int) -> int:
    return size ** 4


def square_to_index(sq: Square, size: int) -> int:
    return sq[0] * size + sq[1]


def index_to_square(idx: int, size: int) -> Square:
    return divmod(int(idx), size)


def encode_action(frm: Square, to: Square, size: int) -> int:
    return square_to_index(frm, size) * size * size + square_to_index(to, size)


def decode_action(action: int, size: int) -> Tuple[Square, Square]:
    frm_idx, to_idx = divmod(int(action), size * size)
    return index_to_square(frm_idx, size), index_to_square(to_idx, size)


def to_canonical_square(sq: Square, player: int, size: int) -> Square:
    """Absolute <-> canonical square (the mapping is its own inverse)."""
    if player == PLAYER_1:
        return tuple(sq)
    return size - 1 - sq[0], size - 1 - sq[1]


def move_to_action(move: Move, player: int, size: int) -> int:
    """Absolute ``Move`` -> canonical action id for ``player``."""
    return encode_action(
        to_canonical_square(move.frm, player, size), to_canonical_square(move.to, player, size), size
    )


def action_to_squares(action: int, player: int, size: int) -> Tuple[Square, Square]:
    """Canonical action id for ``player`` -> absolute (from, to) squares."""
    frm, to = decode_action(action, size)
    return to_canonical_square(frm, player, size), to_canonical_square(to, player, size)


def canonical_board(game: CheckersGame, player: int) -> np.ndarray:
    """Board from ``player``'s view: own pieces positive, own side at the bottom."""
    b = game.board * np.int8(player)
    return b if player == PLAYER_1 else b[::-1, ::-1].copy()


def observe(game: CheckersGame, player: int, obs_mode: str = "planes") -> np.ndarray:
    """Observation for ``player``.

    planes: float32 (5, N, N) -> own men, own kings, opponent men, opponent
            kings, and a constant plane = 1 if ``player`` is to move.
    board:  int8 (N, N) canonical board with values in {-2..2}.
    """
    b = canonical_board(game, player)
    if obs_mode == "board":
        return b
    if obs_mode != "planes":
        raise ValueError(f"obs_mode must be one of {OBS_MODES}")
    to_move = np.full(b.shape, float(game.current_player == player), dtype=np.float32)
    return np.stack([b == MAN, b == KING, b == -MAN, b == -KING, to_move]).astype(np.float32)


def legal_actions(game: CheckersGame, player: int) -> list[int]:
    if game.current_player != player:
        return []
    return [move_to_action(m, player, game.size) for m in game.legal_moves()]


def action_mask(game: CheckersGame, player: int) -> np.ndarray:
    mask = np.zeros(num_actions(game.size), dtype=bool)
    mask[legal_actions(game, player)] = True
    return mask


def policy_inputs(game: CheckersGame, obs_mode: str = "planes"):
    """(obs, action_mask, info) for the player to move -- the input of ``Policy.act``."""
    player = game.current_player
    actions = legal_actions(game, player)
    mask = np.zeros(num_actions(game.size), dtype=bool)
    mask[actions] = True
    info = {
        "player": player,
        "legal_actions": actions,
        "legal_moves": [decode_action(a, game.size) for a in actions],  # canonical (from, to)
        "game": game.copy(),  # absolute-frame copy, for search-based policies
    }
    return observe(game, player, obs_mode), mask, info
