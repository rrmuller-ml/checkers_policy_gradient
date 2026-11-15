"""Core checkers rules, independent of any RL framework.

Board encoding (absolute frame, row 0 is the top of the board):
    0  empty square
   +1  man of player 1 (starts at the bottom, moves "up" = decreasing row)
   +2  king of player 1
   -1  man of player -1 (starts at the top, moves "down" = increasing row)
   -2  king of player -1

Rules implemented:
  * Men move one square diagonally forward.
  * Men capture by jumping 2 squares diagonally over an adjacent opponent
    piece, forward or backward.
  * Only one piece can be captured per move (no multi-jump chains).
  * A man reaching the far row is promoted to king. Kings move any number of
    empty squares diagonally in any direction ("flying" kings) and capture by
    jumping over a single opponent piece at any distance along a diagonal,
    landing on the square immediately beyond it.
  * Captures are mandatory when ``forced_capture=True`` (the default).
  * A player with no legal move loses (this includes having no pieces).
  * The game is a draw after ``no_progress_limit`` plies without a capture or
    a man move, or after ``max_plies`` plies in total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

Square = Tuple[int, int]

EMPTY, MAN, KING = 0, 1, 2
PLAYER_1, PLAYER_2 = 1, -1
DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass(frozen=True)
class Move:
    frm: Square
    to: Square
    captured: Optional[Square] = None

    @property
    def is_capture(self) -> bool:
        return self.captured is not None


@dataclass
class MoveResult:
    move: Move
    player: int
    captured_piece: int = 0  # absolute value of the captured piece (MAN/KING), 0 if none
    promoted: bool = False


class CheckersGame:
    def __init__(
        self,
        size: int = 8,
        rows_per_player: Optional[int] = None,
        forced_capture: bool = True,
        flying_kings: bool = True,
        no_progress_limit: Optional[int] = 80,
        max_plies: Optional[int] = None,
    ):
        if size < 4:
            raise ValueError("board size must be >= 4")
        self.size = size
        self.rows_per_player = rows_per_player if rows_per_player is not None else (size - 2) // 2
        if not 1 <= self.rows_per_player <= (size - 1) // 2:
            raise ValueError(f"rows_per_player={self.rows_per_player} does not fit a {size}x{size} board")
        self.forced_capture = forced_capture
        self.flying_kings = flying_kings
        self.no_progress_limit = no_progress_limit
        self.max_plies = max_plies
        self.reset()

    # ------------------------------------------------------------------ setup

    def reset(self) -> None:
        n, k = self.size, self.rows_per_player
        self.board = np.zeros((n, n), dtype=np.int8)
        for r in range(n):
            for c in range(n):
                if (r + c) % 2 == 1:
                    if r < k:
                        self.board[r, c] = PLAYER_2 * MAN
                    elif r >= n - k:
                        self.board[r, c] = PLAYER_1 * MAN
        self.current_player = PLAYER_1
        self.plies = 0
        self.plies_since_progress = 0
        self.winner: Optional[int] = None  # 1, -1, 0 (draw) or None while ongoing
        self.last_move: Optional[Move] = None
        self._moves_cache: Optional[list[Move]] = None

    @classmethod
    def from_board(cls, board: Union[np.ndarray, Sequence[Sequence[int]]], current_player: int = PLAYER_1, **kwargs) -> "CheckersGame":
        """Build a game from an arbitrary position (useful for tests / puzzles)."""
        board = np.asarray(board, dtype=np.int8)
        game = cls(size=board.shape[0], **kwargs)
        game.board = board.copy()
        game.current_player = current_player
        game._check_game_over()
        return game

    def copy(self) -> "CheckersGame":
        other = object.__new__(CheckersGame)
        other.__dict__.update(self.__dict__)
        other.board = self.board.copy()
        other._moves_cache = list(self._moves_cache) if self._moves_cache is not None else None
        return other

    # ---------------------------------------------------------------- queries

    @property
    def is_over(self) -> bool:
        return self.winner is not None

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size

    def forward(self, player: int) -> int:
        """Row delta of a forward step for ``player``."""
        return -1 if player == PLAYER_1 else 1

    def promotion_row(self, player: int) -> int:
        return 0 if player == PLAYER_1 else self.size - 1

    def piece_counts(self) -> dict:
        b = self.board
        return {
            PLAYER_1: {"men": int((b == MAN).sum()), "kings": int((b == KING).sum())},
            PLAYER_2: {"men": int((b == -MAN).sum()), "kings": int((b == -KING).sum())},
        }

    def legal_moves(self) -> list[Move]:
        """Legal moves for ``current_player`` (empty once the game is over)."""
        if self.winner is not None:
            return []
        if self._moves_cache is None:
            self._moves_cache = self._generate_moves(self.current_player)
        return list(self._moves_cache)

    def _generate_moves(self, player: int) -> list[Move]:
        simple: list[Move] = []
        captures: list[Move] = []
        rows, cols = np.nonzero(self.board * player > 0)
        for r, c in zip(rows.tolist(), cols.tolist()):
            if abs(self.board[r, c]) == KING and self.flying_kings:
                self._flying_king_moves(r, c, player, simple, captures)
            else:
                self._step_moves(r, c, player, simple, captures)
        if self.forced_capture and captures:
            return captures
        return captures + simple

    def _step_moves(self, r: int, c: int, player: int, simple: list, captures: list) -> None:
        is_king = abs(self.board[r, c]) == KING
        fwd = self.forward(player)
        for dr, dc in DIRECTIONS:
            r1, c1 = r + dr, c + dc
            if not self.in_bounds(r1, c1):
                continue
            target = self.board[r1, c1]
            if target == EMPTY:
                if is_king or dr == fwd:
                    simple.append(Move((r, c), (r1, c1)))
            elif target * player < 0:
                r2, c2 = r1 + dr, c1 + dc
                if self.in_bounds(r2, c2) and self.board[r2, c2] == EMPTY:
                    captures.append(Move((r, c), (r2, c2), (r1, c1)))

    def _flying_king_moves(self, r: int, c: int, player: int, simple: list, captures: list) -> None:
        for dr, dc in DIRECTIONS:
            r1, c1 = r + dr, c + dc
            while self.in_bounds(r1, c1) and self.board[r1, c1] == EMPTY:
                simple.append(Move((r, c), (r1, c1)))
                r1, c1 = r1 + dr, c1 + dc
            if not self.in_bounds(r1, c1) or self.board[r1, c1] * player > 0:
                continue
            r2, c2 = r1 + dr, c1 + dc
            if self.in_bounds(r2, c2) and self.board[r2, c2] == EMPTY:
                captures.append(Move((r, c), (r2, c2), (r1, c1)))

    def find_move(self, frm: Square, to: Square) -> Optional[Move]:
        frm, to = tuple(frm), tuple(to)
        for m in self.legal_moves():
            if m.frm == frm and m.to == to:
                return m
        return None

    # ---------------------------------------------------------------- actions

    def step(self, move: Union[Move, Tuple[Square, Square]]) -> MoveResult:
        """Apply a legal move for ``current_player``. Raises ValueError if illegal."""
        if self.winner is not None:
            raise ValueError("game is over")
        frm, to = (move.frm, move.to) if isinstance(move, Move) else move
        legal = self.find_move(frm, to)
        if legal is None:
            raise ValueError(f"illegal move {tuple(frm)} -> {tuple(to)} for player {self.current_player}")

        player = self.current_player
        piece = int(self.board[legal.frm])
        result = MoveResult(move=legal, player=player)
        self.board[legal.frm] = EMPTY
        if legal.captured is not None:
            result.captured_piece = abs(int(self.board[legal.captured]))
            self.board[legal.captured] = EMPTY
        if abs(piece) == MAN and legal.to[0] == self.promotion_row(player):
            piece = player * KING
            result.promoted = True
        self.board[legal.to] = piece

        self.plies += 1
        if legal.is_capture or abs(int(self.board[legal.to])) == MAN or result.promoted:
            self.plies_since_progress = 0
        else:
            self.plies_since_progress += 1
        self.last_move = legal
        self.current_player = -player
        self._moves_cache = None
        self._check_game_over()
        return result

    def _check_game_over(self) -> None:
        self._moves_cache = None
        self.winner = None
        if not self.legal_moves():
            self.winner = -self.current_player
        elif self.no_progress_limit is not None and self.plies_since_progress >= self.no_progress_limit:
            self.winner = 0
        elif self.max_plies is not None and self.plies >= self.max_plies:
            self.winner = 0
        if self.winner is not None:
            self._moves_cache = []

    # ---------------------------------------------------------------- display

    SYMBOLS = {0: ".", MAN: "x", KING: "X", -MAN: "o", -KING: "O"}

    def __str__(self) -> str:
        n = self.size
        header = "   " + " ".join(f"{c:>2}" for c in range(n))
        lines = [header]
        for r in range(n):
            cells = []
            for c in range(n):
                cells.append(self.SYMBOLS[int(self.board[r, c])] if (r + c) % 2 == 1 else " ")
            lines.append(f"{r:>2}  " + "  ".join(cells))
        turn = "x (player 1)" if self.current_player == PLAYER_1 else "o (player -1)"
        status = f"to move: {turn}" if self.winner is None else f"winner: {self.winner}"
        lines.append(f"ply {self.plies} | {status}")
        return "\n".join(lines)
