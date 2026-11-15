"""Pygame UI to play checkers against a policy (or watch two policies play).

    python -m checkers.ui                                   # you vs random, 8x8
    python -m checkers.ui --size 10 --policy greedy
    python -m checkers.ui --policy my_agent.py:MyPolicy --human-player -1
    python -m checkers.ui --watch --policy greedy --policy2 random
    python -m checkers.ui --policy checkers.mlp:MLPPolicy --obs-mode board --policy-kwargs '{"checkpoint": "mlp.pt"}'

Controls: click a piece, then a highlighted square. R restart, S swap sides,
U undo, Q / Esc quit.
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

import pygame

from .encoding import action_to_squares, policy_inputs
from .game import KING, PLAYER_1, PLAYER_2, CheckersGame
from .policies import Policy, load_policy

LIGHT = (238, 222, 190)
DARK = (120, 84, 60)
P1_COLOR = (230, 230, 225)
P2_COLOR = (40, 40, 45)
SELECT = (80, 170, 250)
TARGET = (90, 200, 120)
CAPTURE = (230, 80, 70)
LAST = (245, 210, 90)
TEXT = (30, 30, 30)
BAR = (245, 242, 235)
STATUS_H = 64


class CheckersUI:
    def __init__(
        self,
        game: CheckersGame,
        policies: dict[int, Policy],
        human_player: Optional[int],
        obs_mode: str = "planes",
        cell: Optional[int] = None,
        delay_ms: int = 400,
    ):
        self.game = game
        self.policies = policies
        self.human_player = human_player
        self.obs_mode = obs_mode
        self.delay_ms = delay_ms
        self.n = game.size
        self.cell = cell or max(36, min(90, 720 // self.n))
        self.history: list[CheckersGame] = []
        self.selected = None
        self.pending_since: Optional[int] = None
        self.message = ""

        pygame.init()
        side = self.cell * self.n
        self.screen = pygame.display.set_mode((side, side + STATUS_H))
        pygame.display.set_caption("Checkers")
        self.font = pygame.font.SysFont(None, 26)
        self.small = pygame.font.SysFont(None, 20)
        self.clock = pygame.time.Clock()

    # --------------------------------------------------------- coordinates

    @property
    def flipped(self) -> bool:
        return self.human_player == PLAYER_2

    def to_screen(self, sq):
        r, c = sq
        if self.flipped:
            r, c = self.n - 1 - r, self.n - 1 - c
        return c * self.cell, r * self.cell

    def from_pixel(self, pos):
        x, y = pos
        if y >= self.n * self.cell:
            return None
        r, c = y // self.cell, x // self.cell
        if self.flipped:
            r, c = self.n - 1 - r, self.n - 1 - c
        return r, c

    # ---------------------------------------------------------------- loop

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self.on_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.on_click(event.pos)
            self.maybe_policy_move()
            self.draw()
            self.clock.tick(60)
        pygame.quit()

    def is_human_turn(self) -> bool:
        return not self.game.is_over and self.game.current_player == self.human_player

    def on_key(self, key) -> bool:
        if key in (pygame.K_q, pygame.K_ESCAPE):
            return False
        if key == pygame.K_r:
            self.restart()
        elif key == pygame.K_s and self.human_player is not None:
            self.human_player = -self.human_player
            self.policies = {-p: pol for p, pol in self.policies.items()}
            self.restart()
        elif key == pygame.K_u and self.history:
            self.game = self.history.pop()
            self.selected, self.pending_since, self.message = None, None, "undo"
        return True

    def restart(self) -> None:
        self.game.reset()
        self.history.clear()
        for pol in self.policies.values():
            pol.reset()
        self.selected, self.pending_since, self.message = None, None, ""

    def on_click(self, pos) -> None:
        if not self.is_human_turn():
            return
        sq = self.from_pixel(pos)
        if sq is None:
            return
        moves = self.game.legal_moves()
        if self.selected is not None:
            move = next((m for m in moves if m.frm == self.selected and m.to == sq), None)
            if move is not None:
                self.history.append(self.game.copy())
                self.game.step(move)
                self.message = f"you: {move.frm} -> {move.to}"
                self.selected = None
                return
        if any(m.frm == sq for m in moves):
            self.selected = sq
        else:
            self.selected = None
            piece = self.game.board[sq]
            if piece * self.human_player > 0 and self.game.forced_capture and any(m.is_capture for m in moves):
                self.message = "a capture is mandatory"

    def maybe_policy_move(self) -> None:
        if self.game.is_over or self.game.current_player == self.human_player:
            return
        now = pygame.time.get_ticks()
        if self.pending_since is None:
            self.pending_since = now
            return
        if now - self.pending_since < self.delay_ms:
            return
        self.pending_since = None
        player = self.game.current_player
        obs, mask, info = policy_inputs(self.game, self.obs_mode)
        action = int(self.policies[player].act(obs, mask, info))
        if not mask[action]:
            self.message = f"policy chose illegal action {action}!"
            return
        frm, to = action_to_squares(action, player, self.n)
        self.game.step((frm, to))
        who = "policy" if self.human_player is not None else ("white" if player == PLAYER_1 else "black")
        self.message = f"{who}: {frm} -> {to}"

    # ---------------------------------------------------------------- draw

    def draw(self) -> None:
        cs, n = self.cell, self.n
        for r in range(n):
            for c in range(n):
                x, y = self.to_screen((r, c))
                pygame.draw.rect(self.screen, DARK if (r + c) % 2 else LIGHT, (x, y, cs, cs))

        last = self.game.last_move
        if last is not None:
            for sq in (last.frm, last.to):
                x, y = self.to_screen(sq)
                pygame.draw.rect(self.screen, LAST, (x, y, cs, cs), 3)

        movable = {m.frm for m in self.game.legal_moves()} if self.is_human_turn() else set()
        for sq in movable:
            x, y = self.to_screen(sq)
            pygame.draw.rect(self.screen, SELECT if sq == self.selected else TARGET, (x, y, cs, cs), 2)

        for r in range(n):
            for c in range(n):
                piece = int(self.game.board[r, c])
                if piece:
                    self.draw_piece((r, c), piece)

        if self.selected is not None:
            for m in self.game.legal_moves():
                if m.frm == self.selected:
                    x, y = self.to_screen(m.to)
                    color = CAPTURE if m.is_capture else TARGET
                    pygame.draw.circle(self.screen, color, (x + cs // 2, y + cs // 2), cs // 6)
                    if m.is_capture:
                        cx, cy = self.to_screen(m.captured)
                        pygame.draw.circle(self.screen, CAPTURE, (cx + cs // 2, cy + cs // 2), cs // 2 - 4, 3)

        self.draw_status()
        pygame.display.flip()

    def draw_piece(self, sq, piece: int) -> None:
        cs = self.cell
        x, y = self.to_screen(sq)
        center = (x + cs // 2, y + cs // 2)
        radius = int(cs * 0.38)
        fill = P1_COLOR if piece > 0 else P2_COLOR
        edge = P2_COLOR if piece > 0 else P1_COLOR
        pygame.draw.circle(self.screen, (0, 0, 0), (center[0] + 2, center[1] + 3), radius)
        pygame.draw.circle(self.screen, fill, center, radius)
        pygame.draw.circle(self.screen, edge, center, radius, 2)
        if abs(piece) == KING:
            pygame.draw.circle(self.screen, LAST, center, radius // 2, max(2, cs // 18))

    def draw_status(self) -> None:
        top = self.n * self.cell
        pygame.draw.rect(self.screen, BAR, (0, top, self.n * self.cell, STATUS_H))
        g = self.game
        names = {PLAYER_1: "white", PLAYER_2: "black"}
        if g.is_over:
            if g.winner == 0:
                line = "Draw."
            elif self.human_player is None:
                line = f"{names[g.winner].capitalize()} wins."
            else:
                line = "You win!" if g.winner == self.human_player else "Policy wins."
        elif self.is_human_turn():
            line = f"Your move ({names[self.human_player]})"
        else:
            line = f"Policy thinking ({names[g.current_player]})..."
        counts = g.piece_counts()
        stats = (
            f"white {counts[PLAYER_1]['men']}+{counts[PLAYER_1]['kings']}K  "
            f"black {counts[PLAYER_2]['men']}+{counts[PLAYER_2]['kings']}K  ply {g.plies}"
        )
        self.screen.blit(self.font.render(line, True, TEXT), (10, top + 8))
        self.screen.blit(self.small.render(f"{stats}   {self.message}", True, TEXT), (10, top + 36))
        hint = self.small.render("R restart  S swap  U undo  Q quit", True, (110, 110, 110))
        self.screen.blit(hint, (self.n * self.cell - hint.get_width() - 10, top + 10))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Play checkers against a policy.")
    p.add_argument("--size", type=int, default=8, help="board side length (N x N)")
    p.add_argument("--policy", default="random", help="'random', 'greedy', 'module:Name' or 'file.py:Name'")
    p.add_argument("--policy-kwargs", default="{}", help='JSON kwargs for --policy, e.g. \'{"checkpoint": "mlp.pt"}\'')
    p.add_argument("--policy2", default="random", help="second policy for --watch")
    p.add_argument("--human-player", type=int, choices=[1, -1], default=1, help="1 = white (moves first), -1 = black")
    p.add_argument("--watch", action="store_true", help="policy (white) vs policy2 (black), no human")
    p.add_argument("--obs-mode", choices=["planes", "board"], default="planes", help="observation format the policy expects")
    p.add_argument("--no-forced-capture", action="store_true", help="make captures optional")
    p.add_argument("--no-flying-kings", action="store_true", help="kings move a single square")
    p.add_argument("--delay", type=int, default=400, help="ms before a policy moves")
    args = p.parse_args(argv)
    policy_kwargs = json.loads(args.policy_kwargs)

    game = CheckersGame(
        size=args.size, forced_capture=not args.no_forced_capture, flying_kings=not args.no_flying_kings
    )
    if args.watch:
        human = None
        policies = {PLAYER_1: load_policy(args.policy, **policy_kwargs), PLAYER_2: load_policy(args.policy2)}
    else:
        human = args.human_player
        policies = {-human: load_policy(args.policy, **policy_kwargs)}
    CheckersUI(game, policies, human, obs_mode=args.obs_mode, delay_ms=args.delay).run()


if __name__ == "__main__":
    main()
