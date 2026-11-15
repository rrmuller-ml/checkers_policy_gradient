import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from checkers import (
    CheckersEnv,
    CheckersGame,
    GreedyPolicy,
    RandomPolicy,
    RewardConfig,
    decode_action,
    encode_action,
    load_policy,
    policy_inputs,
)
from checkers.encoding import action_to_squares, move_to_action


def empty(n=8):
    return np.zeros((n, n), dtype=np.int8)


def moves(game):
    return {(m.frm, m.to) for m in game.legal_moves()}


def test_initial_position():
    g = CheckersGame(size=8)
    assert (g.board == 1).sum() == 12 and (g.board == -1).sum() == 12
    assert len(g.legal_moves()) == 7
    g10 = CheckersGame(size=10)
    assert (g10.board == 1).sum() == 20


def test_man_moves_forward_only():
    b = empty()
    b[5, 2] = 1
    b[0, 1] = -1
    assert moves(CheckersGame.from_board(b)) == {((5, 2), (4, 1)), ((5, 2), (4, 3))}


def test_man_captures_backward_and_single_capture_only():
    b = empty()
    b[4, 3] = 1
    b[5, 4] = -1  # behind the man
    b[2, 3] = -1  # would be a second jump in a multi-jump game
    b[0, 7] = -1
    g = CheckersGame.from_board(b)
    assert moves(g) == {((4, 3), (6, 5))}  # forced capture, backwards
    g.step(((4, 3), (6, 5)))
    assert g.board[5, 4] == 0 and g.current_player == -1  # turn ends after one capture


def test_optional_capture():
    b = empty()
    b[4, 3] = 1
    b[3, 4] = -1
    b[0, 7] = -1
    g = CheckersGame.from_board(b, forced_capture=False)
    assert ((4, 3), (2, 5)) in moves(g) and ((4, 3), (3, 2)) in moves(g)


def test_promotion_and_flying_king():
    b = empty()
    b[1, 2] = 1
    b[5, 0] = -1
    g = CheckersGame.from_board(b)
    g.step(((1, 2), (0, 1)))
    assert g.board[0, 1] == 2
    g.step(((5, 0), (6, 1)))
    king_moves = {m.to for m in g.legal_moves() if m.frm == (0, 1)}
    assert {(1, 0), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)} == king_moves


def test_flying_king_captures_from_distance_lands_right_behind():
    b = empty()
    b[7, 0] = 2
    b[4, 3] = -1
    b[0, 1] = -1
    g = CheckersGame.from_board(b)
    assert moves(g) == {((7, 0), (3, 4))}
    g.step(((7, 0), (3, 4)))
    assert g.board[4, 3] == 0 and g.board[3, 4] == 2


def test_flying_king_capture_blocked_behind():
    b = empty()
    b[7, 0] = 2
    b[4, 3] = -1
    b[3, 4] = -1  # square right behind is occupied -> no capture
    b[0, 1] = -1
    g = CheckersGame.from_board(b)
    assert not any(m.is_capture for m in g.legal_moves())


def test_king_cannot_jump_two_pieces():
    b = empty()
    b[7, 0] = 2
    b[5, 2] = -1
    b[4, 3] = -1
    b[0, 1] = -1
    g = CheckersGame.from_board(b)
    assert all(m.frm != (7, 0) or not m.is_capture for m in g.legal_moves())


def test_non_flying_kings():
    b = empty()
    b[4, 3] = 2
    b[0, 1] = -1
    g = CheckersGame.from_board(b, flying_kings=False)
    assert {m.to for m in g.legal_moves()} == {(3, 2), (3, 4), (5, 2), (5, 4)}


def test_no_moves_loses():
    b = empty()
    b[7, 0] = 1
    b[6, 1] = -1
    b[5, 2] = -1  # white man blocked and can't capture
    g = CheckersGame.from_board(b)
    assert g.winner == -1


def test_action_encoding_roundtrip():
    n = 8
    for frm, to in [((5, 0), (4, 1)), ((2, 3), (4, 5))]:
        assert decode_action(encode_action(frm, to, n), n) == (frm, to)
    g = CheckersGame()
    for m in g.legal_moves():
        for player in (1, -1):
            assert action_to_squares(move_to_action(m, player, n), player, n) == (m.frm, m.to)


def test_canonical_view_is_symmetric():
    g = CheckersGame()
    obs1, mask1, _ = policy_inputs(g)
    g.step(g.legal_moves()[0])
    g2 = CheckersGame()
    obs_black, mask_black, _ = policy_inputs(g2)
    # at the start both sides see the same board from their own perspective
    obs_p2, _, _ = policy_inputs(CheckersGame.from_board(-CheckersGame().board[::-1, ::-1], current_player=-1))
    assert np.array_equal(obs1, obs_p2)


@pytest.mark.parametrize("agent_player", [1, -1, "random"])
@pytest.mark.parametrize("obs_mode", ["planes", "board"])
def test_env_random_rollouts(agent_player, obs_mode):
    env = CheckersEnv(size=8, agent_player=agent_player, obs_mode=obs_mode)
    agent = RandomPolicy(seed=0)
    for ep in range(5):
        obs, info = env.reset(seed=ep)
        assert env.observation_space.contains(obs)
        done = False
        total = 0.0
        while not done:
            mask = info["action_mask"]
            assert mask.sum() > 0 and np.array_equal(mask, env.action_masks())
            obs, reward, term, trunc, info = env.step(agent.act(obs, mask, info))
            total += reward
            done = term or trunc
        assert total in (-1.0, 0.0, 1.0)
        assert info["winner"] in (-1, 0, 1)


def test_env_checker():
    check_env(CheckersEnv(illegal_action="penalize"), skip_render_check=True)


def test_illegal_action():
    env = CheckersEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(0)
    env = CheckersEnv(illegal_action="penalize")
    env.reset(seed=0)
    _, reward, term, _, info = env.step(0)
    assert reward == -1.0 and term and info["illegal_action"]


def test_shaped_rewards_and_win():
    b = empty()
    b[4, 3] = 1
    b[3, 4] = -1
    env = CheckersEnv(rewards=RewardConfig(capture=0.1))
    env.reset()
    env.game = CheckersGame.from_board(b)
    _, reward, term, _, info = env.step(env.action_from_squares((4, 3), (2, 5)))
    assert term and info["winner"] == 1 and reward == pytest.approx(1.1)


def test_greedy_prefers_king_capture():
    b = empty()
    b[4, 3] = 1
    b[3, 2] = -1
    b[3, 4] = -2
    g = CheckersGame.from_board(b)
    obs, mask, info = policy_inputs(g)
    assert action_to_squares(GreedyPolicy(seed=0).act(obs, mask, info), 1, 8) == ((4, 3), (2, 5))


def test_load_policy_and_callable_opponent(tmp_path):
    f = tmp_path / "my_policy.py"
    f.write_text("import numpy as np\ndef first(obs, mask, info):\n    return int(np.flatnonzero(mask)[0])\n")
    pol = load_policy(f"{f}:first")
    env = CheckersEnv(opponent=pol, size=6)
    obs, info = env.reset()
    env.step(info["valid_actions"][0])
    assert isinstance(load_policy("checkers.policies:GreedyPolicy"), GreedyPolicy)
