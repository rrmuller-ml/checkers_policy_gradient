import numpy as np
import torch

from checkers import CheckersEnv, load_policy
from checkers.mlp import MLPPolicy, MLPPolicyNet, load_mlp, save_mlp


def test_masked_softmax_only_on_valid_actions():
    env = CheckersEnv(obs_mode="board")
    obs, info = env.reset(seed=0)
    net = MLPPolicyNet(size=8, hidden=32)
    logits = net(obs, info["action_mask"])
    assert logits.shape == (1, 8 ** 4)
    probs = net.distribution(obs, info["action_mask"]).probs[0]
    valid = torch.as_tensor(info["action_mask"])
    assert torch.all(probs[~valid] == 0)
    assert torch.isclose(probs[valid].sum(), torch.tensor(1.0))


def test_batched_forward_and_planes_input():
    env = CheckersEnv(obs_mode="planes", size=6)
    obs, info = env.reset(seed=0)
    net = MLPPolicyNet(size=6, hidden=16, in_features=5 * 36)
    batch_obs = np.stack([obs, obs])
    batch_mask = np.stack([info["action_mask"], info["action_mask"]])
    assert net(batch_obs, batch_mask).shape == (2, 6 ** 4)
    assert net(obs, info["action_mask"]).shape == (1, 6 ** 4)


def test_mlp_policy_plays_full_games_as_opponent_and_agent():
    env = CheckersEnv(obs_mode="board", opponent=MLPPolicy(size=8, seed=0))
    agent = MLPPolicy(size=8, seed=1)
    for ep in range(3):
        obs, info = env.reset(seed=ep)
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(agent.act(obs, info["action_mask"]))
            done = term or trunc


def test_gradients_flow_through_log_prob():
    env = CheckersEnv(obs_mode="board")
    obs, info = env.reset(seed=0)
    net = MLPPolicyNet(size=8, hidden=32)
    dist = net.distribution(obs, info["action_mask"])
    loss = -dist.log_prob(dist.sample()).sum()
    loss.backward()
    assert net.net[1].weight.grad is not None and torch.isfinite(net.net[1].weight.grad).all()


def test_save_load_and_load_policy(tmp_path):
    net = MLPPolicyNet(size=6, hidden=16)
    path = str(tmp_path / "mlp.pt")
    save_mlp(net, path)
    loaded = load_mlp(path)
    x = torch.randn(1, 6, 6)
    assert torch.allclose(net(x), loaded(x))
    pol = load_policy("checkers.mlp:MLPPolicy", checkpoint=path, deterministic=True)
    assert isinstance(pol, MLPPolicy) and pol.net.size == 6
