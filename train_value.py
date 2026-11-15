"""Fit a value network V(s) for a frozen policy playing against random.

The policy is never updated: it only generates games. Each batch is B states
(the board before each agent action) with the discounted return from that state.

    .venv/bin/python train_value.py --policy checkpoints/mlp_vs_random_400it.pt --iters 400
"""

import argparse

import numpy as np
import torch

from checkers import CheckersEnv, RandomPolicy
from checkers.mlp import load_mlp
from checkers.value import ValueNet, save_value
from train import add_returns, play_episode, to_batch


def collect_states(env, policy_net, n, gamma, seed_offset):
    """Play full games until at least n states; return the first n as a batch."""
    buffer, episode = [], 0
    while len(buffer) < n:
        steps, _ = play_episode(env, policy_net, seed=seed_offset + episode)
        buffer.extend(add_returns(steps, gamma))
        episode += 1
    return to_batch(buffer[:n]), episode


def compute_value_loss(value_net, batch):
    """MSE between predicted V(s) and the observed return from s."""
    return ((value_net(batch["obs"]) - batch["returns"]) ** 2).mean()


@torch.no_grad()
def evaluate(value_net, batch):
    """Held-out MSE, and R^2 = 1 - MSE / Var(returns) (0 = no better than predicting the mean)."""
    mse = ((value_net(batch["obs"]) - batch["returns"]) ** 2).mean().item()
    var = batch["returns"].var(unbiased=False).item()
    return mse, 1 - mse / var


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="checkpoints/mlp_vs_random_400it.pt", help="frozen policy checkpoint")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch-size", "-B", type=int, default=256, help="states per gradient step")
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--eval-states", type=int, default=5000, help="size of the held-out set")
    p.add_argument("--save", default="checkpoints/value_vs_random.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    policy_net = load_mlp(args.policy)
    policy_net.requires_grad_(False)  # frozen: only used to play
    env = CheckersEnv(
        size=policy_net.size,
        obs_mode="board",
        agent_player="random",
        opponent=RandomPolicy(seed=args.seed),
    )
    value_net = ValueNet(size=policy_net.size, hidden=args.hidden)
    optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr)

    # held-out games (different seeds from training)
    eval_batch, eval_games = collect_states(env, policy_net, args.eval_states, args.gamma, seed_offset=10**9)
    print(f"held-out: {args.eval_states} states from {eval_games} games, mean return {eval_batch['returns'].mean():+.3f}")

    episode = 0
    for it in range(1, args.iters + 1):
        batch, games = collect_states(env, policy_net, args.batch_size, args.gamma, seed_offset=episode)
        episode += games

        loss = compute_value_loss(value_net, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 10 == 0:
            print(f"iter {it:4d} | train mse {loss.item():.4f} | games {games:3d}")
        if it % args.eval_every == 0:
            mse, r2 = evaluate(value_net, eval_batch)
            print(f"  held-out mse {mse:.4f} | R^2 {r2:+.3f}")
            save_value(value_net, args.save)

    start = CheckersEnv(size=policy_net.size, obs_mode="board")
    obs, _ = start.reset()
    print(f"V(initial position, moving first) = {value_net(obs).item():+.3f}")


if __name__ == "__main__":
    main()
