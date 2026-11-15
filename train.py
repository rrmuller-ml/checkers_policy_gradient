"""Minimal training loop: MLP policy vs a random opponent.

Plays full games, records every agent action with its reward, and yields
batches of exactly B actions. Write your loss in ``compute_loss``.

    .venv/bin/python train.py --batch-size 256 --iters 200
"""

import argparse
import copy
from collections import deque

import numpy as np
import torch

from checkers import CheckersEnv, RandomPolicy
from checkers.mlp import MLPPolicy, MLPPolicyNet, load_mlp, save_mlp


def play_episode(env, net, seed=None, greedy=False):
    """Play one full game. Returns one list entry per agent action.
    greedy=True picks the most likely valid action instead of sampling."""
    obs, info = env.reset(seed=seed)
    steps = []
    done = False
    while not done:
        mask = info["action_mask"]
        with torch.no_grad():
            dist = net.distribution(obs, mask)
            action = dist.probs.argmax(-1) if greedy else dist.sample()
            log_prob = dist.log_prob(action)
        next_obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        steps.append(
            {
                "obs": obs,                      # (N, N) board before the action
                "mask": mask,                    # (N**4,) valid actions at that state
                "action": int(action),           # action taken (from_idx * N*N + to_idx)
                "reward": float(reward),         # reward received for this action
                "log_prob": float(log_prob),     # log pi(action) when it was sampled
                "done": done,                    # True on the last action of the game
            }
        )
        obs = next_obs
    return steps, info["winner"]


def add_returns(steps, gamma):
    """Discounted reward-to-go, per action of one episode:
    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...  (future rewards only, from t to game end).
    With gamma=1 and only a terminal win/loss reward, every action gets the game outcome (+1/-1/0).
    """
    g = 0.0
    for step in reversed(steps):
        g = step["reward"] + gamma * g
        step["return"] = g
    return steps


def to_batch(steps):
    """List of step dicts -> dict of tensors, each with leading dim B."""
    return {
        "obs": torch.as_tensor(np.stack([s["obs"] for s in steps]), dtype=torch.float32),  # (B, N, N)
        "mask": torch.as_tensor(np.stack([s["mask"] for s in steps])),                      # (B, N**4) bool
        "actions": torch.tensor([s["action"] for s in steps]),                              # (B,)
        "rewards": torch.tensor([s["reward"] for s in steps], dtype=torch.float32),         # (B,)
        "returns": torch.tensor([s["return"] for s in steps], dtype=torch.float32),         # (B,)
        "old_log_probs": torch.tensor([s["log_prob"] for s in steps], dtype=torch.float32), # (B,)
        "dones": torch.tensor([s["done"] for s in steps]),                                   # (B,)
    }


def compute_loss(net, batch, is_clip=None, center=False):
    """Your loss here. ``is_clip=(lo, hi)`` turns on clipped importance sampling;
    ``center=True`` turns on score centering (needs batch["behaviour_probs"]).

    Useful pieces:
        dist = net.distribution(batch["obs"], batch["mask"])   # softmax over valid actions only
        log_probs = dist.log_prob(batch["actions"])            # (B,), differentiable
        entropy = dist.entropy()                               # (B,)
        batch["rewards"], batch["returns"], batch["old_log_probs"], batch["dones"]
    """
    dist = net.distribution(batch["obs"], batch["mask"])
    log_probs = dist.log_prob(batch["actions"])
    if is_clip is None:
        weight = torch.ones_like(log_probs)
    else:
        # data came from a stale behaviour policy mu: reweight each action by
        # w = pi(a|s) / mu(a|s), clipped, and treated as a constant
        ratio = torch.exp(log_probs.detach() - batch["old_log_probs"])
        weight = ratio.clamp(*is_clip)
    # score(a) = clip(w(a)) * dlog pi(a|s)/dtheta, expressed through its surrogate
    score = weight * log_probs
    if center:
        # score_new(a) = score(a) - E_{a'~mu(.|s)}[clip(w(a')) * dlog pi(a'|s)/dtheta],
        # summed over every valid action a' with its behaviour probability mu(a'|s).
        # mu(a') * clip(pi(a')/mu(a')) = clamp(pi(a'), lo*mu(a'), hi*mu(a')), no division needed.
        mu = batch["behaviour_probs"]
        pi = dist.probs.detach()
        coef = mu if is_clip is None else torch.clamp(pi, min=is_clip[0] * mu, max=is_clip[1] * mu)
        all_log_probs = torch.where(batch["mask"], dist.logits, torch.zeros_like(dist.logits))
        score = score - (coef * all_log_probs).sum(-1)
    return -(batch["returns"] * score).mean()


@torch.no_grad()
def is_stats(net, batch, is_clip):
    """Mean IS ratio and the fraction of actions whose ratio was clipped."""
    log_probs = net.distribution(batch["obs"], batch["mask"]).log_prob(batch["actions"])
    ratio = torch.exp(log_probs - batch["old_log_probs"])
    lo, hi = is_clip or (0.0, float("inf"))
    return ratio.mean().item(), ((ratio < lo) | (ratio > hi)).float().mean().item()


def evaluate(net, n_games=50, size=8, greedy=False, opponent_net=None):
    """Play n_games (half per colour), sampling or greedy; return win/draw/loss rates.
    Opponent: random, or a frozen ``opponent_net`` that samples its moves."""
    results = []
    for side in (1, -1):
        if opponent_net is None:
            opponent = RandomPolicy(seed=123 + side)
        else:
            opponent = MLPPolicy(net=opponent_net, seed=123 + side)
        env = CheckersEnv(size=size, obs_mode="board", agent_player=side, opponent=opponent)
        for ep in range(n_games // 2):
            _, winner = play_episode(env, net, seed=10_000 + ep, greedy=greedy)
            results.append(0 if winner == 0 else (1 if winner == side else -1))
    results = np.array(results)
    return (results == 1).mean(), (results == 0).mean(), (results == -1).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch-size", "-B", type=int, default=256, help="actions per gradient step")
    p.add_argument("--iters", type=int, default=200, help="number of gradient steps")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--save", default="mlp.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", help="checkpoint to start training from")
    p.add_argument("--opponent", default="random",
                   help="'random', 'self' (frozen copies of the learner), or a policy checkpoint path (sampled moves)")
    p.add_argument("--opponent-update-every", type=int, default=50, help="self-play: replace the frozen opponent with a new copy every N iters")
    p.add_argument("--eval-ref", help="checkpoint of a reference model to also evaluate against")
    p.add_argument("--eval-games", type=int, default=50)
    p.add_argument("--stale-k", type=int, default=0,
                   help="roll out with the policy snapshot from k iters ago (0 = on-policy)")
    p.add_argument("--snapshot-every", type=int, default=5,
                   help="with --stale-k: keep a policy snapshot every N iters (k must be a multiple of N)")
    p.add_argument("--is-clip", type=float, nargs=2, metavar=("LO", "HI"),
                   help="importance-sampling correction with ratios clipped to [LO, HI], e.g. 0.5 2.0")
    p.add_argument("--score-center", action="store_true",
                   help="subtract E_{a~behaviour}[clip(w) dlog pi(a)] from each action's score")
    p.add_argument("--save-best", help="save the checkpoint with the best eval win rate here (vs ref if given, else vs random greedy)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    net = load_mlp(args.init) if args.init else MLPPolicyNet(size=args.size, hidden=args.hidden)
    ref_net = load_mlp(args.eval_ref) if args.eval_ref else None

    def frozen_self():
        """Frozen copy of the current network, sampling its moves."""
        return MLPPolicy(net=copy.deepcopy(net), seed=args.seed + episode)

    episode = 0
    env = CheckersEnv(
        size=args.size,
        obs_mode="board",
        agent_player="random",          # learn to play both colours
        opponent=(
            RandomPolicy(seed=args.seed) if args.opponent == "random"
            else frozen_self() if args.opponent == "self"
            else MLPPolicy(net=load_mlp(args.opponent), seed=args.seed)
        ),
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    best_win = -1.0
    behaviour = net
    if args.stale_k:
        if args.stale_k % args.snapshot_every:
            raise SystemExit("--stale-k must be a multiple of --snapshot-every")
        lag = args.stale_k // args.snapshot_every
        snapshots = deque(maxlen=lag + 1)  # in memory; oldest = the one lag snapshots ago
    for it in range(1, args.iters + 1):
        # 1) roll out full games until we have at least B actions. With --stale-k the games
        #    are played by an old frozen copy of the policy: a snapshot is kept every N iters
        #    and the one k/N snapshots ago is used (staleness k..k+N-1; before that much
        #    history exists, the oldest snapshot)
        if args.stale_k:
            if (it - 1) % args.snapshot_every == 0:
                snapshots.append(copy.deepcopy(net))
            behaviour = snapshots[0]
        buffer = []
        wins = games = 0
        while len(buffer) < args.batch_size:
            steps, winner = play_episode(env, behaviour, seed=args.seed * 1_000_000 + episode)
            buffer.extend(add_returns(steps, args.gamma))
            episode += 1
            games += 1
            wins += winner == env.agent_player

        # 2) take exactly B actions and discard the rest: nothing carries over to the
        #    next batch. Returns were computed over full episodes, so they stay correct.
        batch = to_batch(buffer[: args.batch_size])

        # 3) one gradient step per batch
        if args.score_center:
            with torch.no_grad():
                batch["behaviour_probs"] = behaviour.distribution(batch["obs"], batch["mask"]).probs
        is_mean, is_clipped = is_stats(net, batch, args.is_clip)
        loss = compute_loss(net, batch, args.is_clip, args.score_center)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"iter {it:4d} | loss {loss.item():+.4f} | games {games:3d} | "
            f"train win {wins / max(games, 1):.0%} | mean return {batch['returns'].mean():+.3f} | "
            f"IS ratio mean {is_mean:.3f} clipped {is_clipped:.0%}"
        )
        if args.opponent == "self" and it % args.opponent_update_every == 0:
            env.opponent = frozen_self()
            print(f"  opponent <- frozen copy of policy at iter {it}")
        if it % args.eval_every == 0:
            n = args.eval_games
            for greedy in (False, True):
                w, d, l = evaluate(net, n_games=n, size=args.size, greedy=greedy)
                mode = "greedy " if greedy else "sampled"
                print(f"  eval {mode} vs random ({n} games): win {w:.0%} draw {d:.0%} loss {l:.0%}")
            if ref_net is not None:
                for greedy in (False, True):
                    w, d, l = evaluate(net, n_games=n, size=args.size, greedy=greedy, opponent_net=ref_net)
                    mode = "greedy " if greedy else "sampled"
                    print(f"  eval {mode} vs ref    ({n} games): win {w:.0%} draw {d:.0%} loss {l:.0%}")
            save_mlp(net, args.save)
            if args.save_best and w > best_win:
                best_win = w
                save_mlp(net, args.save_best)
                print(f"  new best ({w:.1%}) -> {args.save_best}")


if __name__ == "__main__":
    main()
