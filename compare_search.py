"""Does policy+value tree search beat the plain policy?

Each agent plays n games (half per colour) against each opponent.

    .venv/bin/python compare_search.py --games 200
"""

import argparse
import time

import numpy as np

from checkers import CheckersEnv, GreedyPolicy, RandomPolicy, policy_inputs
from checkers.mlp import MLPPolicy, load_mlp
from checkers.search import SearchPolicy
from checkers.value import load_value


def play(agent, make_opponent, n_games, size):
    results = []
    for side in (1, -1):
        env = CheckersEnv(size=size, obs_mode="board", agent_player=side, opponent=make_opponent(side))
        for ep in range(n_games // 2):
            env.reset(seed=ep)
            agent.reset(seed=ep)
            done = False
            while not done:
                obs, mask, info = policy_inputs(env.game, "board")
                _, _, term, trunc, info = env.step(agent.act(obs, mask, info))
                done = term or trunc
            results.append(0 if info["winner"] == 0 else (1 if info["winner"] == side else -1))
    r = np.array(results)
    return (r == 1).mean(), (r == 0).mean(), (r == -1).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="checkpoints/mlp_vs_random_400it.pt")
    p.add_argument("--value", default="checkpoints/value_vs_random.pt")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--width", type=int, default=2)
    p.add_argument("--opponents", nargs="+", default=["random", "greedy-capture", "policy"],
                   help="opponents to play: random, greedy-capture, policy (the --policy net, sampled) "
                        "or a checkpoint path (sampled)")
    args = p.parse_args()

    policy_net, value_net = load_mlp(args.policy), load_value(args.value)
    size = policy_net.size
    agents = {
        "policy (top move)": MLPPolicy(net=policy_net, deterministic=True),
        f"search d{args.depth} w{args.width} min": SearchPolicy(policy_net, value_net, args.depth, args.width, "min"),
        f"search d{args.depth} w{args.width} mean": SearchPolicy(policy_net, value_net, args.depth, args.width, "mean"),
    }
    builtin = {
        "random": lambda side: RandomPolicy(seed=100 + side),
        "greedy-capture": lambda side: GreedyPolicy(seed=200 + side),
        "policy": lambda side: MLPPolicy(net=policy_net, seed=300 + side),
    }
    opponents = {}
    for name in args.opponents:
        if name in builtin:
            opponents[name if name != "policy" else "policy (sampled)"] = builtin[name]
        else:
            opp_net = load_mlp(name)
            opponents[f"{name} (sampled)"] = lambda side, n=opp_net: MLPPolicy(net=n, seed=500 + side)
    print(f"{args.games} games per pairing (half per colour); win / draw / loss\n")
    for opp_name, make_opp in opponents.items():
        print(f"vs {opp_name}")
        for agent_name, agent in agents.items():
            t = time.time()
            w, d, l = play(agent, make_opp, args.games, size)
            print(f"  {agent_name:22s} {w:5.1%} / {d:5.1%} / {l:5.1%}   ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
