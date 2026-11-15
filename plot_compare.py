"""Overlay several train.py runs: eval win rate (top) and rolling mean return (bottom).

Each run is ``log_path:label``; give several seeds as ``log1,log2,log3:label`` to plot
the mean across seeds with a shaded min-max band.

    .venv/bin/python plot_compare.py run1_log.txt:"label 1" a_s0.txt,a_s1.txt,a_s2.txt:"label 2" -o compare.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from plot_training import AQUA, BLUE, ORANGE, SURFACE, TEXT, TEXT_2, parse, rolling, style

COLORS = (BLUE, ORANGE, AQUA, "#eda100")  # categorical slots 1-4, validated together
MARKERS = ("o", "s", "D", "^")


def spaced_labels(ax, x, items, top, gap):
    """Direct labels at the line ends, pushed apart so they never overlap."""
    label_y = top
    for text, y in sorted(items, key=lambda t: -t[1]):
        label_y = min(label_y, y)
        ax.annotate(text, (x, y), xytext=(x + 12, label_y), textcoords="data", va="center",
                    fontsize=9, color=TEXT, annotation_clip=False)
        label_y -= gap


def load_seeds(paths, eval_mode, window, max_iter=None):
    """Per-seed eval win rates and rolling returns, stacked as (n_seeds, T) arrays."""
    eval_x, wins, rets = None, [], []
    for path in paths:
        iters, _, ret, evals = parse(path)
        if max_iter is not None:
            keep = iters <= max_iter
            iters, ret = iters[keep], ret[keep]
            evals = {k: [(i, v) for i, v in e if i <= max_iter] for k, e in evals.items()}
        x, y = zip(*evals[eval_mode])
        eval_x = np.array(x)
        wins.append(y)
        rets.append(rolling(ret, window))
    return eval_x, np.array(wins, float), iters, np.array(rets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="log_path:label, or log1,log2,...:label to average seeds")
    p.add_argument("-o", "--out", default="compare.png")
    p.add_argument("--window", type=int, default=25)
    p.add_argument("--title", default="Training against random")
    p.add_argument("--eval", choices=["sampled", "greedy", "sampled-ref", "greedy-ref"], default="sampled",
                   help="which eval to plot; *-ref = against the --eval-ref model")
    p.add_argument("--max-iter", type=int, help="only plot up to this iteration")
    p.add_argument("--opponent-name", default="random", help="name of the eval opponent, for the title")
    p.add_argument("--short", nargs="+", help="short names for the line-end labels (default: the labels)")
    args = p.parse_args()
    if len(args.runs) > len(COLORS):
        raise SystemExit(f"at most {len(COLORS)} runs")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, facecolor=SURFACE)
    fig.subplots_adjust(left=0.08, right=0.8, top=0.9, bottom=0.08, hspace=0.32)
    fig.suptitle(args.title, x=0.08, ha="left", fontsize=13, color=TEXT, fontweight="bold")
    style(ax1)
    style(ax2)
    moves = "sampled moves" if args.eval.startswith("sampled") else "top move"
    vs = args.opponent_name if args.eval.endswith("-ref") else "random"
    n_seeds = max(len(spec.partition(":")[0].split(",")) for spec in args.runs)
    across = f", mean of {n_seeds} seeds (band = min-max)" if n_seeds > 1 else ""
    ax1.set_title(f"Eval win rate vs {vs}, {moves} (%){across}", loc="left", fontsize=10, color=TEXT_2)
    ax2.set_title(f"Mean discounted return per batch ({args.window}-iter rolling avg)", loc="left",
                  fontsize=10, color=TEXT_2)

    ends1, ends2, handles = [], [], []
    lows1, lows2 = [], []
    shorts = args.short or [spec.partition(":")[2] for spec in args.runs]
    for (spec, short, color, marker) in zip(args.runs, shorts, COLORS, MARKERS):
        paths, _, label = spec.partition(":")
        eval_x, wins, iters, rets = load_seeds(paths.split(","), args.eval, args.window, args.max_iter)
        win_mean, ret_mean = wins.mean(0), rets.mean(0)
        lows1.append(wins.min())
        lows2.append(np.nanmin(rets))
        if len(wins) > 1:
            ax1.fill_between(eval_x, wins.min(0), wins.max(0), color=color, alpha=0.15, linewidth=0)
            ax2.fill_between(iters, rets.min(0), rets.max(0), color=color, alpha=0.15, linewidth=0)
        ax1.plot(eval_x, win_mean, color=color, linewidth=2, marker=marker, markersize=7,
                 markeredgecolor=SURFACE, markeredgewidth=2)
        ax2.plot(iters, ret_mean, color=color, linewidth=2)
        ax2.plot([iters[-1]], [ret_mean[-1]], marker, color=color, markersize=6, markeredgecolor=SURFACE,
                 markeredgewidth=2)
        ends1.append((f"{short} {win_mean[-1]:.0f}%", win_mean[-1]))
        ends2.append((f"{short} {ret_mean[-1]:+.2f}", ret_mean[-1]))
        handles.append(plt.Line2D([], [], color=color, linewidth=2, marker=marker, label=label))

    spaced_labels(ax1, eval_x[-1], ends1, top=104, gap=6 if min(lows1) >= 40 else 8)
    spaced_labels(ax2, iters[-1], ends2, top=1.0, gap=0.12)
    ax1.axhline(50, color=TEXT_2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax1.set_ylim(40 if min(lows1) >= 40 else 0, 105)
    ax1.legend(handles=handles, loc="upper left", fontsize=8, frameon=False, labelcolor=TEXT_2)
    ax2.axhline(0, color=TEXT_2, linewidth=0.8)
    ax2.set_ylim(-0.3 if min(lows2) >= -0.3 else -1.05, 1.0)
    ax2.set_xlabel("training iteration (1 gradient step, B = 256 actions)", fontsize=9, color=TEXT_2)

    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
