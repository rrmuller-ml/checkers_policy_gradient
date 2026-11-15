"""Plot a train.py log: win rate (top) and mean return (bottom).

    .venv/bin/python plot_training.py checkpoints/mlp_vs_random_400it_log.txt -o checkpoints/mlp_vs_random_400it_curve.png
"""

import argparse
import re

import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT = "#9ec5f4"

ITER_RE = re.compile(r"^iter\s+(\d+) .*train win\s+(\d+)% \| mean return ([+-][\d.]+)")
EVAL_RE = re.compile(r"eval (sampled|greedy)\s+vs (random|ref)\s+\(\d+ games\): win (\d+)%")


def parse(path):
    iters, train_win, ret = [], [], []
    evals = {"sampled": [], "greedy": [], "sampled-ref": [], "greedy-ref": []}
    last_iter = 0
    for line in open(path):
        if m := ITER_RE.match(line):
            last_iter = int(m[1])
            iters.append(last_iter)
            train_win.append(int(m[2]))
            ret.append(float(m[3]))
        elif m := EVAL_RE.search(line):
            evals[m[1] if m[2] == "random" else f"{m[1]}-ref"].append((last_iter, int(m[3])))
    return np.array(iters), np.array(train_win, float), np.array(ret), evals


def rolling(x, w):
    out = np.convolve(x, np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), out])


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_2, length=0, labelsize=9)


def end_label(ax, x, y, text, color):
    ax.plot([x], [y], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=TEXT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("-o", "--out", default="training_curve.png")
    p.add_argument("--window", type=int, default=25, help="rolling-average window (iterations)")
    p.add_argument("--title", default="MLP policy trained against random")
    args = p.parse_args()

    iters, train_win, ret, evals = parse(args.log)
    w = args.window

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, facecolor=SURFACE)
    fig.subplots_adjust(left=0.08, right=0.82, top=0.9, bottom=0.08, hspace=0.32)
    fig.suptitle(args.title, x=0.08, ha="left", fontsize=13, color=TEXT, fontweight="bold")

    # win rate
    style(ax1)
    ax1.set_title("Win rate (%)", loc="left", fontsize=10, color=TEXT_2)
    tw = rolling(train_win, w)
    ax1.plot(iters, tw, color=BLUE, linewidth=2)
    ax1.plot([iters[-1]], [tw[-1]], "o", color=BLUE, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
    end_values = {"training": tw[-1]}
    for name, color, marker in (("sampled", ORANGE, "o"), ("greedy", AQUA, "s")):
        x, y = zip(*evals[name])
        ax1.plot(x, y, color=color, linewidth=2, marker=marker, markersize=7,
                 markeredgecolor=SURFACE, markeredgewidth=2)
        end_values[name] = y[-1]
    # end labels, stacked top-down in value order and spaced so they never overlap
    names = {"training": "training", "sampled": "eval, sampled", "greedy": "eval, top move"}
    label_y = 104
    for name, v in sorted(end_values.items(), key=lambda kv: -kv[1]):
        label_y = min(label_y, v)
        ax1.annotate(f"{names[name]} {v:.0f}%", (iters[-1], v), xytext=(iters[-1] + 12, label_y),
                     textcoords="data", va="center", fontsize=9, color=TEXT, annotation_clip=False)
        label_y -= 8
    ax1.axhline(50, color=TEXT_2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax1.annotate("50%", (iters[0], 50), xytext=(0, 4), textcoords="offset points", fontsize=8, color=TEXT_2)
    ax1.set_ylim(30, 105)
    ax1.legend(
        handles=[
            plt.Line2D([], [], color=BLUE, linewidth=2, label=f"training games ({w}-iter rolling avg)"),
            plt.Line2D([], [], color=ORANGE, linewidth=2, marker="o", label="eval vs random, sampled moves (50 games)"),
            plt.Line2D([], [], color=AQUA, linewidth=2, marker="s", label="eval vs random, top move (50 games)"),
        ],
        loc="lower right", fontsize=8, frameon=False, labelcolor=TEXT_2,
    )

    # mean return
    style(ax2)
    ax2.set_title("Mean discounted return per batch", loc="left", fontsize=10, color=TEXT_2)
    ax2.plot(iters, ret, color=BLUE_LIGHT, linewidth=1)
    rr = rolling(ret, w)
    ax2.plot(iters, rr, color=BLUE, linewidth=2)
    end_label(ax2, iters[-1], rr[-1], f"{w}-iter avg {rr[-1]:+.2f}", BLUE)
    ax2.annotate("per batch", (iters[5], ret[:40].min()), xytext=(0, -12), textcoords="offset points",
                 fontsize=8, color=TEXT_2)
    ax2.axhline(0, color=TEXT_2, linewidth=0.8)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_xlabel("training iteration (1 gradient step, B = 256 actions)", fontsize=9, color=TEXT_2)

    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
