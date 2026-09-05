"""Train an n-tuple afterstate value function on N x N 2048.

Plays 2048 well and is CPU-only (no neural network). Start with 3x3:

    python -m twenty48.train_afterstate --size 3 --episodes 20000
    python -m twenty48.train_afterstate --size 4 --episodes 60000 --lr 0.1
    python -m twenty48.train_afterstate --size 5 --episodes 100000

Outputs (under --out, default runs/ntuple_<N>x<N>):
    best.npz     weights with the best moving-average score
    last.npz     most recent weights
    scores.csv   episode,score,max_tile,steps
    curve.png    moving-average score curve (if matplotlib present)
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections import deque

import numpy as np

from .env import Env2048
from .ntuple import AfterstateAgent, NTupleNetwork, train_episode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train n-tuple afterstate agent for 2048")
    p.add_argument("--size", type=int, default=3, help="board size N (2..10)")
    p.add_argument("--episodes", type=int, default=20_000)
    p.add_argument("--lr", type=float, default=0.1, help="TD learning rate")
    p.add_argument("--eps", type=float, default=0.0, help="epsilon-random exploration")
    p.add_argument("--empty-bonus", type=float, default=0.0, dest="empty_bonus",
                   help="reward shaping: bonus per empty cell after a move")
    p.add_argument("--max-exp", type=int, default=15, dest="max_exp",
                   help="largest tile exponent the tables cover (15 -> tile 16384)")
    p.add_argument("--max-steps", type=int, default=None, dest="max_steps",
                   help="cap moves per episode (big boards otherwise run forever)")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or os.path.join("runs", f"ntuple_{args.size}x{args.size}")
    os.makedirs(out, exist_ok=True)

    env = Env2048(size=args.size, seed=args.seed, max_steps=args.max_steps)
    net = NTupleNetwork(size=args.size, max_exp=args.max_exp)
    agent = AfterstateAgent(net, eps=args.eps, seed=args.seed)

    n_weights = sum(t.size for t in net.tables)
    print(
        f"board {args.size}x{args.size} | {len(net.templates)} tuples | "
        f"{n_weights:,} weights | lr {args.lr} | eps {args.eps}"
    )

    window: deque[int] = deque(maxlen=200)
    tile_window: deque[int] = deque(maxlen=200)
    best_avg = -1.0
    start = time.time()

    csv_path = os.path.join(out, "scores.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["episode", "score", "max_tile", "steps"])
        scores: list[int] = []

        for ep in range(1, args.episodes + 1):
            score, max_tile, steps = train_episode(
                env, agent, lr=args.lr, empty_bonus=args.empty_bonus
            )
            scores.append(score)
            window.append(score)
            tile_window.append(max_tile)
            writer.writerow([ep, score, max_tile, steps])

            avg = float(np.mean(window))
            if ep % args.log_every == 0:
                rate = ep / (time.time() - start)
                tiles, counts = np.unique(np.array(tile_window), return_counts=True)
                top = ", ".join(
                    f"{int(t)}:{100 * c / len(tile_window):.0f}%"
                    for t, c in sorted(zip(tiles, counts), key=lambda x: -x[0])[:4]
                )
                print(
                    f"ep {ep:7d} | avg200 {avg:8.1f} | best-tile mix [{top}] "
                    f"| {rate:.0f} ep/s",
                    flush=True,
                )
                fh.flush()

            if len(window) == window.maxlen and avg > best_avg:
                best_avg = avg
                net.save(os.path.join(out, "best.npz"))
            if ep % args.save_every == 0:
                net.save(os.path.join(out, "last.npz"))

    net.save(os.path.join(out, "last.npz"))
    _plot(out, scores)
    print(f"done. best avg200 score = {best_avg:.1f}. weights in {out}/")


def _plot(out: str, scores: list[int]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not scores:
        return
    k = min(200, len(scores))
    ma = np.convolve(scores, np.ones(k) / k, mode="valid")
    plt.figure(figsize=(8, 4))
    plt.plot(scores, alpha=0.2, label="episode score")
    plt.plot(range(k - 1, len(scores)), ma, label=f"moving avg ({k})")
    plt.xlabel("episode")
    plt.ylabel("score")
    plt.legend()
    plt.title("n-tuple afterstate agent")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "curve.png"), dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
