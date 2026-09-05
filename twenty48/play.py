"""Watch an agent play, or run a random baseline, or play yourself.

Examples
--------
    python -m twenty48.play --size 4 --checkpoint runs/ntuple_4x4/best.npz --games 5
    python -m twenty48.play --size 3 --random --games 100      # baseline stats
    python -m twenty48.play --size 4 --human                   # w/a/s/d + q
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import board as B
from .env import Env2048

KEYMAP = {"w": B.UP, "s": B.DOWN, "a": B.LEFT, "d": B.RIGHT}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play / evaluate 2048")
    p.add_argument("--size", type=int, default=4)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--random", action="store_true", help="random legal-move baseline")
    p.add_argument("--human", action="store_true", help="play from the keyboard")
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--delay", type=float, default=0.15, help="seconds between rendered moves")
    p.add_argument("--quiet", action="store_true", help="no board rendering, stats only")
    return p.parse_args()


def _load_agent(env: Env2048, checkpoint: str):
    """Load a trained n-tuple afterstate agent (.npz)."""
    from .ntuple import AfterstateAgent, NTupleNetwork

    return AfterstateAgent(NTupleNetwork.load(checkpoint))


def _pick_action(mode, env, obs, info, agent):
    legal = info["legal_actions"]
    if mode == "human":
        while True:
            key = input("move [w/a/s/d, q=quit]: ").strip().lower()
            if key == "q":
                return None
            if key in KEYMAP and KEYMAP[key] in legal:
                return KEYMAP[key]
            print("  illegal / unknown, try again")
    if mode == "random":
        return int(np.random.choice(legal))
    return agent.act(obs, legal, greedy=True, board=env.board)


def main() -> None:
    args = parse_args()
    mode = "human" if args.human else "random" if args.random else "agent"
    if mode == "agent" and not args.checkpoint:
        raise SystemExit("agent mode needs --checkpoint (or use --random / --human)")

    env = Env2048(size=args.size, seed=args.seed)
    agent = _load_agent(env, args.checkpoint) if mode == "agent" else None

    results = []
    for game in range(1, args.games + 1):
        obs, info = env.reset()
        done = False
        while not done:
            if not args.quiet:
                print(f"\ngame {game}  score {info['score']}  steps {info['steps']}")
                print(env.render())
                if mode != "human":
                    time.sleep(args.delay)
            action = _pick_action(mode, env, obs, info, agent)
            if action is None:
                return
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        results.append((info["score"], info["max_tile"], info["steps"]))
        print(
            f"game {game:3d}: score={info['score']:7d}  "
            f"max_tile={info['max_tile']:5d}  steps={info['steps']}"
        )

    arr = np.array(results, dtype=float)
    print("\n--- summary over", len(results), "games ---")
    print(f"score     mean {arr[:,0].mean():9.1f}  max {arr[:,0].max():9.0f}")
    print(f"max_tile  mean {arr[:,1].mean():9.1f}  max {arr[:,1].max():9.0f}")
    print(f"steps     mean {arr[:,2].mean():9.1f}")


if __name__ == "__main__":
    main()
