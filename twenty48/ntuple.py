"""N-tuple network + afterstate TD learning for 2048.

Why afterstates
----------------
A 2048 move is *deterministic*: given the board and a direction, the slide and
merges are fixed. Only the tile that spawns afterward is random. So instead of
learning Q(state, action) we learn V(afterstate) - the value of the board
*right after the slide, before the spawn*. The policy is then a 1-ply search:

    a* = argmax over legal a of [ reward(s, a) + V(afterstate(s, a)) ]

The value function is an **n-tuple network**: pick many small groups of cells
("tuples"); each tuple indexes its own lookup table by the tile exponents on
those cells; V(board) is the sum of the looked-up weights. It is basically a
huge sparse linear model over local board patterns - no neural network, no
torch, learns in minutes, and reliably reaches large tiles.

Update rule (temporal-difference on afterstates, Szubert & Jaskowski 2014):

    delta = r_next + V(afterstate_next) - V(afterstate)
    V(afterstate) += lr * delta          # spread across the tuple tables
"""

from __future__ import annotations

import numpy as np

from . import board as B


def default_templates(size: int) -> list[list[tuple[int, int]]]:
    """Cell groups to use as tuples.

    Small boards (<=5): every row, every column, every 2x2 square - full
    coverage, tables stay small.
    Bigger boards: only length<=4 tuples (triples + squares) so the lookup
    tables do not explode (`max_exp ** len`).
    """
    rows = [[(r, c) for c in range(size)] for r in range(size)]
    cols = [[(r, c) for r in range(size)] for c in range(size)]
    squares = [
        [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
        for r in range(size - 1)
        for c in range(size - 1)
    ]
    if size <= 5:
        return rows + cols + squares

    if size <= 7:
        h_tri = [
            [(r, c), (r, c + 1), (r, c + 2)]
            for r in range(size)
            for c in range(size - 2)
        ]
        v_tri = [
            [(r, c), (r + 1, c), (r + 2, c)]
            for r in range(size - 2)
            for c in range(size)
        ]
        return h_tri + v_tri + squares

    # large boards (8..10): 2x2 squares only. Triples + squares is ~240 tuples
    # on a 10x10, which makes every value() call painfully slow. ~81 squares
    # still cover every cell with overlapping patterns.
    return squares


class NTupleNetwork:
    def __init__(self, size: int, max_exp: int = 15, templates=None):
        self.size = size
        self.max_exp = max_exp
        self.templates = templates if templates is not None else default_templates(size)
        self.tables = [
            np.zeros(max_exp ** len(t), dtype=np.float64) for t in self.templates
        ]
        self._coords = [np.array(t, dtype=np.int64) for t in self.templates]
        self._pows = [
            np.array([max_exp ** i for i in range(len(t))], dtype=np.int64)
            for t in self.templates
        ]

    # -- core -------------------------------------------------------
    def _exponents(self, board: np.ndarray) -> np.ndarray:
        e = np.zeros((self.size, self.size), dtype=np.int64)
        nz = board > 0
        e[nz] = np.log2(board[nz]).astype(np.int64)
        np.clip(e, 0, self.max_exp - 1, out=e)
        return e

    def _indices(self, e: np.ndarray) -> list[int]:
        out = []
        for coords, pw in zip(self._coords, self._pows):
            vals = e[coords[:, 0], coords[:, 1]]
            out.append(int(vals @ pw))
        return out

    def value(self, board: np.ndarray) -> float:
        e = self._exponents(board)
        total = 0.0
        for tbl, idx in zip(self.tables, self._indices(e)):
            total += tbl[idx]
        return total

    def update(self, board: np.ndarray, delta: float, lr: float) -> None:
        e = self._exponents(board)
        step = lr * delta / len(self.tables)
        for tbl, idx in zip(self.tables, self._indices(e)):
            tbl[idx] += step

    # -- io --------------------------------------------------------
    def save(self, path: str) -> None:
        data = {
            "size": self.size,
            "max_exp": self.max_exp,
            "n_templates": len(self.templates),
        }
        for i, (tpl, tbl) in enumerate(zip(self.templates, self.tables)):
            data[f"tpl_{i}"] = np.array(tpl, dtype=np.int64)
            data[f"tbl_{i}"] = tbl
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str) -> "NTupleNetwork":
        z = np.load(path, allow_pickle=False)
        size = int(z["size"])
        max_exp = int(z["max_exp"])
        n = int(z["n_templates"])
        templates = [[tuple(c) for c in z[f"tpl_{i}"].tolist()] for i in range(n)]
        net = cls(size, max_exp, templates)
        for i in range(n):
            net.tables[i] = z[f"tbl_{i}"].astype(np.float64)
        return net


class AfterstateAgent:
    """Greedy 1-ply policy over `reward + V(afterstate)`, with optional epsilon."""

    def __init__(self, net: NTupleNetwork, eps: float = 0.0, seed: int | None = None):
        self.net = net
        self.eps = eps
        self.rng = np.random.default_rng(seed)

    def best(self, board: np.ndarray, legal: list[int]):
        """Return (action, afterstate, gained). action is None if no legal move."""
        if not legal:
            return None, None, 0.0
        if self.eps and self.rng.random() < self.eps:
            a = int(self.rng.choice(legal))
            after, gained, _ = B.step_move(board, a)
            return a, after, float(gained)

        best = (None, None, 0.0)
        best_val = -1e18
        for a in legal:
            after, gained, changed = B.step_move(board, a)
            if not changed:
                continue
            val = gained + self.net.value(after)
            if val > best_val:
                best_val = val
                best = (a, after, float(gained))
        return best

    # `obs`/`greedy` kept in the signature so play.py / gui.py call this the
    # same way regardless of agent type; only `board` is actually used.
    def act(self, obs, legal, greedy: bool = True, board: np.ndarray | None = None) -> int:
        if board is None:
            raise ValueError("AfterstateAgent.act needs board=<current board>")
        a, _, _ = self.best(board, legal)
        if a is not None:
            return a
        return legal[0] if legal else 0


def train_episode(env, agent: AfterstateAgent, lr: float, empty_bonus: float = 0.0):
    """Play one game, doing online TD(0) afterstate updates.

    Returns (score, max_tile, steps).
    """
    _, info = env.reset()
    action, afterstate, gained = agent.best(env.board, info["legal_actions"])

    while action is not None:
        _, _, terminated, truncated, info = env.step(action)  # spawns a tile
        if truncated and not terminated:
            break  # episode cut by a step cap - no terminal update
        if terminated:
            # terminal: value of this afterstate should predict 0 future reward
            agent.net.update(afterstate, -agent.net.value(afterstate), lr)
            break

        next_action, next_after, next_gained = agent.best(
            env.board, info["legal_actions"]
        )
        if next_action is None:
            agent.net.update(afterstate, -agent.net.value(afterstate), lr)
            break

        shaped = next_gained + empty_bonus * int((env.board == 0).sum())
        delta = shaped + agent.net.value(next_after) - agent.net.value(afterstate)
        agent.net.update(afterstate, delta, lr)

        action, afterstate, gained = next_action, next_after, next_gained

    return info["score"], info["max_tile"], info["steps"]
