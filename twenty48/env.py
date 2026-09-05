"""RL environment wrapper around the 2048 board.

Follows the classic Gym contract without the dependency:

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Observation
-----------
A float32 tensor of shape (channels, size, size). Channel k is a 1.0 where the
tile value is 2**k (channel 0 = empty cell). This one-hot-by-exponent encoding
is the standard input for 2048 value/Q networks - raw tile values span several
orders of magnitude and train badly.

Reward
------
`log2(gained_score + 1)`. Merges that build big tiles pay off, but the log
keeps the target range small and stable. Illegal actions are filtered by the
agent via `legal_actions`, so there is no illegal-move penalty here.
"""

from __future__ import annotations

import numpy as np

from . import board as B

DEFAULT_CHANNELS = 18  # supports tiles up to 2**17 = 131072


class Env2048:
    def __init__(
        self,
        size: int = 4,
        channels: int = DEFAULT_CHANNELS,
        seed: int | None = None,
        max_steps: int | None = None,
    ):
        if not 2 <= size <= 10:
            raise ValueError("size must be in 2..10")
        self.size = size
        self.channels = channels
        self.max_steps = max_steps  # None = play until game over
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((size, size), dtype=np.int32)
        self.score = 0
        self.steps = 0

    # -- core API ---------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board = B.new_board(self.size, self.rng)
        self.score = 0
        self.steps = 0
        return self.observe(), self._info()

    def step(self, action: int):
        moved, gained, changed = B.step_move(self.board, action)
        if not changed:
            # Should not happen if the agent respects legal_actions(); treat
            # as a no-op step so training stays robust.
            return self.observe(), 0.0, B.is_game_over(self.board), False, self._info()

        self.board = B.spawn_tile(moved, self.rng)
        self.score += gained
        self.steps += 1
        reward = float(np.log2(gained + 1.0))
        terminated = B.is_game_over(self.board)
        truncated = self.max_steps is not None and self.steps >= self.max_steps
        return self.observe(), reward, terminated, truncated, self._info()

    # -- helpers --------------------------------------------------------
    def legal_actions(self) -> list[int]:
        return B.legal_actions(self.board)

    def observe(self) -> np.ndarray:
        exponents = np.zeros_like(self.board)
        nonzero = self.board > 0
        exponents[nonzero] = np.log2(self.board[nonzero]).astype(np.int32)
        exponents = np.clip(exponents, 0, self.channels - 1)
        onehot = np.zeros((self.channels, self.size, self.size), dtype=np.float32)
        for k in range(self.channels):
            onehot[k] = exponents == k
        return onehot

    @property
    def obs_dim(self) -> int:
        return self.channels * self.size * self.size

    @property
    def n_actions(self) -> int:
        return 4

    def _info(self) -> dict:
        return {
            "score": self.score,
            "max_tile": B.max_tile(self.board),
            "steps": self.steps,
            "legal_actions": self.legal_actions(),
        }

    def render(self) -> str:
        return B.render(self.board)
