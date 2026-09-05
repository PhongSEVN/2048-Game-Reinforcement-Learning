"""Pure 2048 game logic for an N x N board.

Design choices that matter for RL:
- The board is a numpy int32 array holding *tile values* (0, 2, 4, 8, ...).
- All move logic is built from one primitive: "slide + merge a single row
  to the left". Right / up / down are that primitive plus reflto/transpose.
- `step_move` returns (new_board, gained_score, changed). `changed` tells the
  environment whether the move was legal (a move that changes nothing is
  illegal in 2048 and must not spawn a tile).
"""

from __future__ import annotations

import numpy as np

# Probability a newly spawned tile is a 2 (otherwise it is a 4).
SPAWN_TWO_PROB = 0.9

ACTIONS = ("up", "down", "left", "right")
UP, DOWN, LEFT, RIGHT = range(4)


def new_board(size: int, rng: np.random.Generator) -> np.ndarray:
    """Empty board with two starting tiles."""
    board = np.zeros((size, size), dtype=np.int32)
    board = spawn_tile(board, rng)
    board = spawn_tile(board, rng)
    return board


def spawn_tile(board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a copy with one new tile on a random empty cell.

    If the board is full the same board is returned unchanged.
    """
    empties = np.argwhere(board == 0)
    if len(empties) == 0:
        return board
    r, c = empties[rng.integers(len(empties))]
    value = 2 if rng.random() < SPAWN_TWO_PROB else 4
    out = board.copy()
    out[r, c] = value
    return out


def _slide_merge_row(row: np.ndarray) -> tuple[np.ndarray, int]:
    """Slide one row toward index 0 and merge equal neighbours once.

    Returns the new row (same length) and the score gained (sum of tiles
    created by merges).
    """
    size = len(row)
    tiles = [int(x) for x in row if x != 0]
    merged: list[int] = []
    gained = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            value = tiles[i] * 2
            merged.append(value)
            gained += value
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    merged.extend([0] * (size - len(merged)))
    return np.array(merged, dtype=np.int32), gained


def _move_left(board: np.ndarray) -> tuple[np.ndarray, int]:
    out = np.zeros_like(board)
    gained = 0
    for r in range(board.shape[0]):
        out[r], row_gain = _slide_merge_row(board[r])
        gained += row_gain
    return out, gained


def step_move(board: np.ndarray, action: int) -> tuple[np.ndarray, int, bool]:
    """Apply an action. Returns (new_board, gained_score, changed).

    Does NOT spawn a new tile - the environment does that only when
    `changed` is True.
    """
    if action == LEFT:
        moved, gained = _move_left(board)
    elif action == RIGHT:
        flipped = board[:, ::-1]
        moved, gained = _move_left(flipped)
        moved = moved[:, ::-1]
    elif action == UP:
        moved, gained = _move_left(board.T)
        moved = moved.T
    elif action == DOWN:
        flipped = board.T[:, ::-1]
        moved, gained = _move_left(flipped)
        moved = moved[:, ::-1].T
    else:
        raise ValueError(f"action must be 0..3, got {action!r}")

    changed = not np.array_equal(moved, board)
    return moved, gained, changed


def legal_actions(board: np.ndarray) -> list[int]:
    """Actions that would change the board."""
    result = []
    for action in range(4):
        _, _, changed = step_move(board, action)
        if changed:
            result.append(action)
    return result


def is_game_over(board: np.ndarray) -> bool:
    return len(legal_actions(board)) == 0


def max_tile(board: np.ndarray) -> int:
    return int(board.max())


def render(board: np.ndarray) -> str:
    """Human-readable board for the terminal."""
    width = max(4, len(str(max_tile(board))) + 1)
    lines = []
    bar = "+" + ("-" * width + "+") * board.shape[1]
    for row in board:
        cells = "".join(
            f"{('.' if v == 0 else v):>{width}}" + "|" for v in row
        )
        lines.append(bar)
        lines.append("|" + cells)
    lines.append(bar)
    return "\n".join(lines)
