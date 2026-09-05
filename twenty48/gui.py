"""Tkinter GUI: watch the agent play 2048 (or random, or play yourself).

Pure standard-library GUI - no extra install. Works on Windows/macOS/Linux
wherever Python was built with Tk (the default).

Examples
--------
    # trained agent auto-plays a 3x3 board
    python -m twenty48.gui --size 3 --checkpoint runs/ntuple_3x3/best.npz

    # random baseline, watch it flail
    python -m twenty48.gui --size 3 --random

    # play it yourself with the arrow keys
    python -m twenty48.gui --size 4 --human

    # headless check that the GUI builds and steps (CI / no display)
    python -m twenty48.gui --size 3 --random --selftest

Controls
--------
    Play / Pause : toggle auto-play (agent or random mode)
    Step         : advance one move
    New Game     : reset the board
    speed slider : milliseconds between auto-play moves
    mode radios  : agent / random / human
    arrow keys   : make a move in human mode
"""

from __future__ import annotations

import argparse

from . import board as B
from .env import Env2048

# 2048 palette keyed by tile exponent (log2). Index 0 = empty cell.
TILE_COLORS = [
    ("#cdc1b4", "#776e65"),   # empty
    ("#eee4da", "#776e65"),   # 2
    ("#ede0c8", "#776e65"),   # 4
    ("#f2b179", "#f9f6f2"),   # 8
    ("#f59563", "#f9f6f2"),   # 16
    ("#f67c5f", "#f9f6f2"),   # 32
    ("#f65e3b", "#f9f6f2"),   # 64
    ("#edcf72", "#f9f6f2"),   # 128
    ("#edcc61", "#f9f6f2"),   # 256
    ("#edc850", "#f9f6f2"),   # 512
    ("#edc53f", "#f9f6f2"),   # 1024
    ("#edc22e", "#f9f6f2"),   # 2048
]
HIGH_TILE = ("#3c3a32", "#f9f6f2")  # 4096 and up

BG = "#bbada0"
FRAME_BG = "#faf8ef"
KEYMAP = {"Up": B.UP, "Down": B.DOWN, "Left": B.LEFT, "Right": B.RIGHT}


def _tile_style(value: int) -> tuple[str, str]:
    if value <= 0:
        return TILE_COLORS[0]
    exp = value.bit_length() - 1  # log2 for powers of two
    return TILE_COLORS[exp] if exp < len(TILE_COLORS) else HIGH_TILE


class GameGUI:
    def __init__(self, args: argparse.Namespace):
        import tkinter as tk

        self.tk = tk
        self.args = args
        self.env = Env2048(size=args.size, seed=args.seed)
        self.agent = self._load_agent()
        self.mode = "human" if args.human else "random" if args.random else "agent"
        if self.mode == "agent" and self.agent is None:
            print("no usable checkpoint -> falling back to random mode")
            self.mode = "random"

        self.playing = False
        self.games = 0
        self.score_sum = 0
        self.best_score = 0

        self.root = tk.Tk()
        self.root.title(f"2048 RL  -  {args.size}x{args.size}")
        self.root.configure(bg=FRAME_BG)
        self.root.resizable(False, False)

        self._build_widgets()
        self.obs, self.info = self.env.reset()
        self._redraw()
        self.root.bind("<Key>", self._on_key)

    # -- setup ---------------------------------------------------------
    def _load_agent(self):
        ckpt = self.args.checkpoint
        if not ckpt:
            return None
        try:
            from .ntuple import AfterstateAgent, NTupleNetwork

            return AfterstateAgent(NTupleNetwork.load(ckpt))
        except Exception as exc:  # bad file / shape mismatch
            print(f"could not load agent: {exc}")
            return None

    def _build_widgets(self) -> None:
        tk = self.tk
        size = self.args.size

        # Fit the board to the screen: leave room for the title bar, status
        # line and control rows, and never let a big board overflow.
        screen_h = self.root.winfo_screenheight()
        screen_w = self.root.winfo_screenwidth()
        budget = min(screen_h - 240, screen_w - 60, 640)
        self.gap = 4 if size >= 8 else 6
        self.cell = max(26, min(120, (budget - (size + 1) * self.gap) // size))
        px = size * self.cell + (size + 1) * self.gap

        self.status = tk.Label(
            self.root, text="", font=("Segoe UI", 10), bg=FRAME_BG, fg="#776e65",
            anchor="w", justify="left", wraplength=px,
        )
        self.status.grid(row=0, column=0, sticky="we", padx=12, pady=(10, 4))

        self.canvas = tk.Canvas(
            self.root, width=px, height=px, bg=BG, highlightthickness=0
        )
        self.canvas.grid(row=1, column=0, padx=12)

        bar = tk.Frame(self.root, bg=FRAME_BG)
        bar.grid(row=2, column=0, sticky="we", padx=12, pady=10)

        self.play_btn = tk.Button(bar, text="Play", width=7, command=self._toggle_play)
        self.play_btn.pack(side="left")
        tk.Button(bar, text="Step", width=7, command=self._manual_step).pack(
            side="left", padx=4
        )
        tk.Button(bar, text="New Game", width=9, command=self._new_game).pack(
            side="left", padx=4
        )

        self.speed = tk.Scale(
            bar, from_=500, to=20, orient="horizontal", label="ms / move",
            bg=FRAME_BG, highlightthickness=0, length=150,
        )
        self.speed.set(self.args.delay_ms)
        self.speed.pack(side="left", padx=8)

        self.mode_var = tk.StringVar(value=self.mode)
        modes = tk.Frame(self.root, bg=FRAME_BG)
        modes.grid(row=3, column=0, sticky="we", padx=12, pady=(0, 10))
        for name in ("agent", "random", "human"):
            state = "normal" if (name != "agent" or self.agent) else "disabled"
            tk.Radiobutton(
                modes, text=name, value=name, variable=self.mode_var,
                command=self._on_mode_change, bg=FRAME_BG, state=state,
            ).pack(side="left", padx=6)

    # -- rounded rect helper -----------------------------------------
    def _round_rect(self, x0, y0, x1, y1, r, **kw):
        pts = [
            x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
            x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
        ]
        return self.canvas.create_polygon(pts, smooth=True, **kw)

    # -- rendering -------------------------------------------------
    def _redraw(self) -> None:
        self.canvas.delete("all")
        board = self.env.board
        size = self.args.size
        r = max(3, self.cell // 12)
        base_px = max(8, int(self.cell * 0.36))
        for i in range(size):
            for j in range(size):
                v = int(board[i, j])
                x0 = self.gap + j * (self.cell + self.gap)
                y0 = self.gap + i * (self.cell + self.gap)
                x1, y1 = x0 + self.cell, y0 + self.cell
                fill, fg = _tile_style(v)
                self._round_rect(x0, y0, x1, y1, r, fill=fill, outline=fill)
                if v:
                    # shrink the font so long numbers (65536, ...) stay inside
                    digits = len(str(v))
                    font_px = max(7, int(min(base_px, self.cell * 1.55 / digits)))
                    self.canvas.create_text(
                        (x0 + x1) / 2, (y0 + y1) / 2, text=str(v),
                        fill=fg, font=("Segoe UI", font_px, "bold"),
                    )

        over_flag = B.is_game_over(board)
        if over_flag:
            px = int(self.canvas["width"])
            self.canvas.create_rectangle(
                0, 0, px, px, fill="#000000", stipple="gray50", outline="",
            )
            self.canvas.create_text(
                px / 2, px / 2 - 14, text="GAME OVER",
                fill="#f9f6f2", font=("Segoe UI", max(16, px // 16), "bold"),
            )
            self.canvas.create_text(
                px / 2, px / 2 + 16, text="Play / Step / New Game",
                fill="#f9f6f2", font=("Segoe UI", max(9, px // 34)),
            )

        avg = self.score_sum / self.games if self.games else 0
        over = " - GAME OVER" if over_flag else ""
        self.status.config(
            text=(
                f"mode {self.mode_var.get()}   score {self.info['score']}   "
                f"max {self.info['max_tile']}   moves {self.info['steps']}   "
                f"|  games {self.games}   avg {avg:.0f}   best {self.best_score}{over}"
            )
        )

    # -- game flow -----------------------------------------------
    def _choose_action(self) -> int | None:
        legal = self.info["legal_actions"]
        if not legal:
            return None
        mode = self.mode_var.get()
        if mode == "random":
            import random

            return random.choice(legal)
        if mode == "agent" and self.agent is not None:
            return self.agent.act(self.obs, legal, greedy=True, board=self.env.board)
        return None  # human: driven by key events

    def _advance(self, action: int) -> None:
        self.obs, _, terminated, truncated, self.info = self.env.step(action)
        if terminated or truncated:
            self._end_game()
        self._redraw()

    def _end_game(self) -> None:
        self.games += 1
        self.score_sum += self.info["score"]
        self.best_score = max(self.best_score, self.info["score"])
        if self.args.autorestart and self.mode_var.get() != "human":
            self.root.after(700, self._new_game)
        else:
            self.playing = False
            self.play_btn.config(text="Play")

    def _tick(self) -> None:
        if not self.playing or self.mode_var.get() == "human":
            return
        action = self._choose_action()
        if action is not None:
            self._advance(action)
        self.root.after(int(self.speed.get()), self._tick)

    # -- controls -----------------------------------------------
    def _toggle_play(self) -> None:
        if self.mode_var.get() == "human":
            return
        if B.is_game_over(self.env.board):
            self._new_game()  # dead board -> start fresh on Play
        self.playing = not self.playing
        self.play_btn.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self._tick()

    def _manual_step(self) -> None:
        if self.mode_var.get() == "human":
            return
        if B.is_game_over(self.env.board):
            self._new_game()
            return
        action = self._choose_action()
        if action is not None:
            self._advance(action)

    def _new_game(self) -> None:
        self.obs, self.info = self.env.reset()
        self._redraw()

    def _on_mode_change(self) -> None:
        self.playing = False
        self.play_btn.config(text="Play")
        self._redraw()

    def _on_key(self, event) -> None:
        if self.mode_var.get() != "human" or event.keysym not in KEYMAP:
            return
        if B.is_game_over(self.env.board):
            self._new_game()
            return
        action = KEYMAP[event.keysym]
        if action in self.info["legal_actions"]:
            self._advance(action)

    # -- entry points ---------------------------------------------
    def run(self) -> None:
        if not self.args.human and self.mode != "human":
            self.playing = True
            self.play_btn.config(text="Pause")
            self._tick()
        self.root.mainloop()

    def selftest(self, moves: int = 40) -> None:
        """Build the window, run some steps, no mainloop. For CI / no display."""
        for _ in range(moves):
            self.root.update()
            if B.is_game_over(self.env.board):
                self._new_game()
                continue
            action = self._choose_action()
            if action is None:
                self._new_game()
                continue
            self._advance(action)
        print(
            f"selftest OK  games={self.games}  best={self.best_score}  "
            f"last_score={self.info['score']}"
        )
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tkinter GUI for 2048 RL")
    p.add_argument("--size", type=int, default=3)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--random", action="store_true", help="random-move mode")
    p.add_argument("--human", action="store_true", help="start in human mode")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--delay-ms", type=int, default=140, dest="delay_ms")
    p.add_argument(
        "--autorestart", action="store_true",
        help="immediately start a new game on game over (default: stop and wait)",
    )
    p.add_argument("--selftest", action="store_true", help="headless build+step check")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gui = GameGUI(args)
    if args.selftest:
        gui.selftest()
    else:
        gui.run()


if __name__ == "__main__":
    main()
