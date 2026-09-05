"""Variable-size 2048 game + Reinforcement Learning training code.

Modules
-------
board            : pure game logic (slide / merge / spawn), any board size N.
env              : RL environment wrapper (reset / step / legal-action mask).
ntuple           : n-tuple network + afterstate TD agent (the model used here).
train_afterstate : training loop with CLI (--size 2..10).
play             : watch a trained agent, or play a random baseline.
gui              : Tkinter GUI - agent auto-plays, or you play.
"""
