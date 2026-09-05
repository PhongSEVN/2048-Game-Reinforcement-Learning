2048 RL - scripts
================

Double-click a .bat file to run it. Training / eval windows stay open at the
end (press a key to close). GUI windows close when you close the window.

Any extra arguments you pass are forwarded to the Python command, e.g.
    train_ntuple_3x3.bat --episodes 40000

setup.bat ................ (optional) make a .venv and install deps
                           the other scripts also work with a system Python
                           that already has numpy / matplotlib

baseline_random_3x3.bat .. random-move score to beat (300 games)

train_ntuple_3x3.bat ..... train the agent on 3x3          -> runs\ntuple_3x3\best.npz
train_ntuple_4x4.bat ..... train on 4x4                    -> runs\ntuple_4x4\best.npz
train_ntuple_10x10.bat ... train on 10x10                  -> runs\ntuple_10x10\best.npz

eval_ntuple_3x3.bat ...... 200 greedy games + stats for the 3x3 agent

gui_ntuple_3x3.bat ....... watch the trained 3x3 agent
gui_ntuple_4x4.bat ....... watch the trained 4x4 agent
gui_ntuple_10x10.bat ..... watch the trained 10x10 agent
gui_random_4x4.bat ....... watch a random baseline
play_human_4x4.bat ....... play 2048 yourself (arrow keys)

Order to try: setup (optional) -> baseline -> train_ntuple_3x3 -> gui_ntuple_3x3
