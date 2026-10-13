# SARSA and Q-Learning — Mariam Ghazaryan

Week 4 Practical Reinforcement Machine Learning homework.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python check_my_work.py
python sarsa_ql_assignment.py
python cliff_diagnostics.py
```

`check_my_work.py` is the original supplied checker. The provided learning, DP, evaluation, and Part A/C runner functions are preserved.

## Deliverables

- `sarsa_ql_assignment.py`: completed snapshots, heat maps, animation, custom FrozenLake and tuned hyperparameters.
- `cliff_q_video.gif`: 100 frames, SARSA above Q-learning, shared fixed colour scale.
- `cliff_q_final.png`: final CliffWalking Q-value maps.
- `frozenlake_results.png`: final FrozenLake comparison .
- `REPORT.md`: answers D1–D8 with measured results.
- `Mariam_Ghazaryan_SARSA_QL_homework.zip`: submission archive.
- `part_c.log` and `self_check.log`: final run and all 13 passing supplied checks.

Student ID: **7** (16-state 4×4 map). Both tuned agents achieve mean value ratio **1.000** over three seeds.
- `cliff_diagnostics.py` and `cliff_diagnostics.json`: exact episode-level route measurements.

The animated GIF is available [here](cliff_q_video.gif).
