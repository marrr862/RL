# Completed headline bandit assignment

Student ID: 36127 (TT361, group 2, journal number 7).

## Submission files

- mab_assignment.py: UCB1, Thompson Sampling and enabled extension C2.
- mab_report_36127.pdf: tables, plots and answers for Parts B-D.

All 12 supplied self-checks passed. The original supplied files remain unchanged. The given CTR generator, environments, baseline agents, simulation and plotting code are preserved in the completed copy.

## Reproduce

From this folder:

```sh
MPLBACKEND=Agg .venv/bin/python check_my_work.py
MPLBACKEND=Agg .venv/bin/python mab_assignment.py
```

On another machine, install numpy and matplotlib and use python in place of .venv/bin/python. The experiment runs 20 repetitions of 20,000 users per strategy. It prints the tables and saves part_b.png and part_c2.png.

checks.txt records validation; results.txt records the experiment tables. build_report.py generates the PDF from the results and plots (requires reportlab and pymupdf).
