# Can You Predict the NFL?
### Measuring Offensive Predictability, Play-Call Surprise, and Deception in the 2023 NFL Season

## Overview

This project asks three connected questions about NFL offenses in the 2023 season:

1. **Can we predict a play call (run vs. pass) using only information available before the snap?**
2. **Which teams are the easiest — and hardest — to read, and in which situations?**
3. **Does being unpredictable actually help an offense, or can good execution beat predictability anyway?**

The goal isn't just a run/pass classifier — the model is a tool for discovering football insights about tendency, deception, and situational play-calling.

## Data

- **Source:** `play_by_play_2023.csv` — 2023 NFL season play-by-play (nflfastR schema), 49,665 rows, 372 columns.
- **Included here:** `play_by_play_2023_trimmed.csv` (22 MB) / `play_by_play_2023_trimmed.csv.gz` (5 MB) — a 74-column subset containing everything this project uses (situation, play call, outcome, EPA/WP, and skill-position player names), so the repo stays well under GitHub's file-size limits without losing anything the analysis needs. The full 372-column file is ~96 MB and isn't included.

### What counts as a play

- **Usable run/pass plays:** `play_type` is `run` or `pass`, `down` is not null, `posteam` is not null, `season_type == 'REG'` (regular season only, so the team pool and stakes are consistent).
- This filter yields **35,474 clean plays**, excluding penalties/no-plays, kicks, punts, kneels, and spikes — none of those are real run/pass decisions.

### Pre-snap features (the only inputs the model is allowed to see)

| Feature | Why it's pre-snap-legal |
|---|---|
| `down`, `ydstogo`, `yardline_100` | Known before the play starts |
| `qtr`, `half_seconds_remaining` | Game clock, known pre-snap |
| `score_differential` | Known pre-snap |
| `shotgun`, `no_huddle`, `goal_to_go` | Formation/situation, set before the snap |
| `posteam` (team identity) | Captures each team's baseline tendency |

Nothing derived from the play's outcome (yards gained, EPA, completion, etc.) is used as a model input — those are only used afterward to evaluate whether predictability and surprise matter.

## Methodology

### No-leakage rule
Every prediction for a given week is made by a model trained **only on prior weeks**. Concretely: to predict week *w*, the model trains on all weeks `< w`. The first 4 weeks are used only as training history and excluded from scored results, leaving **25,819 out-of-sample predictions** across weeks 5–18. This mirrors how a real analyst would have to work in-season — no using the future to predict the past.

### Model
Gradient-boosted trees (`HistGradientBoostingClassifier`) predicting `P(pass)` from the pre-snap features above, with team identity handled as a native categorical feature. Walk-forward accuracy: **72.1%**, with mean model confidence (72.0%) closely matching accuracy — the model is well-calibrated, not just overconfident.

### Core metrics
- **Confidence** = `max(P(pass), P(run))` for a given play
- **Surprise** = `1 − P(actual play)` — how much the model didn't see the actual call coming
- **Predictability Score** (per team) = `50% × accuracy + 50% × mean confidence`, on a 0–100 scale
- **Deception Value** (planned) = `Surprise × Play Success (EPA)` — do teams create value specifically by breaking tendency?

## Results so far

- **League pass rate:** 58.2% overall; climbs from ~22–40% on early-and-short situations to 87%+ on 3rd-and-long.
- **Most predictable offense:** New Orleans Saints (score: 78.2)
- **Least predictable offense:** Green Bay Packers (score: 67.7)
- **Biggest predictability driver:** Shotgun formation — more influential than down, distance, or even team identity (permutation importance).
- **Surprise pays off, on average:** Raw play-level correlation between surprise and EPA is weak and noisy (r ≈ 0.03), but binning plays into surprise deciles reveals a clear, mostly monotonic upward trend — more surprising play calls average better EPA than predictable ones once the noise is averaged out.
- **Predictable ≠ bad:** Some efficient offenses (e.g., New Orleans) are both good and predictable — supporting the idea that execution can outweigh disguise.

## Visualizations produced

| # | File | Shows |
|---|---|---|
| 1 | `nfl_2023_run_pass_heatmap.png` | League-wide pass rate by down & distance |
| 2 | `02_team_offensive_dna.png` | Team predictability vs. pass rate |
| 3 | `03_predictability_rankings.png` | Full 32-team predictability leaderboard |
| 4 | `04_predictability_by_situation.png` | Model confidence by down & distance |
| 5 | `05_surprise_distribution.png` | Distribution of surprise scores, most vs. least predictable team |
| 6 | `06_surprise_vs_epa.png` | Surprise score vs. offensive EPA (binned + trend) |
| 7 | `07_predictability_vs_success.png` | Team predictability vs. offensive EPA, four-quadrant view |
| 8 | `08_feature_importance.png` | What situational factors drive predictability |

## Project structure

```
NFL Predictability Project
│
├── data/
│   ├── play_by_play_2023_trimmed.csv       # trimmed dataset used by this project
│   └── play_by_play_2023_trimmed.csv.gz    # compressed version
│
├── notebooks/ or src/
│   ├── build_model.py        # walk-forward model training + per-play scoring
│   ├── build_visuals.py      # generates visuals #2–8
│   └── team_stats.csv        # per-team predictability/EPA/pass-rate summary
│
└── outputs/
    └── *.png                 # all visualizations
```

## Stack

Python: `pandas`, `numpy`, `scikit-learn` (HistGradientBoostingClassifier, permutation importance), `matplotlib`. Planned next: `XGBoost`/`SHAP` for a deeper model, `Streamlit` for an interactive dashboard.

## Next steps

1. Build the "Top 10 Most Surprising Successful Plays" table (team, situation, expected play, actual play, EPA).
2. Add SHAP explanations for individual play predictions ("why did the model think this was 92% pass?").
3. Build the Streamlit dashboard (team selector, situational heatmap, surprise-play explorer).
4. Formal hypothesis tests (predictability vs. EPA, obvious-passing-downs penalty) with confidence intervals rather than point estimates.
