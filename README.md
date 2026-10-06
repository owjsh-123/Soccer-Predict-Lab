# ⚽ Soccer Predict Lab

> End-to-end soccer analytics and machine-learning project for predicting match outcomes, goal markets and scorelines.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![ML](https://img.shields.io/badge/ML-CatBoost-orange)
![App](https://img.shields.io/badge/App-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)

## What it predicts

- Home / Draw / Away
- Over / Under 2.5 goals
- Both Teams To Score
- Home and away goal estimates
- Most likely scorelines

## Supported leagues

| Code | League |
|---|---|
| E0 | Premier League |
| SP1 | La Liga |
| I1 | Serie A |
| D1 | Bundesliga |
| F1 | Ligue 1 |

## Feature engineering

The model uses only information available **before** the target match:

- Elo ratings + home advantage
- 5-game and 10-game form
- points per game
- goals scored / conceded
- goal-difference trends
- win rate
- clean-sheet rate
- rest days
- shots
- shots on target

## Architecture

```text
Historical match data
        ↓
Cleaning
        ↓
Chronological feature engineering
        ↓
Elo + rolling form
        ↓
CatBoost models
        ↓
Holdout evaluation
        ↓
Streamlit dashboard
```

## Models

| Task | Model |
|---|---|
| 1X2 | CatBoost multiclass classifier |
| Over 2.5 | CatBoost classifier |
| BTTS | CatBoost classifier |
| Home goals | CatBoost regressor |
| Away goals | CatBoost regressor |
| Scorelines | Poisson model |

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/Soccer-Predict-Lab.git
cd Soccer-Predict-Lab
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -e .
```

Build the five-league dataset and train:

```bash
python -m soccer_predict_lab.cli build-all
```

Run the dashboard:

```bash
streamlit run app/app.py
```

Run tests:

```bash
pytest -q
```

## Resume bullet

> Built an end-to-end soccer analytics platform using Python, CatBoost, Elo ratings and leakage-safe rolling time-series features to predict 1X2 outcomes, Over/Under 2.5, BTTS, goals and scoreline probabilities across five major European leagues.

## Data

Historical CSVs are downloaded at runtime from football-data.co.uk. Full third-party datasets are not redistributed by this repository.

## Disclaimer

For education, research and portfolio use. Predictions are probabilistic, not guarantees.

## License

MIT
