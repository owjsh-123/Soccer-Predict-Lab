# ⚽ Soccer Predict Lab

A portfolio-ready machine-learning project for predicting soccer match outcomes and common goal markets.

## What it predicts

- **1X2 match result** — Home / Draw / Away
- **Over / Under 2.5 goals**
- **BTTS** — Both Teams To Score
- **Expected goals proxy** — predicted home and away goals
- **Most likely scorelines** — Poisson-based probability estimates

## Why this project is useful

This repository demonstrates:

- data ingestion and cleaning
- rolling time-series feature engineering
- leakage-safe model training
- multi-target machine learning
- model evaluation and persistence
- a Streamlit prediction dashboard
- a command-line workflow
- unit tests
- GitHub Actions CI

## Tech stack

- Python
- pandas / NumPy
- scikit-learn
- CatBoost
- joblib
- Streamlit
- Plotly
- pytest

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/soccer-predict-lab.git
cd soccer-predict-lab
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download historical data

The downloader supports CSV files published by football-data.co.uk.

```bash
python -m soccer_predict_lab.cli download --league E0 --seasons 2324 2425 2526
```

Common league codes:

| League | Code |
|---|---|
| Premier League | E0 |
| La Liga | SP1 |
| Serie A | I1 |
| Bundesliga | D1 |
| Ligue 1 | F1 |

### 5. Build features

```bash
python -m soccer_predict_lab.cli prepare
```

### 6. Train models

```bash
python -m soccer_predict_lab.cli train
```

### 7. Evaluate

```bash
python -m soccer_predict_lab.cli evaluate
```

### 8. Launch dashboard

```bash
streamlit run app/app.py
```

---

## Example dashboard output

```text
Manchester City vs Arsenal

Home Win      48%
Draw          27%
Away Win      25%

Over 2.5      63%
Under 2.5     37%

BTTS Yes      59%

Predicted goals
Manchester City  1.72
Arsenal          1.21

Most likely scores
2-1   14%
1-1   12%
2-0    9%
```

---

## Project structure

```text
soccer-predict-lab/
│
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   └── soccer_predict_lab/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── modeling.py
│       ├── predict.py
│       └── scorelines.py
├── tests/
├── .github/workflows/ci.yml
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Modeling approach

The project creates rolling features using only matches that happened **before** each target match.

Examples:

- recent points per game
- recent goals scored
- recent goals conceded
- home / away form
- win rate
- clean-sheet rate
- goal-difference trend
- rest days
- rolling shot metrics when available

Models:

- CatBoost multiclass classifier → 1X2 result
- CatBoost binary classifier → Over 2.5
- CatBoost binary classifier → BTTS
- CatBoost regressors → home and away goals

The scoreline module converts predicted goal means into Poisson score probabilities.

---

## Data source

This project can download publicly available historical match CSVs from:

- football-data.co.uk

The repository does **not** redistribute their full dataset. Review the source website's terms before using data commercially.

---

## Important note

This project is for **education, analytics, and portfolio demonstration**. Predictions are probabilistic and should not be treated as guaranteed outcomes or financial advice.

---

## Future improvements

- xG-based features
- Elo ratings
- team-strength embeddings
- injuries and lineups
- bookmaker closing odds
- calibration curves
- walk-forward backtesting
- Champions League support
- automated upcoming-fixture ingestion
- FastAPI endpoint
- Docker deployment
- experiment tracking with MLflow

---

## Resume bullet

> Built an end-to-end soccer prediction platform in Python using CatBoost, rolling time-series feature engineering, multi-target classification/regression, Poisson score simulation, and an interactive Streamlit dashboard.

## License

MIT
