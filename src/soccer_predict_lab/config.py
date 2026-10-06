from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}

DEFAULT_SEASONS = ["2324", "2425", "2526"]

FEATURE_COLUMNS = [
    "home_elo","away_elo","elo_diff",
    "home_points_5","away_points_5",
    "home_points_10","away_points_10",
    "home_goals_for_5","away_goals_for_5",
    "home_goals_against_5","away_goals_against_5",
    "home_goal_diff_5","away_goal_diff_5",
    "home_win_rate_5","away_win_rate_5",
    "home_clean_sheet_rate_5","away_clean_sheet_rate_5",
    "home_goals_for_10","away_goals_for_10",
    "home_goals_against_10","away_goals_against_10",
    "home_rest_days","away_rest_days",
    "home_shots_5","away_shots_5",
    "home_sot_5","away_sot_5",
]
