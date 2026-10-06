from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

DEFAULT_LEAGUE = "E0"
DEFAULT_SEASONS = ["2324", "2425", "2526"]

ROLLING_WINDOW = 5

FEATURE_COLUMNS = [
    "home_points_avg",
    "away_points_avg",
    "home_goals_for_avg",
    "away_goals_for_avg",
    "home_goals_against_avg",
    "away_goals_against_avg",
    "home_goal_diff_avg",
    "away_goal_diff_avg",
    "home_win_rate",
    "away_win_rate",
    "home_clean_sheet_rate",
    "away_clean_sheet_rate",
    "home_rest_days",
    "away_rest_days",
]

OPTIONAL_FEATURE_COLUMNS = [
    "home_shots_avg",
    "away_shots_avg",
    "home_shots_on_target_avg",
    "away_shots_on_target_avg",
]
