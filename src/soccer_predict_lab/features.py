from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import pandas as pd

from .config import ROLLING_WINDOW


@dataclass
class TeamMatch:
    date: pd.Timestamp
    points: int
    goals_for: int
    goals_against: int
    win: int
    clean_sheet: int
    shots: float | None = None
    shots_on_target: float | None = None


def _mean(values, default=0.0):
    values = [v for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(values)) if values else float(default)


def _team_features(history: Deque[TeamMatch], match_date: pd.Timestamp, prefix: str) -> dict:
    if not history:
        return {
            f"{prefix}_points_avg": 0.0,
            f"{prefix}_goals_for_avg": 0.0,
            f"{prefix}_goals_against_avg": 0.0,
            f"{prefix}_goal_diff_avg": 0.0,
            f"{prefix}_win_rate": 0.0,
            f"{prefix}_clean_sheet_rate": 0.0,
            f"{prefix}_rest_days": 7.0,
            f"{prefix}_shots_avg": 0.0,
            f"{prefix}_shots_on_target_avg": 0.0,
        }

    recent = list(history)[-ROLLING_WINDOW:]
    last_date = recent[-1].date
    rest_days = max((match_date - last_date).days, 0)

    gf = _mean([m.goals_for for m in recent])
    ga = _mean([m.goals_against for m in recent])

    return {
        f"{prefix}_points_avg": _mean([m.points for m in recent]),
        f"{prefix}_goals_for_avg": gf,
        f"{prefix}_goals_against_avg": ga,
        f"{prefix}_goal_diff_avg": gf - ga,
        f"{prefix}_win_rate": _mean([m.win for m in recent]),
        f"{prefix}_clean_sheet_rate": _mean([m.clean_sheet for m in recent]),
        f"{prefix}_rest_days": float(rest_days),
        f"{prefix}_shots_avg": _mean([m.shots for m in recent]),
        f"{prefix}_shots_on_target_avg": _mean([m.shots_on_target for m in recent]),
    }


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.sort_values("Date").reset_index(drop=True)
    histories: dict[str, Deque[TeamMatch]] = defaultdict(lambda: deque(maxlen=20))
    rows = []

    for _, row in matches.iterrows():
        date = row["Date"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        features = {
            "Date": date,
            "HomeTeam": home,
            "AwayTeam": away,
        }
        features.update(_team_features(histories[home], date, "home"))
        features.update(_team_features(histories[away], date, "away"))

        home_goals = int(row["FTHG"])
        away_goals = int(row["FTAG"])
        result = row["FTR"]

        features["target_result"] = result
        features["target_over25"] = int(home_goals + away_goals > 2.5)
        features["target_btts"] = int(home_goals > 0 and away_goals > 0)
        features["target_home_goals"] = home_goals
        features["target_away_goals"] = away_goals
        rows.append(features)

        if result == "H":
            hp, ap = 3, 0
        elif result == "A":
            hp, ap = 0, 3
        else:
            hp, ap = 1, 1

        hs = row["HS"] if "HS" in row.index else None
        ass = row["AS"] if "AS" in row.index else None
        hst = row["HST"] if "HST" in row.index else None
        ast = row["AST"] if "AST" in row.index else None

        histories[home].append(
            TeamMatch(
                date=date,
                points=hp,
                goals_for=home_goals,
                goals_against=away_goals,
                win=int(result == "H"),
                clean_sheet=int(away_goals == 0),
                shots=hs,
                shots_on_target=hst,
            )
        )
        histories[away].append(
            TeamMatch(
                date=date,
                points=ap,
                goals_for=away_goals,
                goals_against=home_goals,
                win=int(result == "A"),
                clean_sheet=int(home_goals == 0),
                shots=ass,
                shots_on_target=ast,
            )
        )

    return pd.DataFrame(rows)
