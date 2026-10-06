from collections import defaultdict, deque
from dataclasses import dataclass
import numpy as np
import pandas as pd

ELO_START = 1500.0
ELO_K = 24.0
HOME_ADVANTAGE = 65.0

@dataclass
class TeamMatch:
    date: pd.Timestamp
    points: int
    goals_for: int
    goals_against: int
    win: int
    clean_sheet: int
    shots: float | None = None
    sot: float | None = None

def _mean(values):
    vals = [float(v) for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(vals)) if vals else 0.0

def _stats(history, date, prefix):
    r5, r10 = list(history)[-5:], list(history)[-10:]
    def block(rows, n):
        gf = _mean([m.goals_for for m in rows])
        ga = _mean([m.goals_against for m in rows])
        return {
            f"{prefix}_points_{n}": _mean([m.points for m in rows]),
            f"{prefix}_goals_for_{n}": gf,
            f"{prefix}_goals_against_{n}": ga,
            f"{prefix}_goal_diff_{n}": gf-ga,
            f"{prefix}_win_rate_{n}": _mean([m.win for m in rows]),
            f"{prefix}_clean_sheet_rate_{n}": _mean([m.clean_sheet for m in rows]),
        }
    out = {}
    out.update(block(r5, 5))
    out.update(block(r10, 10))
    out[f"{prefix}_rest_days"] = float(max((date-history[-1].date).days,0)) if history else 7.0
    out[f"{prefix}_shots_5"] = _mean([m.shots for m in r5])
    out[f"{prefix}_sot_5"] = _mean([m.sot for m in r5])
    return out

def expected_home_score(home_elo, away_elo):
    return 1/(1+10**((away_elo-(home_elo+HOME_ADVANTAGE))/400))

def update_elo(home_elo, away_elo, result):
    expected = expected_home_score(home_elo, away_elo)
    actual = 1.0 if result=="H" else 0.5 if result=="D" else 0.0
    delta = ELO_K*(actual-expected)
    return home_elo+delta, away_elo-delta

def _points(result):
    if result=="H": return 3,0
    if result=="A": return 0,3
    return 1,1

def build_features(matches):
    matches = matches.sort_values("Date").reset_index(drop=True)
    history = defaultdict(lambda: deque(maxlen=30))
    elo = defaultdict(lambda: ELO_START)
    rows = []

    for _, row in matches.iterrows():
        date, home, away = row["Date"], row["HomeTeam"], row["AwayTeam"]
        h_elo, a_elo = elo[home], elo[away]
        f = {
            "Date": date, "League": row.get("League","unknown"),
            "Season": row.get("Season","unknown"),
            "HomeTeam": home, "AwayTeam": away,
            "home_elo": h_elo, "away_elo": a_elo,
            "elo_diff": h_elo + HOME_ADVANTAGE - a_elo,
        }
        f.update(_stats(history[home], date, "home"))
        f.update(_stats(history[away], date, "away"))

        hg, ag, result = int(row["FTHG"]), int(row["FTAG"]), row["FTR"]
        f.update({
            "target_result": result,
            "target_over25": int(hg+ag >= 3),
            "target_btts": int(hg>0 and ag>0),
            "target_home_goals": hg,
            "target_away_goals": ag,
        })
        rows.append(f)

        hp, ap = _points(result)
        hs = row["HS"] if "HS" in row.index else None
        ass = row["AS"] if "AS" in row.index else None
        hst = row["HST"] if "HST" in row.index else None
        ast = row["AST"] if "AST" in row.index else None

        history[home].append(TeamMatch(date,hp,hg,ag,int(result=="H"),int(ag==0),hs,hst))
        history[away].append(TeamMatch(date,ap,ag,hg,int(result=="A"),int(hg==0),ass,ast))
        elo[home], elo[away] = update_elo(h_elo,a_elo,result)

    return pd.DataFrame(rows)

def build_matchup_features(matches, home_team, away_team, match_date=None):
    matches = matches.sort_values("Date").reset_index(drop=True)
    history = defaultdict(lambda: deque(maxlen=30))
    elo = defaultdict(lambda: ELO_START)

    for _, row in matches.iterrows():
        date, home, away = row["Date"], row["HomeTeam"], row["AwayTeam"]
        hg, ag, result = int(row["FTHG"]), int(row["FTAG"]), row["FTR"]
        hp, ap = _points(result)
        hs = row["HS"] if "HS" in row.index else None
        ass = row["AS"] if "AS" in row.index else None
        hst = row["HST"] if "HST" in row.index else None
        ast = row["AST"] if "AST" in row.index else None
        h_elo, a_elo = elo[home], elo[away]

        history[home].append(TeamMatch(date,hp,hg,ag,int(result=="H"),int(ag==0),hs,hst))
        history[away].append(TeamMatch(date,ap,ag,hg,int(result=="A"),int(hg==0),ass,ast))
        elo[home], elo[away] = update_elo(h_elo,a_elo,result)

    date = pd.Timestamp(match_date) if match_date else matches["Date"].max()+pd.Timedelta(days=7)
    out = {
        "home_elo": elo[home_team], "away_elo": elo[away_team],
        "elo_diff": elo[home_team]+HOME_ADVANTAGE-elo[away_team],
    }
    out.update(_stats(history[home_team], date, "home"))
    out.update(_stats(history[away_team], date, "away"))
    return out
