from pathlib import Path
import pandas as pd
import requests
from .config import RAW_DIR, LEAGUES

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

def download_season(league, season, out_dir=RAW_DIR):
    if league not in LEAGUES:
        raise ValueError(f"Unsupported league: {league}")
    out_dir.mkdir(parents=True, exist_ok=True)
    url = BASE_URL.format(season=season, league=league)
    path = out_dir / f"{league}_{season}.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path

def download_dataset(leagues, seasons, out_dir=RAW_DIR):
    return [download_season(lg, ss, out_dir) for lg in leagues for ss in seasons]

def normalize_columns(df):
    required = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    x = df.copy()
    x["Date"] = pd.to_datetime(x["Date"], dayfirst=True, errors="coerce")
    x["FTHG"] = pd.to_numeric(x["FTHG"], errors="coerce")
    x["FTAG"] = pd.to_numeric(x["FTAG"], errors="coerce")
    for c in ["HS","AS","HST","AST"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR"])
    x["FTHG"] = x["FTHG"].astype(int)
    x["FTAG"] = x["FTAG"].astype(int)
    return x.sort_values("Date").reset_index(drop=True)

def load_raw_matches(raw_dir=RAW_DIR):
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No raw CSV files found. Run the download command first.")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        bits = path.stem.split("_", 1)
        df["League"] = bits[0]
        df["Season"] = bits[1] if len(bits) > 1 else "unknown"
        frames.append(df)
    return normalize_columns(pd.concat(frames, ignore_index=True))
