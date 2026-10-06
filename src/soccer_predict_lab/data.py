from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .config import RAW_DIR


BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


def download_season(league: str, season: str, out_dir: Path = RAW_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = BASE_URL.format(season=season, league=league)
    destination = out_dir / f"{league}_{season}.csv"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def download_many(
    league: str,
    seasons: Iterable[str],
    out_dir: Path = RAW_DIR,
) -> list[Path]:
    paths = []
    for season in seasons:
        paths.append(download_season(league, season, out_dir))
    return paths


def load_raw_matches(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Run the download command first."
        )

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)

    matches = pd.concat(frames, ignore_index=True)
    return normalize_columns(matches)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
    out = out.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    out["FTHG"] = pd.to_numeric(out["FTHG"], errors="coerce")
    out["FTAG"] = pd.to_numeric(out["FTAG"], errors="coerce")
    out = out.dropna(subset=["FTHG", "FTAG"])
    out["FTHG"] = out["FTHG"].astype(int)
    out["FTAG"] = out["FTAG"].astype(int)

    for col in ["HS", "AS", "HST", "AST"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out.sort_values("Date").reset_index(drop=True)
