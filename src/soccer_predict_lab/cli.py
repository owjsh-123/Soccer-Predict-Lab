from __future__ import annotations

import argparse
import json

import pandas as pd

from .config import PROCESSED_DIR, RAW_DIR
from .data import download_many, load_raw_matches
from .features import build_features
from .modeling import train_models, load_models, chronological_split, evaluate_models
from .config import MODEL_DIR


def cmd_download(args):
    paths = download_many(args.league, args.seasons)
    for p in paths:
        print(f"Downloaded: {p}")


def cmd_prepare(args):
    matches = load_raw_matches()
    features = build_features(matches)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "features.csv"
    features.to_csv(out, index=False)
    print(f"Saved {len(features)} feature rows to {out}")


def cmd_train(args):
    path = PROCESSED_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError("Run `prepare` before `train`.")
    df = pd.read_csv(path, parse_dates=["Date"])
    metrics = train_models(df)
    print(json.dumps(metrics, indent=2))


def cmd_evaluate(args):
    path = PROCESSED_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError("Run `prepare` before `evaluate`.")
    df = pd.read_csv(path, parse_dates=["Date"])
    _, test_df = chronological_split(df)
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    models = load_models()
    metrics = evaluate_models(models, test_df, metadata["features"])
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Soccer Predict Lab")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Download historical match CSVs")
    p.add_argument("--league", default="E0")
    p.add_argument("--seasons", nargs="+", default=["2324", "2425", "2526"])
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("prepare", help="Create leakage-safe rolling features")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="Train all prediction models")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="Evaluate trained models")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
