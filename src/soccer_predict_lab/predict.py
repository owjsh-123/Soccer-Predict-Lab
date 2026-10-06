from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODEL_DIR
from .modeling import load_models
from .scorelines import likely_scorelines


def predict_match(feature_row: dict, model_dir: Path = MODEL_DIR) -> dict:
    models = load_models(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text())
    features = metadata["features"]

    X = pd.DataFrame([feature_row])[features]

    result_probs = models["result"].predict_proba(X)[0]
    result_classes = list(models["result"].classes_)
    result_map = {str(c): float(p) for c, p in zip(result_classes, result_probs)}

    over25 = float(models["over25"].predict_proba(X)[0, 1])
    btts = float(models["btts"].predict_proba(X)[0, 1])

    home_goals = max(0.05, float(models["home_goals"].predict(X)[0]))
    away_goals = max(0.05, float(models["away_goals"].predict(X)[0]))

    return {
        "home_win": result_map.get("H", 0.0),
        "draw": result_map.get("D", 0.0),
        "away_win": result_map.get("A", 0.0),
        "over25": over25,
        "under25": 1.0 - over25,
        "btts_yes": btts,
        "btts_no": 1.0 - btts,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "scorelines": likely_scorelines(home_goals, away_goals),
    }
