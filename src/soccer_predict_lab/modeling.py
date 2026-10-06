from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    classification_report,
)

from .config import FEATURE_COLUMNS, OPTIONAL_FEATURE_COLUMNS, MODEL_DIR


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return FEATURE_COLUMNS + [c for c in OPTIONAL_FEATURE_COLUMNS if c in df.columns]


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.2):
    df = df.sort_values("Date").reset_index(drop=True)
    cut = max(1, int(len(df) * (1 - test_fraction)))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def train_models(df: pd.DataFrame, model_dir: Path = MODEL_DIR) -> dict:
    model_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df = chronological_split(df)
    features = _feature_columns(df)

    X_train = train_df[features]
    X_test = test_df[features]

    models = {
        "result": CatBoostClassifier(
            iterations=350,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            verbose=False,
            random_seed=42,
        ),
        "over25": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            verbose=False,
            random_seed=42,
        ),
        "btts": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            verbose=False,
            random_seed=42,
        ),
        "home_goals": CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="MAE",
            verbose=False,
            random_seed=42,
        ),
        "away_goals": CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="MAE",
            verbose=False,
            random_seed=42,
        ),
    }

    targets = {
        "result": "target_result",
        "over25": "target_over25",
        "btts": "target_btts",
        "home_goals": "target_home_goals",
        "away_goals": "target_away_goals",
    }

    for name, model in models.items():
        model.fit(X_train, train_df[targets[name]])
        joblib.dump(model, model_dir / f"{name}.joblib")

    metadata = {
        "features": features,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_end": str(train_df["Date"].max()),
        "test_start": str(test_df["Date"].min()),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return evaluate_models(models, test_df, features)


def load_models(model_dir: Path = MODEL_DIR) -> dict:
    names = ["result", "over25", "btts", "home_goals", "away_goals"]
    return {name: joblib.load(model_dir / f"{name}.joblib") for name in names}


def evaluate_models(models: dict, test_df: pd.DataFrame, features: list[str]) -> dict:
    X = test_df[features]
    metrics = {}

    result_pred = models["result"].predict(X).reshape(-1)
    result_prob = models["result"].predict_proba(X)
    classes = list(models["result"].classes_)
    metrics["result_accuracy"] = float(
        accuracy_score(test_df["target_result"], result_pred)
    )
    metrics["result_log_loss"] = float(
        log_loss(test_df["target_result"], result_prob, labels=classes)
    )
    metrics["result_report"] = classification_report(
        test_df["target_result"], result_pred, output_dict=True, zero_division=0
    )

    for name, target in [("over25", "target_over25"), ("btts", "target_btts")]:
        pred = models[name].predict(X).astype(int).reshape(-1)
        prob = models[name].predict_proba(X)[:, 1]
        metrics[f"{name}_accuracy"] = float(accuracy_score(test_df[target], pred))
        metrics[f"{name}_brier"] = float(brier_score_loss(test_df[target], prob))

    for name, target in [
        ("home_goals", "target_home_goals"),
        ("away_goals", "target_away_goals"),
    ]:
        pred = models[name].predict(X)
        metrics[f"{name}_mae"] = float(mean_absolute_error(test_df[target], pred))

    return metrics
