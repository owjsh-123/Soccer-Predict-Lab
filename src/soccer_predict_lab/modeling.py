from pathlib import Path
import json, joblib
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error
from .config import FEATURE_COLUMNS, MODEL_DIR

def chronological_split(df, test_fraction=.2):
    x = df.sort_values("Date").reset_index(drop=True)
    cut = max(1,int(len(x)*(1-test_fraction)))
    return x.iloc[:cut].copy(), x.iloc[cut:].copy()

def train_models(df, model_dir=MODEL_DIR):
    model_dir.mkdir(parents=True, exist_ok=True)
    train,test = chronological_split(df)
    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    Xtr = train[features]

    models = {
        "result": CatBoostClassifier(iterations=450,depth=6,learning_rate=.04,loss_function="MultiClass",verbose=False,random_seed=42),
        "over25": CatBoostClassifier(iterations=350,depth=6,learning_rate=.05,loss_function="Logloss",verbose=False,random_seed=42),
        "btts": CatBoostClassifier(iterations=350,depth=6,learning_rate=.05,loss_function="Logloss",verbose=False,random_seed=42),
        "home_goals": CatBoostRegressor(iterations=350,depth=6,learning_rate=.05,loss_function="MAE",verbose=False,random_seed=42),
        "away_goals": CatBoostRegressor(iterations=350,depth=6,learning_rate=.05,loss_function="MAE",verbose=False,random_seed=42),
    }
    targets = {
        "result":"target_result","over25":"target_over25","btts":"target_btts",
        "home_goals":"target_home_goals","away_goals":"target_away_goals"
    }
    for name, model in models.items():
        model.fit(Xtr, train[targets[name]])
        joblib.dump(model, model_dir/f"{name}.joblib")

    metrics = evaluate_models(models,test,features)
    (model_dir/"metadata.json").write_text(json.dumps({
        "features":features,"train_rows":len(train),"test_rows":len(test)
    },indent=2))
    (model_dir/"metrics.json").write_text(json.dumps(metrics,indent=2))
    return metrics

def load_models(model_dir=MODEL_DIR):
    return {n:joblib.load(model_dir/f"{n}.joblib") for n in ["result","over25","btts","home_goals","away_goals"]}

def evaluate_models(models,test,features):
    X = test[features]
    out = {}
    pred = models["result"].predict(X).reshape(-1)
    probs = models["result"].predict_proba(X)
    classes = list(models["result"].classes_)
    out["result_accuracy"] = float(accuracy_score(test["target_result"],pred))
    out["result_log_loss"] = float(log_loss(test["target_result"],probs,labels=classes))
    for name,target in [("over25","target_over25"),("btts","target_btts")]:
        p = models[name].predict(X).reshape(-1).astype(int)
        pp = models[name].predict_proba(X)[:,1]
        out[f"{name}_accuracy"] = float(accuracy_score(test[target],p))
        out[f"{name}_brier"] = float(brier_score_loss(test[target],pp))
    for name,target in [("home_goals","target_home_goals"),("away_goals","target_away_goals")]:
        out[f"{name}_mae"] = float(mean_absolute_error(test[target],models[name].predict(X)))
    return out
