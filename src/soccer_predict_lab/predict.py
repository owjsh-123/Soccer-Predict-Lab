import json
import pandas as pd
from .config import MODEL_DIR
from .modeling import load_models
from .scorelines import likely_scorelines

def predict_match(feature_row, model_dir=MODEL_DIR):
    meta = json.loads((model_dir/"metadata.json").read_text())
    X = pd.DataFrame([feature_row])[meta["features"]]
    models = load_models(model_dir)

    rp = models["result"].predict_proba(X)[0]
    rc = list(models["result"].classes_)
    m = {str(c):float(p) for c,p in zip(rc,rp)}
    over = float(models["over25"].predict_proba(X)[0,1])
    btts = float(models["btts"].predict_proba(X)[0,1])
    hg = max(.05,float(models["home_goals"].predict(X)[0]))
    ag = max(.05,float(models["away_goals"].predict(X)[0]))

    return {
        "home_win":m.get("H",0),"draw":m.get("D",0),"away_win":m.get("A",0),
        "over25":over,"under25":1-over,
        "btts_yes":btts,"btts_no":1-btts,
        "home_goals":hg,"away_goals":ag,
        "scorelines":likely_scorelines(hg,ag)
    }
