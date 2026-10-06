from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from soccer_predict_lab.config import MODEL_DIR, PROCESSED_DIR
from soccer_predict_lab.predict import predict_match


st.set_page_config(
    page_title="Soccer Predict Lab",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Soccer Predict Lab")
st.caption("Machine-learning soccer match predictions for portfolio and research use.")

feature_path = PROCESSED_DIR / "features.csv"
metadata_path = MODEL_DIR / "metadata.json"

if not feature_path.exists() or not metadata_path.exists():
    st.warning(
        "No trained model found yet. Run the download → prepare → train commands from the README."
    )
    st.stop()

df = pd.read_csv(feature_path, parse_dates=["Date"])
metadata = json.loads(metadata_path.read_text())
feature_cols = metadata["features"]

teams = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))
latest_by_home = df.sort_values("Date").groupby("HomeTeam").tail(1)
latest_by_away = df.sort_values("Date").groupby("AwayTeam").tail(1)

home_team = st.selectbox("Home team", teams, index=0)
away_choices = [t for t in teams if t != home_team]
away_team = st.selectbox("Away team", away_choices, index=min(1, len(away_choices)-1))

def latest_team_features(team: str, prefix: str) -> dict:
    source = latest_by_home if prefix == "home" else latest_by_away
    row = source[source["HomeTeam" if prefix == "home" else "AwayTeam"] == team]
    if row.empty:
        return {}
    r = row.iloc[-1]
    return {c: float(r[c]) for c in feature_cols if c.startswith(prefix + "_")}

feature_row = {}
feature_row.update(latest_team_features(home_team, "home"))
feature_row.update(latest_team_features(away_team, "away"))

missing = [c for c in feature_cols if c not in feature_row]
for c in missing:
    feature_row[c] = float(df[c].median())

if st.button("Predict match", type="primary"):
    result = predict_match(feature_row)

    st.subheader(f"{home_team} vs {away_team}")

    probs = pd.DataFrame(
        {
            "Outcome": [home_team, "Draw", away_team],
            "Probability": [
                result["home_win"],
                result["draw"],
                result["away_win"],
            ],
        }
    )
    fig = px.bar(probs, x="Outcome", y="Probability", text_auto=".1%")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Over 2.5", f"{result['over25']:.1%}")
    c2.metric("BTTS Yes", f"{result['btts_yes']:.1%}")
    c3.metric(
        "Predicted goals",
        f"{home_team} {result['home_goals']:.2f} — {result['away_goals']:.2f} {away_team}",
    )

    st.subheader("Most likely scorelines")
    score_df = pd.DataFrame(result["scorelines"])
    score_df["probability"] = score_df["probability"].map(lambda x: f"{x:.1%}")
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    with st.expander("Model input features"):
        st.json(feature_row)

st.divider()
st.caption(
    "For education and portfolio demonstration. Predictions are probabilistic, not guarantees."
)
