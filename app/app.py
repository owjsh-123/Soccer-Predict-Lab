from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from soccer_predict_lab.config import MODEL_DIR, RAW_DIR, LEAGUES
from soccer_predict_lab.data import load_raw_matches
from soccer_predict_lab.features import build_matchup_features
from soccer_predict_lab.predict import predict_match

st.set_page_config(page_title="Soccer Predict Lab",page_icon="⚽",layout="wide")
st.title("⚽ Soccer Predict Lab")
st.caption("Machine-learning match intelligence powered by Elo ratings, rolling form and CatBoost.")

if not (MODEL_DIR/"metadata.json").exists() or not any(RAW_DIR.glob("*.csv")):
    st.info("Train the project first with: python -m soccer_predict_lab.cli build-all")
    st.stop()

matches = load_raw_matches()
teams = sorted(set(matches["HomeTeam"]).union(matches["AwayTeam"]))

tab1,tab2,tab3 = st.tabs(["🎯 Match Prediction","📊 Model Performance","🧠 Project"])

with tab1:
    c1,c2,c3 = st.columns(3)
    with c1:
        st.selectbox("League",["All supported leagues"]+list(LEAGUES.values()))
    with c2:
        home = st.selectbox("Home team",teams)
    with c3:
        away = st.selectbox("Away team",[t for t in teams if t!=home])

    if st.button("Run prediction",type="primary",use_container_width=True):
        row = build_matchup_features(matches,home,away)
        pred = predict_match(row)

        st.subheader(f"{home} vs {away}")
        outcomes = pd.DataFrame({
            "Outcome":[home,"Draw",away],
            "Probability":[pred["home_win"],pred["draw"],pred["away_win"]]
        })
        fig = px.bar(outcomes,x="Outcome",y="Probability",text_auto=".1%")
        fig.update_yaxes(tickformat=".0%",range=[0,1])
        st.plotly_chart(fig,use_container_width=True)

        a,b,c,d = st.columns(4)
        a.metric("Home win",f"{pred['home_win']:.1%}")
        b.metric("Draw",f"{pred['draw']:.1%}")
        c.metric("Away win",f"{pred['away_win']:.1%}")
        d.metric("Over 2.5",f"{pred['over25']:.1%}")

        a,b,c = st.columns(3)
        a.metric("BTTS Yes",f"{pred['btts_yes']:.1%}")
        b.metric(f"{home} goals",f"{pred['home_goals']:.2f}")
        c.metric(f"{away} goals",f"{pred['away_goals']:.2f}")

        st.subheader("Most likely scorelines")
        score = pd.DataFrame(pred["scorelines"])
        score["probability"] = score["probability"].map(lambda x:f"{x:.1%}")
        st.dataframe(score,use_container_width=True,hide_index=True)

with tab2:
    path = MODEL_DIR/"metrics.json"
    if path.exists():
        m = json.loads(path.read_text())
        a,b,c = st.columns(3)
        a.metric("1X2 accuracy",f"{m.get('result_accuracy',0):.1%}")
        b.metric("Over 2.5 accuracy",f"{m.get('over25_accuracy',0):.1%}")
        c.metric("BTTS accuracy",f"{m.get('btts_accuracy',0):.1%}")
        a,b = st.columns(2)
        a.metric("Home goals MAE",f"{m.get('home_goals_mae',0):.3f}")
        b.metric("Away goals MAE",f"{m.get('away_goals_mae',0):.3f}")

with tab3:
    st.markdown("""
### Project features
- Premier League, La Liga, Serie A, Bundesliga and Ligue 1
- Elo team-strength ratings
- 5-match and 10-match rolling form
- Goals, clean sheets, shots and shots-on-target features
- CatBoost classification and regression
- Time-based holdout evaluation
- Poisson scoreline probabilities
- GitHub Actions CI

**Educational sports analytics project. Predictions are probabilistic.**
""")
