import pandas as pd

from soccer_predict_lab.features import build_features


def test_feature_builder_uses_prior_history_only():
    df = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2026-01-01"),
                "HomeTeam": "A",
                "AwayTeam": "B",
                "FTHG": 2,
                "FTAG": 0,
                "FTR": "H",
            },
            {
                "Date": pd.Timestamp("2026-01-08"),
                "HomeTeam": "A",
                "AwayTeam": "B",
                "FTHG": 1,
                "FTAG": 1,
                "FTR": "D",
            },
        ]
    )
    out = build_features(df)

    assert out.loc[0, "home_points_avg"] == 0
    assert out.loc[1, "home_points_avg"] == 3
    assert out.loc[1, "away_points_avg"] == 0
