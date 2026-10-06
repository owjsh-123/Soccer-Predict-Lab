import pandas as pd

from soccer_predict_lab.data import normalize_columns


def test_normalize_columns():
    df = pd.DataFrame(
        {
            "Date": ["01/09/2026"],
            "HomeTeam": ["A"],
            "AwayTeam": ["B"],
            "FTHG": [2],
            "FTAG": [1],
            "FTR": ["H"],
        }
    )
    out = normalize_columns(df)
    assert len(out) == 1
    assert out.loc[0, "FTHG"] == 2
