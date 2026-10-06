from soccer_predict_lab.scorelines import likely_scorelines

def test_scorelines_sorted():
    rows = likely_scorelines(1.7,1.2,top_n=5)
    assert len(rows)==5
    probs = [r["probability"] for r in rows]
    assert probs == sorted(probs,reverse=True)
