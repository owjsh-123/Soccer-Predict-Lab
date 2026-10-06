from soccer_predict_lab.scorelines import poisson_probability, likely_scorelines


def test_poisson_probability_positive():
    p = poisson_probability(2, 1.5)
    assert 0 < p < 1


def test_likely_scorelines_sorted():
    rows = likely_scorelines(1.8, 1.2, top_n=5)
    assert len(rows) == 5
    probs = [r["probability"] for r in rows]
    assert probs == sorted(probs, reverse=True)
