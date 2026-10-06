from __future__ import annotations

import math


def poisson_probability(k: int, lam: float) -> float:
    lam = max(float(lam), 0.01)
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def likely_scorelines(home_xg: float, away_xg: float, max_goals: int = 6, top_n: int = 5):
    rows = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson_probability(h, home_xg) * poisson_probability(a, away_xg)
            rows.append({"score": f"{h}-{a}", "probability": p})
    rows.sort(key=lambda x: x["probability"], reverse=True)
    return rows[:top_n]
