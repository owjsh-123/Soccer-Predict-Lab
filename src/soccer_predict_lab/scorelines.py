import math

def poisson_probability(k, lam):
    lam = max(float(lam), 0.01)
    return math.exp(-lam)*lam**k/math.factorial(k)

def likely_scorelines(home_goals, away_goals, max_goals=6, top_n=6):
    rows = []
    for h in range(max_goals+1):
        for a in range(max_goals+1):
            rows.append({
                "score": f"{h}-{a}",
                "probability": poisson_probability(h,home_goals)*poisson_probability(a,away_goals)
            })
    return sorted(rows,key=lambda x:x["probability"],reverse=True)[:top_n]
