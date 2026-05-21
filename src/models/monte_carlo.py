import numpy as np

def simulate_football(home_xg, away_xg, n=10000):
    hg = np.random.poisson(home_xg, n)
    ag = np.random.poisson(away_xg, n)
    return {
        "home": float((hg > ag).mean()),
        "draw": float((hg == ag).mean()),
        "away": float((hg < ag).mean()),
        "over25": float(((hg + ag) > 2.5).mean()),
        "btts": float(((hg > 0) & (ag > 0)).mean()),
    }
