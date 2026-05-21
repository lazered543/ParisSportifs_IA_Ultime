import math
import numpy as np


def poisson(k, lam):

    return (
        (lam ** k)
        * math.exp(-lam)
        / math.factorial(k)
    )


def football_poisson_probs(
    home_xg,
    away_xg,
    max_goals=6
):

    matrix = np.zeros(
        (max_goals + 1, max_goals + 1)
    )

    for i in range(max_goals + 1):

        for j in range(max_goals + 1):

            matrix[i, j] = (
                poisson(i, home_xg)
                * poisson(j, away_xg)
            )

    p_home = np.tril(matrix, -1).sum()

    p_draw = np.trace(matrix)

    p_away = np.triu(matrix, 1).sum()

    # OVER / UNDER

    over_15 = 0
    over_25 = 0
    under_25 = 0

    btts_yes = 0

    scores = []

    for i in range(max_goals + 1):

        for j in range(max_goals + 1):

            p = matrix[i, j]

            total = i + j

            if total >= 2:
                over_15 += p

            if total >= 3:
                over_25 += p

            if total <= 2:
                under_25 += p

            if i > 0 and j > 0:
                btts_yes += p

            scores.append(
                (
                    f"{i}-{j}",
                    float(p)
                )
            )

    top_scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )[:5]

    return {

        "p_home": float(p_home),

        "p_draw": float(p_draw),

        "p_away": float(p_away),

        "over_15": float(over_15),

        "over_25": float(over_25),

        "under_25": float(under_25),

        "btts_yes": float(btts_yes),

        "btts_no": float(1 - btts_yes),

        "top_scores": top_scores
    }