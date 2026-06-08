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
    max_goals=8
):
    home_xg = min(max(float(home_xg or 0), 0.15), 4.5)
    away_xg = min(max(float(away_xg or 0), 0.15), 4.5)

    matrix = np.zeros(
        (max_goals + 1, max_goals + 1)
    )

    for i in range(max_goals + 1):

        for j in range(max_goals + 1):

            matrix[i, j] = (
                poisson(i, home_xg)
                * poisson(j, away_xg)
                * score_matrix_context_weight(i, j, home_xg, away_xg)
            )

    total_probability = matrix.sum()
    if total_probability > 0:
        matrix = matrix / total_probability

    p_home = np.tril(matrix, -1).sum()

    p_draw = np.trace(matrix)

    p_away = np.triu(matrix, 1).sum()

    # OVER / UNDER

    over_15 = 0
    over_25 = 0
    under_25 = 0

    btts_yes = 0

    scores = []
    calibrated_scores = []

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

            calibrated_scores.append(
                (
                    f"{i}-{j}",
                    float(p) * score_exact_weight(i, j, home_xg, away_xg)
                )
            )

    calibrated_total = sum(p for _, p in calibrated_scores)
    if calibrated_total > 0:
        calibrated_scores = [
            (score, p / calibrated_total)
            for score, p in calibrated_scores
        ]

    calibrated_scores = blend_with_score_prior(calibrated_scores, model_weight=0.68)

    top_scores = sorted(
        calibrated_scores,
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


def score_exact_weight(home_goals, away_goals, home_xg, away_xg):
    total_goals = home_goals + away_goals
    margin = abs(home_goals - away_goals)
    xg_total = home_xg + away_xg
    xg_gap = abs(home_xg - away_xg)
    weight = 1.0

    if total_goals >= 5:
        weight *= 0.45
    elif total_goals == 4:
        weight *= 0.72

    if margin >= 3 and xg_gap < 1.35:
        weight *= 0.58

    if home_goals >= 3 and away_goals == 0 and away_xg >= 0.45:
        weight *= 0.62
    if away_goals >= 3 and home_goals == 0 and home_xg >= 0.45:
        weight *= 0.62

    if xg_total < 2.30 and total_goals >= 4:
        weight *= 0.55
    if xg_total > 3.20 and total_goals <= 1:
        weight *= 0.78

    if (home_goals, away_goals) in {(1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 0), (0, 2)}:
        weight *= 1.08

    if (home_goals, away_goals) == (0, 0):
        weight *= 1.08 if xg_total <= 2.15 else 0.74

    return weight


def score_matrix_context_weight(home_goals, away_goals, home_xg, away_xg):
    total_goals = home_goals + away_goals
    margin = abs(home_goals - away_goals)
    xg_total = home_xg + away_xg
    xg_gap = abs(home_xg - away_xg)
    tightness = max(0.0, min(1.0, (0.48 - xg_gap) / 0.48))
    weight = 1.0

    if home_goals == away_goals:
        if home_goals <= 1:
            weight *= 1 + 0.12 * tightness
        elif home_goals == 2:
            weight *= 1 + 0.06 * tightness

    if margin == 1 and total_goals <= 3:
        weight *= 1 + 0.04 * (1 - tightness)

    if xg_total < 2.15 and total_goals >= 3:
        weight *= 0.84
    elif xg_total > 3.15 and total_goals <= 1:
        weight *= 0.88

    if xg_gap >= 0.85 and home_goals == away_goals:
        weight *= 0.86

    return weight


def blend_with_score_prior(scores, model_weight=0.68):
    prior = {
        "1-1": 0.115,
        "1-0": 0.105,
        "2-1": 0.090,
        "0-1": 0.078,
        "2-0": 0.075,
        "0-0": 0.066,
        "1-2": 0.064,
        "0-2": 0.046,
        "2-2": 0.043,
        "3-1": 0.036,
        "1-3": 0.028,
        "3-0": 0.026,
        "0-3": 0.020,
    }
    prior_floor = 0.004
    score_map = dict(scores)
    all_scores = set(score_map) | set(prior)
    blended = []

    for score in all_scores:
        model_probability = score_map.get(score, 0.0)
        prior_probability = prior.get(score, prior_floor)
        blended.append(
            (
                score,
                model_probability * model_weight + prior_probability * (1 - model_weight),
            )
        )

    total = sum(p for _, p in blended)
    if total <= 0:
        return scores

    return [(score, p / total) for score, p in blended]
