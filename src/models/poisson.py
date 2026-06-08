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

    calibrated_scores = blend_with_score_prior(
        calibrated_scores,
        model_weight=score_prior_model_weight(home_xg, away_xg),
    )

    top_scores = sorted(
        calibrated_scores,
        key=lambda x: x[1],
        reverse=True
    )[:12]

    score_distribution = [
        (score, probability)
        for score, probability in sorted(calibrated_scores, key=lambda x: x[1], reverse=True)
        if _score_in_display_grid(score)
    ]

    return {

        "p_home": float(p_home),

        "p_draw": float(p_draw),

        "p_away": float(p_away),

        "over_15": float(over_15),

        "over_25": float(over_25),

        "under_25": float(under_25),

        "btts_yes": float(btts_yes),

        "btts_no": float(1 - btts_yes),

        "top_scores": top_scores,

        "score_distribution": score_distribution,
    }


def _score_in_display_grid(score, limit=6):
    try:
        home, away = str(score).split("-", 1)
        return int(home) <= limit and int(away) <= limit
    except Exception:
        return False


def score_exact_weight(home_goals, away_goals, home_xg, away_xg):
    total_goals = home_goals + away_goals
    margin = abs(home_goals - away_goals)
    xg_total = home_xg + away_xg
    xg_gap = abs(home_xg - away_xg)
    weight = 1.0

    if total_goals >= 5:
        weight *= 0.68 if xg_total >= 3.05 else 0.45
    elif total_goals == 4:
        weight *= 0.92 if xg_total >= 2.90 else 0.72

    if margin >= 3 and xg_gap < 1.35:
        weight *= 0.58

    if home_goals >= 3 and away_goals == 0 and away_xg >= 0.45:
        weight *= 0.62
    if away_goals >= 3 and home_goals == 0 and home_xg >= 0.45:
        weight *= 0.62

    if xg_total < 2.30 and total_goals >= 4:
        weight *= 0.55
    if xg_total > 3.20 and total_goals <= 1:
        weight *= 0.48

    if home_xg >= 2.25 and away_xg <= 0.85:
        if (home_goals, away_goals) in {(2, 0), (3, 0), (3, 1), (4, 0)}:
            weight *= 1.22
        if (home_goals, away_goals) in {(4, 1), (5, 0), (5, 1), (6, 0), (6, 1)}:
            weight *= 1.14
        if (home_goals, away_goals) in {(1, 0), (1, 1), (0, 0)}:
            weight *= 0.78

    if away_xg >= 2.15 and home_xg <= 0.85:
        if (home_goals, away_goals) in {(0, 2), (0, 3), (1, 3), (0, 4)}:
            weight *= 1.22
        if (home_goals, away_goals) in {(1, 4), (0, 5), (1, 5), (0, 6), (1, 6)}:
            weight *= 1.14
        if (home_goals, away_goals) in {(0, 1), (1, 1), (0, 0)}:
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
        weight *= 0.72
    elif xg_total > 3.30 and total_goals >= 4:
        weight *= 1.10

    if xg_gap >= 0.85 and home_goals == away_goals:
        weight *= 0.86

    if xg_gap >= 1.15 and margin >= 2 and total_goals >= 2:
        if home_xg > away_xg and home_goals > away_goals:
            weight *= 1.16
        if away_xg > home_xg and away_goals > home_goals:
            weight *= 1.16

    if xg_gap >= 1.45 and margin >= 3 and total_goals >= 3:
        if home_xg > away_xg and home_goals > away_goals:
            weight *= 1.12
        if away_xg > home_xg and away_goals > home_goals:
            weight *= 1.12

    return weight


def score_prior_model_weight(home_xg, away_xg):
    xg_total = home_xg + away_xg
    xg_gap = abs(home_xg - away_xg)
    if xg_gap >= 1.65 or xg_total >= 3.45:
        return 0.93
    if xg_gap >= 1.35 or xg_total >= 3.15:
        return 0.88
    if xg_gap >= 0.85 or xg_total >= 2.85:
        return 0.78
    if xg_total <= 2.05:
        return 0.72
    return 0.68


def blend_with_score_prior(scores, model_weight=0.68):
    prior = build_score_prior()
    prior_floor = 0.0015
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


def build_score_prior(max_goals=6):
    # Broad football prior: every exact score from 0-0 to 6-6 exists,
    # but common scores still start with more baseline mass.
    prior = {}
    for home in range(max_goals + 1):
        for away in range(max_goals + 1):
            total = home + away
            margin = abs(home - away)
            base = 0.018
            base *= 0.72 ** max(total - 2, 0)
            base *= 0.78 ** max(margin - 1, 0)
            if total <= 2:
                base *= 1.25
            if home == away:
                base *= 1.16
            prior[f"{home}-{away}"] = base

    overrides = {
        "1-1": 0.115,
        "1-0": 0.105,
        "2-1": 0.090,
        "2-0": 0.082,
        "0-1": 0.078,
        "0-0": 0.060,
        "1-2": 0.064,
        "0-2": 0.050,
        "3-1": 0.044,
        "2-2": 0.043,
        "3-0": 0.041,
        "1-3": 0.030,
        "0-3": 0.024,
        "4-1": 0.022,
        "4-0": 0.020,
        "3-2": 0.026,
        "2-3": 0.020,
        "5-0": 0.008,
        "5-1": 0.009,
        "0-5": 0.004,
        "1-5": 0.005,
        "6-0": 0.003,
        "6-1": 0.004,
        "0-6": 0.002,
        "1-6": 0.003,
    }
    prior.update(overrides)

    total = sum(prior.values())
    if total <= 0:
        return prior
    return {score: value / total for score, value in prior.items()}
