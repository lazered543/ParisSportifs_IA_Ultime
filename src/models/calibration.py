def calibrate_probability(prob, elo_diff=0):

    prob = float(prob)

    # Réduction des extrêmes
    calibrated = 0.5 + (prob - 0.5) * 0.72

    # Bonus ELO
    if elo_diff > 180:
        calibrated += 0.04

    elif elo_diff > 100:
        calibrated += 0.025

    elif elo_diff < -100:
        calibrated -= 0.04

    # Limites réalistes
    calibrated = max(0.08, min(calibrated, 0.72))

    return round(calibrated, 4)