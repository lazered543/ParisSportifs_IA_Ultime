def kelly_fraction(probability, odds):
    if odds is None or odds <= 1:
        return 0
    b = odds - 1
    q = 1 - probability
    return max(0, (b * probability - q) / b)

def safe_stake(bankroll, probability, odds, max_pct=0.03):
    k = kelly_fraction(probability, odds)
    stake = bankroll * k * 0.25  # quart Kelly pour limiter le risque
    return round(min(stake, bankroll * max_pct), 2)