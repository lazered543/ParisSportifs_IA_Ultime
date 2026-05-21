def implied_probability(odds):
    if odds is None or odds <= 1:
        return None
    return 1 / odds

def value_score(probability, odds):
    if odds is None or odds <= 1:
        return None
    return probability * odds - 1

def classify_confidence(p):
    if p >= 0.80: return "Elite"
    if p >= 0.70: return "Fort"
    if p >= 0.60: return "Moyen"
    if p >= 0.50: return "Faible"
    return "A éviter"
