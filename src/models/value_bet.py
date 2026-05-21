import pandas as pd

def calculate_value_bets(df):

    # Probabilité implicite bookmaker
    df["implied_probability"] = 1 / df["bookmaker_odds"]

    # Value
    df["value"] = (
        df["ai_probability"] * df["bookmaker_odds"]
    ) - 1

    # Confiance
    def confidence(prob):
        if prob >= 0.75:
            return "Elite"
        elif prob >= 0.65:
            return "Forte"
        elif prob >= 0.55:
            return "Moyenne"
        else:
            return "Faible"

    df["confidence"] = df["ai_probability"].apply(confidence)

    # FILTRES IMPORTANTS
    df = df[
        (df["bookmaker_odds"] >= 1.4) &
        (df["bookmaker_odds"] <= 3.5) &
        (df["value"] >= 0.08) &
        (df["ai_probability"] >= 0.45)
    ]

    # Stake sécurisé
    df["suggested_stake"] = 1

    return df