import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


FEATURES = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_xg",
    "away_xg"
]


TARGET = "home_win"


def prepare_dataset(df):

    df = df.copy()

    df["home_win"] = (
        df["FTHG"] > df["FTAG"]
    ).astype(int)

    return df


def train_model(history_df):

    df = prepare_dataset(history_df)

    X = df[FEATURES]

    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        preds
    )

    print(f"Accuracy IA : {accuracy:.4f}")

    joblib.dump(
        model,
        "models/football_ml_model.pkl"
    )

    return model


def load_model():

    return joblib.load(
        "models/football_ml_model.pkl"
    )


def predict_match(
    model,
    home_elo,
    away_elo,
    elo_diff,
    home_xg,
    away_xg
):

    data = pd.DataFrame([{
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": elo_diff,
        "home_xg": home_xg,
        "away_xg": away_xg
    }])

    prob = model.predict_proba(data)[0][1]

    return float(prob)