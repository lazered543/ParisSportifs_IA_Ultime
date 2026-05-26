import pandas as pd
from pathlib import Path

try:
    import joblib
except Exception:
    joblib = None

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier
    ML_DEPS_OK = True
except Exception:
    train_test_split = None
    accuracy_score = None
    XGBClassifier = None
    ML_DEPS_OK = False

MODEL_PATH = "models/xgboost_model.pkl"


def prepare_features(df):

    df = df.copy()

    df["home_win"] = (
        df["FTHG"] > df["FTAG"]
    ).astype(int)

    features = pd.DataFrame()

    features["home_goals"] = df["FTHG"]
    features["away_goals"] = df["FTAG"]

    features["goal_diff"] = (
        df["FTHG"] - df["FTAG"]
    )

    features["total_goals"] = (
        df["FTHG"] + df["FTAG"]
    )

    return features, df["home_win"]


def train_xgboost_model():
    if not ML_DEPS_OK or joblib is None:
        print("Dependances ML manquantes : entrainement XGBoost ignore.")
        return

    hist_path = Path(
        "data/processed/football_history_all.csv"
    )

    if not hist_path.exists():

        print("Historique introuvable.")
        return

    df = pd.read_csv(hist_path)

    df = df.dropna(
        subset=["FTHG", "FTAG"]
    )

    X, y = prepare_features(df)

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
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict(X_test)

    acc = accuracy_score(
        y_test,
        preds
    )

    print(
        f"Accuracy XGBoost : {round(acc,4)}"
    )

    Path("models").mkdir(
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"Modèle sauvegardé : {MODEL_PATH}"
    )


def load_xgboost_model():
    if joblib is None:
        return None

    if Path(MODEL_PATH).exists():

        return joblib.load(MODEL_PATH)

    return None
if __name__ == "__main__":
    train_xgboost_model()
