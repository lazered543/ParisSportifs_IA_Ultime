import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="IA Paris Sportifs Ultime",
    layout="wide",
    page_icon="⚽"
)

# =========================
# STYLE CSS PREMIUM
# =========================

st.markdown("""
<style>

body {
    background-color: #0E1117;
}

.main {
    background-color: #0E1117;
}

.stDataFrame {
    border-radius: 12px;
}

.valuebet {
    background-color: #0f5132;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}

.nobet {
    background-color: #5c1a1a;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}

.metric-box {
    background: #161b22;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================

pred_path = Path("data/predictions/predictions_today.csv")

if not pred_path.exists():
    st.error("Aucune prédiction trouvée.")
    st.stop()

df = pd.read_csv(pred_path)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("⚙️ Filtres")

sports = st.sidebar.multiselect(
    "Championnats",
    options=df["sport"].unique(),
    default=df["sport"].unique()
)

confidence = st.sidebar.multiselect(
    "Confiance",
    options=df["confidence"].unique(),
    default=df["confidence"].unique()
)

only_value = st.sidebar.checkbox("Afficher uniquement les VALUE BETS")

# =========================
# FILTERS
# =========================

filtered = df[
    (df["sport"].isin(sports)) &
    (df["confidence"].isin(confidence))
]

if only_value:
    filtered = filtered[filtered["decision"] == "VALUE BET"]

# =========================
# HEADER
# =========================

st.title("⚽ IA Paris Sportifs Ultime")

col1, col2, col3, col4 = st.columns(4)

col1.metric("📊 Matchs analysés", len(filtered))

value_bets = filtered[filtered["decision"] == "VALUE BET"]

col2.metric("🔥 VALUE BETS", len(value_bets))

if len(filtered) > 0:
    best_value = round(filtered["value"].max() * 100, 2)
else:
    best_value = 0

col3.metric("💎 Meilleure value", f"{best_value}%")

if len(filtered) > 0:
    max_stake = round(filtered["suggested_stake"].max(), 2)
else:
    max_stake = 0

col4.metric("💰 Mise max", f"{max_stake}€")

# =========================
# VALUE BETS
# =========================

st.subheader("🔥 Paris recommandés")

if len(value_bets) == 0:
    st.warning("Aucun VALUE BET actuellement.")
else:
    for _, row in value_bets.iterrows():

        st.markdown(f"""
        <div class="valuebet">

        <h4>{row['home_team']} vs {row['away_team']}</h4>

        <b>Pari :</b> {row['market']}<br>

        <b>Probabilité IA :</b> {round(row['ai_probability']*100,1)}%<br>

        <b>Cote :</b> {row['bookmaker_odds']}<br>

        <b>Value :</b> {round(row['value']*100,1)}%<br>

        <b>Confiance :</b> {row['confidence']}<br>

        <b>Score exact probable :</b> {row['top_scores']}

        </div>
        """, unsafe_allow_html=True)

# =========================
# ALL PREDICTIONS
# =========================

st.subheader("📋 Toutes les prédictions")

st.dataframe(
    filtered.sort_values("value", ascending=False),
    use_container_width=True
)