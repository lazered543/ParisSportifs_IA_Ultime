import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="IA Paris Sportifs Ultime",
    layout="wide",
    page_icon="⚽"
)

st.markdown("""
<style>
body, .main { background-color: #0E1117; }
.card, .value-card {
    padding: 18px;
    border-radius: 16px;
    margin-bottom: 14px;
    border: 1px solid #30363d;
}
.card { background: #161b22; }
.value-card {
    background: linear-gradient(135deg, #0f5132, #10291f);
    border: 1px solid #2ea043;
}
.small { color: #8b949e; font-size: 14px; }
.big { font-size: 22px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

pred_path = Path("data/predictions/predictions_today.csv")
track_path = Path("tracking_results.csv")

if not pred_path.exists():
    st.error("Aucune prédiction trouvée.")
    st.stop()

df = pd.read_csv(pred_path)

st.sidebar.title("⚙️ Filtres")

sports = st.sidebar.multiselect(
    "Championnats",
    options=sorted(df["sport"].dropna().unique()),
    default=sorted(df["sport"].dropna().unique())
)

confidence = st.sidebar.multiselect(
    "Confiance",
    options=sorted(df["confidence"].dropna().unique()),
    default=sorted(df["confidence"].dropna().unique())
)

only_value = st.sidebar.checkbox("Afficher uniquement VALUE BETS")

filtered = df[
    (df["sport"].isin(sports)) &
    (df["confidence"].isin(confidence))
].copy()

if only_value:
    filtered = filtered[filtered["decision"] == "VALUE BET"]

value_bets = filtered[filtered["decision"] == "VALUE BET"].copy()

st.title("⚽ IA Paris Sportifs Ultime")
st.caption("Value Bets • Score Exact • Over/Under • BTTS • ROI • Tracking")

col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Lignes analysées", len(filtered))
col2.metric("🔥 Value Bets", len(value_bets))
col3.metric("💎 Meilleure value", f"{round(filtered['value'].max()*100,2) if len(filtered) else 0}%")
col4.metric("💰 Mise max", f"{round(filtered['suggested_stake'].max(),2) if len(filtered) else 0}€")

tabs = st.tabs([
    "🔥 Value Bets",
    "🎯 Score Exact",
    "⚽ Over/Under",
    "✅ BTTS",
    "📊 ROI / Tracking",
    "📋 Toutes les prédictions"
])

with tabs[0]:
    st.subheader("🔥 Paris recommandés")

    if value_bets.empty:
        st.warning("Aucun VALUE BET actuellement.")
    else:
        for _, row in value_bets.sort_values("value", ascending=False).iterrows():
            st.markdown(f"""
            <div class="value-card">
                <div class="big">{row['home_team']} vs {row['away_team']}</div>
                <div class="small">{row['sport']} — {row['date']}</div><br>
                <b>Pari :</b> {row['market']}<br>
                <b>Probabilité IA :</b> {round(row['ai_probability']*100,1)}%<br>
                <b>Cote :</b> {row['bookmaker_odds']}<br>
                <b>Value :</b> {round(row['value']*100,1)}%<br>
                <b>Confiance :</b> {row['confidence']}<br>
                <b>Stake conseillé :</b> {row['suggested_stake']}€
            </div>
            """, unsafe_allow_html=True)

with tabs[1]:
    st.subheader("🎯 Scores exacts probables")
    cols = [
        "date", "sport", "home_team", "away_team",
        "score_exact_1", "score_exact_1_proba",
        "score_exact_2", "score_exact_2_proba",
        "score_exact_3", "score_exact_3_proba"
    ]
    available = [c for c in cols if c in filtered.columns]
    score_df = filtered[available].drop_duplicates(subset=["home_team", "away_team"])
    st.dataframe(score_df.sort_values("score_exact_1_proba", ascending=False), use_container_width=True)

with tabs[2]:
    st.subheader("⚽ Over / Under")
    cols = ["date", "sport", "home_team", "away_team", "over_15", "over_25", "under_25"]
    available = [c for c in cols if c in filtered.columns]
    goals_df = filtered[available].drop_duplicates(subset=["home_team", "away_team"])
    st.dataframe(goals_df.sort_values("over_25", ascending=False), use_container_width=True)

with tabs[3]:
    st.subheader("✅ BTTS")
    cols = ["date", "sport", "home_team", "away_team", "btts_yes", "btts_no"]
    available = [c for c in cols if c in filtered.columns]
    btts_df = filtered[available].drop_duplicates(subset=["home_team", "away_team"])
    st.dataframe(btts_df.sort_values("btts_yes", ascending=False), use_container_width=True)

with tabs[4]:
    st.subheader("📊 ROI / Tracking")

    if not track_path.exists():
        st.warning("Aucun fichier tracking_results.csv trouvé.")
    else:
        tracking = pd.read_csv(track_path)

        if tracking.empty:
            st.warning("Tracking vide.")
        else:
            tracking["profit"] = pd.to_numeric(tracking["profit"], errors="coerce").fillna(0)
            tracking["stake"] = pd.to_numeric(tracking["stake"], errors="coerce").fillna(0)

            settled = tracking[tracking["result"].isin(["WIN", "LOSS"])]

            total_staked = settled["stake"].sum() if len(settled) else 0
            profit = settled["profit"].sum() if len(settled) else 0
            roi = profit / total_staked if total_staked > 0 else 0
            hit_rate = (settled["result"] == "WIN").mean() if len(settled) else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Paris trackés", len(tracking))
            c2.metric("Paris terminés", len(settled))
            c3.metric("Profit", round(profit, 2))
            c4.metric("ROI", f"{round(roi*100, 2)}%")

            st.metric("Hit Rate", f"{round(hit_rate*100, 2)}%")

            tracking["cumulative_profit"] = tracking["profit"].cumsum()

            st.subheader("📈 Evolution bankroll / profit cumulé")
            st.line_chart(tracking["cumulative_profit"])

            st.subheader("📊 Profit par pari")
            st.bar_chart(tracking["profit"])

            st.dataframe(tracking.sort_values("date", ascending=False), use_container_width=True)

with tabs[5]:
    st.subheader("📋 Toutes les prédictions")
    st.dataframe(filtered.sort_values("value", ascending=False), use_container_width=True)