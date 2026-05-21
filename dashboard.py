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
.card {
    background: #161b22;
    padding: 18px;
    border-radius: 16px;
    margin-bottom: 14px;
    border: 1px solid #30363d;
}
.value-card {
    background: linear-gradient(135deg, #0f5132, #10291f);
    padding: 18px;
    border-radius: 16px;
    margin-bottom: 14px;
    border: 1px solid #2ea043;
}
.warning-card {
    background: #2d1b00;
    padding: 18px;
    border-radius: 16px;
    margin-bottom: 14px;
    border: 1px solid #f0ad4e;
}
.small {
    color: #8b949e;
    font-size: 14px;
}
.big {
    font-size: 22px;
    font-weight: 700;
}
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
st.caption("Dashboard IA : Value Bets, Score Exact, Over/Under, BTTS, ROI et tracking.")

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

    score_cols = [
        "date", "sport", "home_team", "away_team",
        "score_exact_1", "score_exact_1_proba",
        "score_exact_2", "score_exact_2_proba",
        "score_exact_3", "score_exact_3_proba"
    ]

    available_cols = [c for c in score_cols if c in filtered.columns]

    score_df = filtered[available_cols].drop_duplicates(
        subset=["home_team", "away_team"]
    )

    st.dataframe(
        score_df.sort_values("score_exact_1_proba", ascending=False),
        use_container_width=True
    )

with tabs[2]:
    st.subheader("⚽ Over / Under")

    goal_cols = [
        "date", "sport", "home_team", "away_team",
        "over_15", "over_25", "under_25"
    ]

    available_cols = [c for c in goal_cols if c in filtered.columns]

    goals_df = filtered[available_cols].drop_duplicates(
        subset=["home_team", "away_team"]
    )

    st.dataframe(
        goals_df.sort_values("over_25", ascending=False),
        use_container_width=True
    )

with tabs[3]:
    st.subheader("✅ BTTS — Les deux équipes marquent")

    btts_cols = [
        "date", "sport", "home_team", "away_team",
        "btts_yes", "btts_no"
    ]

    available_cols = [c for c in btts_cols if c in filtered.columns]

    btts_df = filtered[available_cols].drop_duplicates(
        subset=["home_team", "away_team"]
    )

    st.dataframe(
        btts_df.sort_values("btts_yes", ascending=False),
        use_container_width=True
    )

with tabs[4]:
    st.subheader("📊 ROI / Tracking")

    if not track_path.exists():
        st.warning("Aucun fichier tracking_results.csv trouvé.")
    else:
        tracking = pd.read_csv(track_path)

        if tracking.empty:
            st.warning("Tracking vide.")
        else:
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

            st.dataframe(
                tracking.sort_values("date", ascending=False),
                use_container_width=True
            )

with tabs[5]:
    st.subheader("📋 Toutes les prédictions")

    st.dataframe(
                    tracking["cumulative_profit"] = (
                tracking["profit"]
                .fillna(0)
                .cumsum()
            )

            st.line_chart(
                tracking.set_index(
                    tracking.index
                )["cumulative_profit"]
            )

            st.bar_chart(
                tracking["profit"].fillna(0)
            )
        filtered.sort_values("value", ascending=False),
        use_container_width=True
    )