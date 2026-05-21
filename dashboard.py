import streamlit as st
PASSWORD = "29052007"
st.sidebar.checkbox("📱 Mode téléphone rapide", key="mobile_mode")
if st.session_state.get("mobile_mode"):
    st.subheader("📱 Mode téléphone rapide")

    mobile_cols = [
        "date", "home_team", "away_team", "market",
        "ai_probability", "bookmaker_odds", "value",
        "confidence", "ia_badge", "suggested_stake",
        "score_exact_1", "score_exact_1_proba"
    ]

    mobile_cols = [c for c in mobile_cols if c in filtered.columns]

    st.dataframe(
        filtered[mobile_cols]
        .sort_values("value", ascending=False)
        .head(30),
        use_container_width=True
    )

    st.stop()

st.set_page_config(
    page_title="IA Paris Sportifs Ultime",
    layout="wide",
    page_icon="⚽"
)

password = st.text_input(
    "Mot de passe",
    type="password"
)

if password != PASSWORD:
    st.warning("Accès privé.")
    st.stop()
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="IA Paris Sportifs Ultime",
    layout="wide",
    page_icon="⚽"
)

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 20% 20%, rgba(120, 80, 255, 0.35), transparent 30%),
        radial-gradient(circle at 80% 30%, rgba(0, 255, 200, 0.18), transparent 25%),
        radial-gradient(circle at 50% 80%, rgba(255, 0, 180, 0.18), transparent 25%),
        linear-gradient(135deg, #050510 0%, #0b1026 45%, #02030a 100%);
    color: white;
    overflow: hidden;
}

.stApp::before {
    content: "";
    position: fixed;
    width: 200%;
    height: 200%;
    top: -50%;
    left: -50%;
    background-image:
        radial-gradient(white 1px, transparent 1px),
        radial-gradient(rgba(255,255,255,0.6) 1px, transparent 1px);
    background-size: 80px 80px, 130px 130px;
    animation: starsMove 80s linear infinite;
    opacity: 0.18;
    z-index: -1;
}

.stApp::after {
    content: "";
    position: fixed;
    width: 120%;
    height: 120%;
    top: -10%;
    left: -10%;
    background:
        radial-gradient(circle, rgba(80, 180, 255, 0.12), transparent 40%);
    animation: galaxyPulse 8s ease-in-out infinite alternate;
    z-index: -1;
}

@keyframes starsMove {
    from {
        transform: translate(0, 0) rotate(0deg);
    }
    to {
        transform: translate(-120px, -120px) rotate(360deg);
    }
}

@keyframes galaxyPulse {
    from {
        transform: scale(1);
        opacity: 0.35;
    }
    to {
        transform: scale(1.08);
        opacity: 0.75;
    }
}

section[data-testid="stSidebar"] {
    background: rgba(5, 8, 25, 0.85);
    backdrop-filter: blur(14px);
    border-right: 1px solid rgba(120, 150, 255, 0.25);
}

h1, h2, h3 {
    color: #ffffff;
    text-shadow:
        0 0 8px rgba(120, 180, 255, 0.8),
        0 0 18px rgba(120, 80, 255, 0.5);
}

[data-testid="stMetric"] {
    background: rgba(12, 18, 45, 0.72);
    border: 1px solid rgba(120, 180, 255, 0.25);
    border-radius: 18px;
    padding: 18px;
    box-shadow:
        0 0 18px rgba(60, 120, 255, 0.15),
        inset 0 0 20px rgba(255,255,255,0.03);
    backdrop-filter: blur(12px);
}

.card, .value-card {
    padding: 18px;
    border-radius: 20px;
    margin-bottom: 16px;
    border: 1px solid rgba(120, 180, 255, 0.25);
    backdrop-filter: blur(14px);
    box-shadow:
        0 0 24px rgba(80, 120, 255, 0.18),
        inset 0 0 20px rgba(255,255,255,0.03);
}

.card {
    background: rgba(10, 15, 40, 0.72);
}

.value-card {
    background:
        linear-gradient(135deg, rgba(0, 255, 170, 0.25), rgba(80, 80, 255, 0.12));
    border: 1px solid rgba(0, 255, 170, 0.45);
    animation: cardGlow 3s ease-in-out infinite alternate;
}

@keyframes cardGlow {
    from {
        box-shadow: 0 0 18px rgba(0, 255, 170, 0.18);
    }
    to {
        box-shadow: 0 0 34px rgba(0, 255, 170, 0.45);
    }
}

.small {
    color: #b8c7ff;
    font-size: 14px;
}

.big {
    font-size: 22px;
    font-weight: 800;
    color: white;
    text-shadow: 0 0 10px rgba(0, 255, 220, 0.75);
}

div[data-testid="stDataFrame"] {
    background: rgba(10, 15, 40, 0.72);
    border-radius: 18px;
    border: 1px solid rgba(120, 180, 255, 0.25);
    box-shadow: 0 0 18px rgba(60, 120, 255, 0.15);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(15, 20, 55, 0.78);
    border-radius: 14px;
    color: #dce6ff;
    border: 1px solid rgba(120, 180, 255, 0.22);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,255,200,0.35), rgba(100,100,255,0.3));
    color: white;
    box-shadow: 0 0 18px rgba(0,255,200,0.25);
}

button {
    border-radius: 14px !important;
}

</style>
""", unsafe_allow_html=True)

pred_path = Path("data/predictions/predictions_today.csv")
track_path = Path("tracking_results.csv")

if not pred_path.exists():
    st.error("Aucune prédiction trouvée.")
    st.stop()

df = pd.read_csv(pred_path)
low_memory=False

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

mobile_mode = st.sidebar.checkbox("📱 Mode téléphone rapide")

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

if mobile_mode:
    st.subheader("📱 Mode téléphone rapide")

    st.info(f"Dernière actualisation : {df['last_update'].iloc[0]}" if "last_update" in df.columns else "Dernière actualisation inconnue")

    st.subheader("🔥 Value Bets")

    mobile_cols = [
        "date", "sport", "home_team", "away_team", "market",
        "ai_probability", "bookmaker_odds", "value",
        "confidence", "ia_badge", "suggested_stake",
        "score_exact_1", "score_exact_1_proba",
        "draw_hunter", "scorer_prediction"
    ]

    mobile_cols = [c for c in mobile_cols if c in filtered.columns]

    st.dataframe(
        filtered[filtered["decision"] == "VALUE BET"][mobile_cols]
        .sort_values("value", ascending=False)
        .head(50),
        use_container_width=True
    )

    st.subheader("🎯 Scores exacts")

    score_cols = [
        "home_team", "away_team",
        "score_exact_1", "score_exact_1_proba",
        "score_exact_2", "score_exact_2_proba",
        "score_exact_3", "score_exact_3_proba"
    ]

    score_cols = [c for c in score_cols if c in filtered.columns]

    st.dataframe(
        filtered[score_cols]
        .drop_duplicates(subset=["home_team", "away_team"])
        .head(50),
        use_container_width=True
    )

    st.stop()

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
    st.subheader("📊 ROI / Tracking Pro")

    if not track_path.exists():
        st.warning("Aucun fichier tracking_results.csv trouvé.")
    else:
        tracking = pd.read_csv(track_path)
        low_memory=False

        if tracking.empty:
            st.warning("Tracking vide.")
        else:
            tracking["profit"] = pd.to_numeric(
                tracking["profit"],
                errors="coerce"
            ).fillna(0)

            tracking["stake"] = pd.to_numeric(
                tracking["stake"],
                errors="coerce"
            ).fillna(0)

            tracking["bookmaker_odds"] = pd.to_numeric(
                tracking["bookmaker_odds"],
                errors="coerce"
            ).fillna(0)

            settled = tracking[
                tracking["result"].isin(["WIN", "LOSS"])
            ].copy()

            total_staked = settled["stake"].sum() if len(settled) else 0
            profit = settled["profit"].sum() if len(settled) else 0
            roi = profit / total_staked if total_staked > 0 else 0
            hit_rate = (settled["result"] == "WIN").mean() if len(settled) else 0

            pending = tracking[tracking["result"] == "PENDING"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Paris trackés", len(tracking))
            c2.metric("Paris terminés", len(settled))
            c3.metric("Paris en attente", len(pending))
            c4.metric("Profit total", round(profit, 2))

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("ROI global", f"{round(roi * 100, 2)}%")
            c6.metric("Hit Rate", f"{round(hit_rate * 100, 2)}%")
            c7.metric("Misé total", round(total_staked, 2))
            c8.metric("Cote moyenne", round(settled["bookmaker_odds"].mean(), 2) if len(settled) else 0)

            if not settled.empty:
                settled["cumulative_profit"] = settled["profit"].cumsum()

                st.subheader("📈 Evolution bankroll / profit cumulé")
                st.line_chart(settled["cumulative_profit"])

                st.subheader("📊 Profit par pari")
                st.bar_chart(settled["profit"])

                st.subheader("🏆 ROI par championnat")

                if "sport" in settled.columns:
                    roi_sport = settled.groupby("sport").agg(
                        paris=("result", "count"),
                        mises=("stake", "sum"),
                        profit=("profit", "sum"),
                        winrate=("result", lambda x: (x == "WIN").mean())
                    ).reset_index()

                    roi_sport["roi_%"] = (
                        roi_sport["profit"] / roi_sport["mises"] * 100
                    ).round(2)

                    roi_sport["winrate_%"] = (
                        roi_sport["winrate"] * 100
                    ).round(2)

                    roi_sport = roi_sport.sort_values(
                        "roi_%",
                        ascending=False
                    )

                    st.dataframe(
                        roi_sport,
                        use_container_width=True
                    )

                st.subheader("🎯 ROI par type de pari")

                if "market" in settled.columns:
                    roi_market = settled.groupby("market").agg(
                        paris=("result", "count"),
                        mises=("stake", "sum"),
                        profit=("profit", "sum"),
                        winrate=("result", lambda x: (x == "WIN").mean())
                    ).reset_index()

                    roi_market["roi_%"] = (
                        roi_market["profit"] / roi_market["mises"] * 100
                    ).round(2)

                    roi_market["winrate_%"] = (
                        roi_market["winrate"] * 100
                    ).round(2)

                    roi_market = roi_market.sort_values(
                        "roi_%",
                        ascending=False
                    )

                    st.dataframe(
                        roi_market,
                        use_container_width=True
                    )

                    best_market = roi_market.iloc[0]
                    worst_market = roi_market.iloc[-1]

                    col_best, col_worst = st.columns(2)

                    col_best.success(
                        f"✅ Meilleur marché : {best_market['market']} | ROI {best_market['roi_%']}%"
                    )

                    col_worst.error(
                        f"⚠️ Pire marché : {worst_market['market']} | ROI {worst_market['roi_%']}%"
                    )

            else:
                st.info("Aucun pari terminé pour calculer le ROI.")

            st.subheader("📋 Historique complet du tracking")

            st.dataframe(
                tracking.sort_values(
                    "date",
                    ascending=False
                ),
                use_container_width=True
            )

with tabs[5]:
    st.subheader("📋 Toutes les prédictions")

st.dataframe(
    filtered.sort_values(
        "value",
        ascending=False
    ).head(100),
    use_container_width=True
)