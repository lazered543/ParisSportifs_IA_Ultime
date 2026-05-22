import streamlit as st
import pandas as pd
from pathlib import Path

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
    AGGRID_OK = True
except Exception:
    AGGRID_OK = False


st.set_page_config(
    page_title="IA Paris Sportifs Ultime",
    layout="wide",
    page_icon="⚽"
)

APP_PASSWORD = "29052007"


st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 20% 20%, rgba(120, 80, 255, 0.35), transparent 30%),
        radial-gradient(circle at 80% 30%, rgba(0, 255, 200, 0.18), transparent 25%),
        radial-gradient(circle at 50% 80%, rgba(255, 0, 180, 0.18), transparent 25%),
        linear-gradient(135deg, #050510 0%, #0b1026 45%, #02030a 100%);
    color: white;
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

@keyframes starsMove {
    from { transform: translate(0, 0) rotate(0deg); }
    to { transform: translate(-120px, -120px) rotate(360deg); }
}

section[data-testid="stSidebar"] {
    background: rgba(5, 8, 25, 0.88);
    backdrop-filter: blur(14px);
    border-right: 1px solid rgba(120, 150, 255, 0.25);
}

h1, h2, h3 {
    color: white;
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

.value-card {
    padding: 18px;
    border-radius: 20px;
    margin-bottom: 16px;
    background: linear-gradient(135deg, rgba(0,255,170,0.25), rgba(80,80,255,0.12));
    border: 1px solid rgba(0,255,170,0.45);
    box-shadow: 0 0 28px rgba(0,255,170,0.25);
    backdrop-filter: blur(14px);
}

.big {
    font-size: 22px;
    font-weight: 800;
    color: white;
}

.small {
    color: #b8c7ff;
    font-size: 14px;
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
</style>
""", unsafe_allow_html=True)


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <div style="text-align:center; padding-top:60px; padding-bottom:30px;">
        <h1 style="font-size:55px; font-weight:900;">🌌 IA PARIS SPORTIFS</h1>
        <p style="font-size:20px; color:#cfd8ff;">
            Plateforme IA privée • Analyse prédictive • Value Betting
        </p>
    </div>
    """, unsafe_allow_html=True)

    password = st.text_input(
        "🔒 Mot de passe",
        type="password",
        key="main_password"
    )

    if password == APP_PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
    elif password:
        st.warning("Mot de passe incorrect.")

    st.stop()


pred_path = Path("data/predictions/predictions_today.csv")
track_path = Path("tracking_results.csv")

if not pred_path.exists():
    st.error("Aucune prédiction trouvée.")
    st.stop()


@st.cache_data(ttl=300)
def load_predictions():
    return pd.read_csv(pred_path, low_memory=False)


@st.cache_data(ttl=300)
def load_tracking():
    if track_path.exists():
        return pd.read_csv(track_path, low_memory=False)
    return pd.DataFrame()


df = load_predictions()


def premium_table(dataframe, height=650, key="table"):
    if dataframe.empty:
        st.warning("Aucune donnée à afficher.")
        return

    data = dataframe.copy()

    preferred_cols = [
        "last_update", "date", "sport", "home_team", "away_team",
        "market", "ai_probability", "bookmaker_odds",
        "implied_probability", "value", "confidence", "ia_badge",
        "decision", "suggested_stake", "score_exact_1",
        "score_exact_1_proba", "score_exact_2", "score_exact_2_proba",
        "score_exact_3", "score_exact_3_proba", "draw_hunter",
        "scorer_prediction", "home_xg", "away_xg",
        "over_25", "under_25", "btts_yes", "btts_no"
    ]

    cols = [c for c in preferred_cols if c in data.columns]
    data = data[cols]

    numeric_cols = [
        "ai_probability", "bookmaker_odds", "implied_probability",
        "value", "suggested_stake", "score_exact_1_proba",
        "score_exact_2_proba", "score_exact_3_proba",
        "home_xg", "away_xg", "over_25", "under_25",
        "btts_yes", "btts_no"
    ]

    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if AGGRID_OK:
        gb = GridOptionsBuilder.from_dataframe(data)

        gb.configure_default_column(
            sortable=True,
            filter=True,
            resizable=True,
            floatingFilter=True,
            editable=False
        )

        gb.configure_grid_options(
            rowHeight=42,
            headerHeight=44,
            animateRows=True,
            enableCellTextSelection=True
        )

        if "value" in data.columns:
            gb.configure_column(
                "value",
                header_name="💎 Value",
                valueFormatter="x == null ? '' : (x * 100).toFixed(1) + '%'",
                cellStyle=JsCode("""
                function(params) {
                    if (params.value >= 0.25) {
                        return {'backgroundColor':'rgba(0,255,170,0.35)','color':'white','fontWeight':'bold'};
                    }
                    if (params.value >= 0.10) {
                        return {'backgroundColor':'rgba(255,215,0,0.25)','color':'white','fontWeight':'bold'};
                    }
                    if (params.value > 0) {
                        return {'backgroundColor':'rgba(255,120,80,0.15)','color':'white'};
                    }
                    return {'color':'#8b949e'};
                }
                """)
            )

        if "ai_probability" in data.columns:
            gb.configure_column(
                "ai_probability",
                header_name="🧠 Proba IA",
                valueFormatter="x == null ? '' : (x * 100).toFixed(1) + '%'",
                cellStyle=JsCode("""
                function(params) {
                    if (params.value >= 0.70) {
                        return {'backgroundColor':'rgba(0,150,255,0.30)','color':'white','fontWeight':'bold'};
                    }
                    if (params.value >= 0.55) {
                        return {'backgroundColor':'rgba(120,80,255,0.22)','color':'white'};
                    }
                    return {'color':'#c9d1d9'};
                }
                """)
            )

        if "decision" in data.columns:
            gb.configure_column(
                "decision",
                header_name="🚦 Décision",
                cellStyle=JsCode("""
                function(params) {
                    if (params.value === 'VALUE BET') {
                        return {'backgroundColor':'rgba(0,255,170,0.26)','color':'#9fffd6','fontWeight':'bold'};
                    }
                    return {'color':'#c9d1d9'};
                }
                """)
            )

        if "ia_badge" in data.columns:
            gb.configure_column(
                "ia_badge",
                header_name="🏷️ Badge IA",
                cellStyle=JsCode("""
                function(params) {
                    if (params.value && params.value.includes('STRONG')) {
                        return {'color':'#7CFFB2','fontWeight':'bold'};
                    }
                    if (params.value && params.value.includes('MEDIUM')) {
                        return {'color':'#FFD76A','fontWeight':'bold'};
                    }
                    if (params.value && params.value.includes('RISKY')) {
                        return {'color':'#FF8A80','fontWeight':'bold'};
                    }
                    return {'color':'#c9d1d9'};
                }
                """)
            )

        if "bookmaker_odds" in data.columns:
            gb.configure_column(
                "bookmaker_odds",
                header_name="💰 Cote",
                valueFormatter="x == null ? '' : Number(x).toFixed(2)"
            )

        if "suggested_stake" in data.columns:
            gb.configure_column(
                "suggested_stake",
                header_name="💸 Stake",
                valueFormatter="x == null ? '' : Number(x).toFixed(2) + '€'"
            )

        grid_options = gb.build()

        custom_css = {
            ".ag-root-wrapper": {
                "background-color": "rgba(5,8,25,0.92) !important",
                "border": "1px solid rgba(120,180,255,0.35) !important",
                "border-radius": "18px !important",
                "box-shadow": "0 0 30px rgba(0,255,255,0.16) !important",
                "overflow": "hidden !important",
            },
            ".ag-header": {
                "background": "linear-gradient(90deg,#0b1026,#111827) !important",
                "color": "white !important",
            },
            ".ag-row": {
                "background-color": "rgba(10,15,40,0.82) !important",
                "color": "#e6edf3 !important",
                "border-bottom": "1px solid rgba(255,255,255,0.06) !important",
            },
            ".ag-row-hover": {
                "background-color": "rgba(0,255,255,0.12) !important",
            },
            ".ag-cell": {
                "border-right": "1px solid rgba(255,255,255,0.05) !important",
            },
        }

        AgGrid(
            data,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.NO_UPDATE,
            allow_unsafe_jscode=True,
            height=height,
            theme="balham",
            custom_css=custom_css,
            fit_columns_on_grid_load=False,
            key=key
        )

    else:
        st.info("Ajoute `streamlit-aggrid` dans requirements.txt pour le tableau premium complet.")

        styled = (
            data.style
            .background_gradient(
                subset=[c for c in ["value"] if c in data.columns],
                cmap="RdYlGn"
            )
            .background_gradient(
                subset=[c for c in ["ai_probability"] if c in data.columns],
                cmap="Blues"
            )
        )

        st.dataframe(styled, use_container_width=True, height=height)


st.sidebar.title("⚙️ Filtres")

if "last_update" in df.columns and not df.empty:
    st.sidebar.info(f"🕒 Dernière MAJ : {df['last_update'].iloc[0]}")

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

markets = st.sidebar.multiselect(
    "Marchés",
    options=sorted(df["market"].dropna().unique()),
    default=sorted(df["market"].dropna().unique())
)

search = st.sidebar.text_input("🔎 Recherche instantanée")

only_value = st.sidebar.checkbox("🔥 Seulement VALUE BETS")
only_reliable = st.sidebar.checkbox("🛡️ Seulement fiables")
mobile_mode = st.sidebar.checkbox("📱 Mode téléphone rapide")

min_value = st.sidebar.slider(
    "Value minimum",
    0.0,
    1.5,
    0.0,
    0.01
)

min_prob = st.sidebar.slider(
    "Probabilité IA minimum",
    0.0,
    1.0,
    0.0,
    0.01
)


filtered = df[
    (df["sport"].isin(sports)) &
    (df["confidence"].isin(confidence)) &
    (df["market"].isin(markets))
].copy()

filtered["value"] = pd.to_numeric(filtered["value"], errors="coerce").fillna(0)
filtered["ai_probability"] = pd.to_numeric(filtered["ai_probability"], errors="coerce").fillna(0)
filtered["suggested_stake"] = pd.to_numeric(filtered["suggested_stake"], errors="coerce").fillna(0)

if only_value:
    filtered = filtered[filtered["decision"] == "VALUE BET"]

if only_reliable and "reliable_only" in filtered.columns:
    filtered = filtered[filtered["reliable_only"] == True]

filtered = filtered[filtered["value"] >= min_value]
filtered = filtered[filtered["ai_probability"] >= min_prob]

if search:
    s = search.lower()
    filtered = filtered[
        filtered.astype(str).apply(
            lambda row: s in " ".join(row.values).lower(),
            axis=1
        )
    ]

value_bets = filtered[filtered["decision"] == "VALUE BET"].copy()


st.title("⚽ IA Paris Sportifs Ultime")
st.caption("Value Bets • Score Exact • Over/Under • BTTS • ROI • Tracking • Tableau style Bloomberg/Betfair")

col1, col2, col3, col4 = st.columns(4)

col1.metric("📊 Lignes analysées", len(filtered))
col2.metric("🔥 Value Bets", len(value_bets))
col3.metric("💎 Meilleure value", f"{round(filtered['value'].max()*100,2) if len(filtered) else 0}%")
col4.metric("💰 Mise max", f"{round(filtered['suggested_stake'].max(),2) if len(filtered) else 0}€")


if mobile_mode:
    st.subheader("📱 Mode téléphone rapide")

    mobile_cols = [
        "date", "sport", "home_team", "away_team", "market",
        "ai_probability", "bookmaker_odds", "value",
        "confidence", "ia_badge", "suggested_stake",
        "score_exact_1", "score_exact_1_proba",
        "draw_hunter", "scorer_prediction"
    ]

    mobile_cols = [c for c in mobile_cols if c in filtered.columns]

    premium_table(
        filtered[mobile_cols].sort_values("value", ascending=False).head(50),
        height=550,
        key="mobile_grid"
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
        for _, row in value_bets.sort_values("value", ascending=False).head(8).iterrows():
            st.markdown(f"""
            <div class="value-card">
                <div class="big">{row['home_team']} vs {row['away_team']}</div>
                <div class="small">{row['sport']} — {row['date']}</div><br>
                <b>Pari :</b> {row['market']}<br>
                <b>Probabilité IA :</b> {round(float(row['ai_probability'])*100,1)}%<br>
                <b>Cote :</b> {row['bookmaker_odds']}<br>
                <b>Value :</b> {round(float(row['value'])*100,1)}%<br>
                <b>Badge :</b> {row.get('ia_badge', '')}<br>
                <b>Confiance :</b> {row['confidence']}<br>
                <b>Stake conseillé :</b> {row['suggested_stake']}€<br>
                <b>Score exact :</b> {row.get('score_exact_1', 'N/A')} — {row.get('score_exact_1_proba', 'N/A')}%<br>
                <b>Anytime goalscorer :</b> {row.get('scorer_prediction', 'N/A')}
            </div>
            """, unsafe_allow_html=True)

        st.subheader("📊 Tableau premium des VALUE BETS")

        premium_table(
            value_bets.sort_values("value", ascending=False).head(100),
            height=620,
            key="value_bets_grid"
        )


with tabs[1]:
    st.subheader("🎯 Scores exacts probables")

    cols = [
        "date", "sport", "home_team", "away_team",
        "score_exact_1", "score_exact_1_proba",
        "score_exact_2", "score_exact_2_proba",
        "score_exact_3", "score_exact_3_proba",
        "draw_probability", "draw_hunter"
    ]

    available = [c for c in cols if c in filtered.columns]

    score_df = (
        filtered[available]
        .drop_duplicates(subset=["home_team", "away_team"])
        .sort_values("score_exact_1_proba", ascending=False)
    )

    premium_table(score_df.head(150), height=650, key="score_grid")


with tabs[2]:
    st.subheader("⚽ Over / Under")

    cols = [
        "date", "sport", "home_team", "away_team",
        "over_25", "under_25", "home_xg", "away_xg"
    ]

    available = [c for c in cols if c in filtered.columns]

    goals_df = (
        filtered[available]
        .drop_duplicates(subset=["home_team", "away_team"])
        .sort_values("over_25", ascending=False)
    )

    premium_table(goals_df.head(150), height=650, key="goals_grid")


with tabs[3]:
    st.subheader("✅ BTTS")

    cols = [
        "date", "sport", "home_team", "away_team",
        "btts_yes", "btts_no", "home_xg", "away_xg"
    ]

    available = [c for c in cols if c in filtered.columns]

    btts_df = (
        filtered[available]
        .drop_duplicates(subset=["home_team", "away_team"])
        .sort_values("btts_yes", ascending=False)
    )

    premium_table(btts_df.head(150), height=650, key="btts_grid")


with tabs[4]:
    st.subheader("📊 ROI / Tracking Pro")

    tracking = load_tracking()

    if tracking.empty:
        st.warning("Aucun fichier tracking_results.csv trouvé ou tracking vide.")
    else:
        if "result" not in tracking.columns:
            tracking["result"] = "PENDING"

        tracking["profit"] = pd.to_numeric(tracking.get("profit", 0), errors="coerce").fillna(0)
        tracking["stake"] = pd.to_numeric(tracking.get("stake", 0), errors="coerce").fillna(0)
        tracking["bookmaker_odds"] = pd.to_numeric(tracking.get("bookmaker_odds", 0), errors="coerce").fillna(0)

        settled = tracking[tracking["result"].isin(["WIN", "LOSS"])].copy()
        pending = tracking[tracking["result"] == "PENDING"]

        total_staked = settled["stake"].sum() if len(settled) else 0
        profit = settled["profit"].sum() if len(settled) else 0
        roi = profit / total_staked if total_staked > 0 else 0
        hit_rate = (settled["result"] == "WIN").mean() if len(settled) else 0

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

        st.subheader("📋 Historique complet du tracking")

        premium_table(
            tracking.sort_values("date", ascending=False).head(300),
            height=650,
            key="tracking_grid"
        )


with tabs[5]:
    st.subheader("📋 Toutes les prédictions")

    premium_table(
        filtered.sort_values("value", ascending=False).head(300),
        height=750,
        key="all_predictions_grid"
    )