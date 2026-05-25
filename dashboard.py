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
    page_icon="PS"
)

APP_PASSWORD = "29052007"


st.markdown("""
<style>
:root {
    --bg: #07100f;
    --panel: #0d1816;
    --panel-soft: #132320;
    --line: rgba(210, 232, 222, 0.14);
    --text: #ecf5f1;
    --muted: #96aaa2;
    --green: #35d07f;
    --blue: #4ea1ff;
    --amber: #f3c34f;
    --red: #ff6b6b;
}

.stApp {
    background:
        linear-gradient(180deg, rgba(53,208,127,0.08) 0%, rgba(7,16,15,0) 260px),
        linear-gradient(135deg, #07100f 0%, #0a1518 46%, #11131a 100%);
    color: var(--text);
    font-family: Inter, "Segoe UI", system-ui, sans-serif;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image: linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
    background-size: 32px 32px;
    mask-image: linear-gradient(to bottom, black, transparent 70%);
    pointer-events: none;
    z-index: -1;
}

.block-container {
    max-width: 1520px;
    padding-top: 26px;
    padding-bottom: 44px;
}

section[data-testid="stSidebar"] {
    background: rgba(9, 19, 18, 0.96);
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] > div {
    padding-top: 22px;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #d8e5df !important;
}

.stTextInput input,
.stMultiSelect [data-baseweb="select"],
.stSlider,
.stCheckbox {
    color: var(--text);
}

.stTextInput input {
    background: #0f1d1b;
    border: 1px solid var(--line);
    border-radius: 8px;
}

h1, h2, h3 {
    color: var(--text);
    letter-spacing: 0;
    text-shadow: none;
}

h1 {
    font-size: 34px !important;
    font-weight: 850 !important;
    margin-bottom: 2px !important;
}

h2, h3 {
    font-weight: 780 !important;
    margin-top: 12px !important;
}

[data-testid="stCaptionContainer"] {
    color: var(--muted);
    font-size: 15px;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(19,35,32,0.98), rgba(12,24,22,0.98));
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 16px 16px 14px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.18);
}

[data-testid="stMetric"] label {
    color: var(--muted) !important;
    font-weight: 650;
}

[data-testid="stMetricValue"] {
    color: var(--text);
    font-weight: 850;
}

.value-card {
    padding: 18px 20px;
    border-radius: 8px;
    margin-bottom: 16px;
    background: linear-gradient(135deg, rgba(53,208,127,0.16), rgba(78,161,255,0.08));
    border: 1px solid rgba(53,208,127,0.34);
    box-shadow: 0 14px 32px rgba(0,0,0,0.22);
}

.big {
    font-size: 20px;
    font-weight: 800;
    color: var(--text);
}

.small {
    color: var(--muted);
    font-size: 14px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 1px solid var(--line);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px 8px 0 0;
    color: #b9cac3;
    border: 1px solid transparent;
    padding: 10px 14px;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    background: rgba(53,208,127,0.12);
    color: #ffffff;
    border-color: var(--line);
    border-bottom-color: rgba(53,208,127,0.42);
}

.stAlert {
    border-radius: 8px;
}

div[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <div style="text-align:center; padding-top:60px; padding-bottom:30px;">
        <h1 style="font-size:55px; font-weight:900;">IA PARIS SPORTIFS</h1>
        <p style="font-size:20px; color:#cfd8ff;">Plateforme IA privee - Analyse predictive - Value Betting</p>
    </div>
    """, unsafe_allow_html=True)

    password = st.text_input(
        "Mot de passe",
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
    st.error("Aucune prediction trouvee.")
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
        st.warning("Aucune donnee a afficher.")
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
            rowHeight=48,
            headerHeight=48,
            animateRows=True,
            enableCellTextSelection=True,
            suppressCellFocus=True,
            rowSelection="single"
        )

        column_ui = {
            "last_update": {"header_name": "MAJ", "width": 150},
            "date": {"header_name": "Date", "width": 155, "pinned": "left"},
            "sport": {"header_name": "Sport", "width": 135},
            "home_team": {"header_name": "Equipe domicile", "width": 210, "pinned": "left"},
            "away_team": {"header_name": "Equipe exterieur", "width": 210, "pinned": "left"},
            "market": {"header_name": "Marche", "width": 175},
            "implied_probability": {"header_name": "Proba book", "width": 125},
            "confidence": {"header_name": "Confiance", "width": 125},
            "score_exact_1": {"header_name": "Score 1", "width": 110},
            "score_exact_1_proba": {"header_name": "Proba score 1", "width": 135},
            "score_exact_2": {"header_name": "Score 2", "width": 110},
            "score_exact_2_proba": {"header_name": "Proba score 2", "width": 135},
            "score_exact_3": {"header_name": "Score 3", "width": 110},
            "score_exact_3_proba": {"header_name": "Proba score 3", "width": 135},
            "draw_hunter": {"header_name": "Draw", "width": 105},
            "scorer_prediction": {"header_name": "Buteur", "width": 190},
            "home_xg": {"header_name": "xG dom.", "width": 105},
            "away_xg": {"header_name": "xG ext.", "width": 105},
            "over_25": {"header_name": "Over 2.5", "width": 115},
            "under_25": {"header_name": "Under 2.5", "width": 115},
            "btts_yes": {"header_name": "BTTS Oui", "width": 115},
            "btts_no": {"header_name": "BTTS Non", "width": 115},
        }

        for col, options in column_ui.items():
            if col in data.columns:
                gb.configure_column(
                    col,
                    **options,
                    wrapText=True,
                    autoHeight=False,
                    cellStyle={"display": "flex", "alignItems": "center"}
                )

        if "value" in data.columns:
            gb.configure_column(
                "value",
                width=112,
                header_name="Value",
                valueFormatter="x == null ? '' : (x * 100).toFixed(1) + '%'",
                cellStyle=JsCode("""
                function(params) {
                    if (params.value >= 0.25) {
                        return {'backgroundColor':'rgba(53,208,127,0.28)','color':'#ecfff5','fontWeight':'800','display':'flex','alignItems':'center'};
                    }
                    if (params.value >= 0.10) {
                        return {'backgroundColor':'rgba(243,195,79,0.24)','color':'#fff4c6','fontWeight':'800','display':'flex','alignItems':'center'};
                    }
                    if (params.value > 0) {
                        return {'backgroundColor':'rgba(78,161,255,0.12)','color':'#eaf4ff','display':'flex','alignItems':'center'};
                    }
                    return {'color':'#7f918a','display':'flex','alignItems':'center'};
                }
                """)
            )

        if "ai_probability" in data.columns:
            gb.configure_column(
                "ai_probability",
                width=122,
                header_name="Proba IA",
                valueFormatter="x == null ? '' : (x * 100).toFixed(1) + '%'",
                cellStyle=JsCode("""
                function(params) {
                    if (params.value >= 0.70) {
                        return {'backgroundColor':'rgba(78,161,255,0.24)','color':'#ffffff','fontWeight':'800','display':'flex','alignItems':'center'};
                    }
                    if (params.value >= 0.55) {
                        return {'backgroundColor':'rgba(53,208,127,0.14)','color':'#eafff3','display':'flex','alignItems':'center'};
                    }
                    return {'color':'#cbd8d3','display':'flex','alignItems':'center'};
                }
                """)
            )

        if "decision" in data.columns:
            gb.configure_column(
                "decision",
                width=140,
                header_name="Decision",
                cellStyle=JsCode("""
                function(params) {
                    if (params.value === 'VALUE BET') {
                        return {'backgroundColor':'rgba(53,208,127,0.22)','color':'#adffd2','fontWeight':'800','display':'flex','alignItems':'center'};
                    }
                    return {'color':'#cbd8d3','display':'flex','alignItems':'center'};
                }
                """)
            )

        if "ia_badge" in data.columns:
            gb.configure_column(
                "ia_badge",
                width=145,
                header_name="Badge IA",
                cellStyle=JsCode("""
                function(params) {
                    if (params.value && params.value.includes('STRONG')) {
                        return {'color':'#75f0a5','fontWeight':'800','display':'flex','alignItems':'center'};
                    }
                    if (params.value && params.value.includes('MEDIUM')) {
                        return {'color':'#f3c34f','fontWeight':'800','display':'flex','alignItems':'center'};
                    }
                    if (params.value && params.value.includes('RISKY')) {
                        return {'color':'#ff8a8a','fontWeight':'800','display':'flex','alignItems':'center'};
                    }
                    return {'color':'#cbd8d3','display':'flex','alignItems':'center'};
                }
                """)
            )

        if "bookmaker_odds" in data.columns:
            gb.configure_column(
                "bookmaker_odds",
                width=95,
                header_name="Cote",
                valueFormatter="x == null ? '' : Number(x).toFixed(2)"
            )

        if "suggested_stake" in data.columns:
            gb.configure_column(
                "suggested_stake",
                width=105,
                header_name="Mise",
                valueFormatter="x == null ? '' : Number(x).toFixed(2) + ' EUR'"
            )

        grid_options = gb.build()

        custom_css = {
            ".ag-root-wrapper": {
                "background-color": "rgba(9,19,18,0.98) !important",
                "border": "1px solid rgba(210,232,222,0.16) !important",
                "border-radius": "8px !important",
                "box-shadow": "0 16px 36px rgba(0,0,0,0.28) !important",
                "overflow": "hidden !important",
            },
            ".ag-header": {
                "background": "#132320 !important",
                "color": "#ecf5f1 !important",
                "border-bottom": "1px solid rgba(210,232,222,0.18) !important",
                "font-weight": "800 !important",
            },
            ".ag-row": {
                "background-color": "#0d1816 !important",
                "color": "#e5f0eb !important",
                "border-bottom": "1px solid rgba(210,232,222,0.08) !important",
            },
            ".ag-row-odd": {
                "background-color": "#101d1b !important",
            },
            ".ag-row-hover": {
                "background-color": "rgba(53,208,127,0.10) !important",
            },
            ".ag-cell": {
                "border-right": "1px solid rgba(210,232,222,0.07) !important",
                "font-size": "13px !important",
                "line-height": "1.25 !important",
            },
            ".ag-floating-filter-input": {
                "background-color": "#091312 !important",
                "color": "#ecf5f1 !important",
                "border": "1px solid rgba(210,232,222,0.16) !important",
            },
            ".ag-pinned-left-header, .ag-pinned-left-cols-container": {
                "box-shadow": "6px 0 14px rgba(0,0,0,0.18) !important",
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


st.sidebar.title("Filtres")

if "last_update" in df.columns and not df.empty:
    st.sidebar.info(f"Derniere MAJ : {df['last_update'].iloc[0]}")

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
    "Marches",
    options=sorted(df["market"].dropna().unique()),
    default=sorted(df["market"].dropna().unique())
)

search = st.sidebar.text_input("Recherche instantanee")

only_value = st.sidebar.checkbox("Seulement VALUE BETS")
only_reliable = st.sidebar.checkbox("Seulement fiables")
mobile_mode = st.sidebar.checkbox("Mode telephone rapide")

min_value = st.sidebar.slider(
    "Value minimum",
    0.0,
    1.5,
    0.0,
    0.01
)

min_prob = st.sidebar.slider(
    "Probabilite IA minimum",
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


st.title("IA Paris Sportifs Ultime")
st.caption("Value Bets - Score Exact - Over/Under - BTTS - ROI - Tracking - Grille claire")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Lignes analysees", len(filtered))
col2.metric("Value Bets", len(value_bets))
col3.metric("Meilleure value", f"{round(filtered['value'].max()*100,2) if len(filtered) else 0}%")
col4.metric("Mise max", f"{round(filtered['suggested_stake'].max(),2) if len(filtered) else 0} EUR")


if mobile_mode:
    st.subheader("Mode telephone rapide")

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
    "Football",
    "Tennis",
    "Value Bets",
    "Score Exact",
    "Over/Under",
    "BTTS",
    "ROI / Tracking",
    "Toutes les predictions"
])


with tabs[0]:
    st.subheader("Football")

    football_df = filtered[
        filtered["sport"].astype(str).str.contains(
            "soccer",
            case=False,
            na=False
        )
    ]

    if football_df.empty:
        st.warning("Aucun match football trouve.")
    else:
        premium_table(
            football_df.sort_values("value", ascending=False).head(300),
            height=700,
            key="football_grid_unique"
        )


with tabs[1]:
    st.subheader("Tennis")

    tennis_df = filtered[
        filtered["sport"].astype(str).str.contains(
            "tennis",
            case=False,
            na=False
        )
    ]

    if tennis_df.empty:
        st.warning("Aucun match tennis trouve.")
    else:
        premium_table(
            tennis_df.sort_values("value", ascending=False).head(300),
            height=700,
            key="tennis_grid_unique"
        )


with tabs[2]:
    st.subheader("Paris recommandes")

    if value_bets.empty:
        st.warning("Aucun VALUE BET actuellement.")
    else:
        for _, row in value_bets.sort_values("value", ascending=False).head(8).iterrows():
            st.markdown(f"""
            <div class="value-card">
                <div class="big">{row['home_team']} vs {row['away_team']}</div>
                <div class="small">{row['sport']} - {row['date']}</div><br>
                <b>Pari :</b> {row['market']}<br>
                <b>Probabilite IA :</b> {round(float(row['ai_probability']) * 100, 1)}%<br>
                <b>Cote :</b> {row['bookmaker_odds']}<br>
                <b>Value :</b> {round(float(row['value']) * 100, 1)}%<br>
                <b>Badge :</b> {row.get('ia_badge', '')}<br>
                <b>Confiance :</b> {row['confidence']}<br>
                <b>Mise conseillee :</b> {row['suggested_stake']} EUR<br>
                <b>Score exact :</b> {row.get('score_exact_1', 'N/A')} - {row.get('score_exact_1_proba', 'N/A')}%<br>
                <b>Anytime goalscorer :</b> {row.get('scorer_prediction', 'N/A')}
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Tableau des VALUE BETS")

        premium_table(
            value_bets.sort_values("value", ascending=False).head(100),
            height=620,
            key="value_bets_grid_unique"
        )


with tabs[3]:
    st.subheader("Scores exacts probables")

    cols = [
        "date", "sport", "home_team", "away_team",
        "score_exact_1", "score_exact_1_proba",
        "score_exact_2", "score_exact_2_proba",
        "score_exact_3", "score_exact_3_proba",
        "draw_probability", "draw_hunter"
    ]

    available = [c for c in cols if c in filtered.columns]

    if not available:
        st.warning("Aucune colonne de score exact trouvee.")
    else:
        score_df = (
            filtered[available]
            .drop_duplicates(subset=["home_team", "away_team"])
        )

        if "score_exact_1_proba" in score_df.columns:
            score_df = score_df.sort_values("score_exact_1_proba", ascending=False)

        premium_table(
            score_df.head(150),
            height=650,
            key="score_grid_unique"
        )


with tabs[4]:
    st.subheader("Over / Under")

    cols = [
        "date", "sport", "home_team", "away_team",
        "over_25", "under_25", "home_xg", "away_xg"
    ]

    available = [c for c in cols if c in filtered.columns]

    if not available:
        st.warning("Aucune donnee Over/Under trouvee.")
    else:
        goals_df = (
            filtered[available]
            .drop_duplicates(subset=["home_team", "away_team"])
        )

        if "over_25" in goals_df.columns:
            goals_df = goals_df.sort_values("over_25", ascending=False)

        premium_table(
            goals_df.head(150),
            height=650,
            key="overunder_grid_unique"
        )


with tabs[5]:
    st.subheader("BTTS")

    cols = [
        "date", "sport", "home_team", "away_team",
        "btts_yes", "btts_no", "home_xg", "away_xg"
    ]

    available = [c for c in cols if c in filtered.columns]

    if not available:
        st.warning("Aucune donnee BTTS trouvee.")
    else:
        btts_df = (
            filtered[available]
            .drop_duplicates(subset=["home_team", "away_team"])
        )

        if "btts_yes" in btts_df.columns:
            btts_df = btts_df.sort_values("btts_yes", ascending=False)

        premium_table(
            btts_df.head(150),
            height=650,
            key="btts_grid_unique"
        )

with tabs[6]:
    st.subheader("Resultats IA / ROI / Rentabilite")

    tracking = load_tracking()

    if tracking.empty:
        st.warning("Aucun fichier tracking_results.csv trouve ou tracking vide.")
    else:
        if "result" not in tracking.columns:
            tracking["result"] = "PENDING"

        tracking["stake"] = pd.to_numeric(
            tracking.get("suggested_stake", tracking.get("stake", 0)),
            errors="coerce"
        ).fillna(0)

        tracking["bookmaker_odds"] = pd.to_numeric(
            tracking.get("bookmaker_odds", 0),
            errors="coerce"
        ).fillna(0)

        def calc_profit(row):
            if row["result"] == "WIN":
                return row["stake"] * (row["bookmaker_odds"] - 1)
            if row["result"] == "LOSS":
                return -row["stake"]
            return 0

        tracking["profit"] = tracking.apply(calc_profit, axis=1)

        finished = tracking[tracking["result"].isin(["WIN", "LOSS"])].copy()
        wins = tracking[tracking["result"] == "WIN"].copy()
        losses = tracking[tracking["result"] == "LOSS"].copy()
        pending = tracking[tracking["result"] == "PENDING"].copy()

        total_bets = len(tracking)
        finished_bets = len(finished)
        total_wins = len(wins)
        total_losses = len(losses)

        total_staked = finished["stake"].sum() if finished_bets else 0
        total_profit = finished["profit"].sum() if finished_bets else 0
        roi = total_profit / total_staked if total_staked > 0 else 0
        win_rate = total_wins / finished_bets if finished_bets > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Paris suivis", total_bets)
        c2.metric("Gagnes", total_wins)
        c3.metric("Perdus", total_losses)
        c4.metric("En attente", len(pending))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Mise totale", f"{round(total_staked, 2)} EUR")
        c6.metric("Profit / Perte", f"{round(total_profit, 2)} EUR")
        c7.metric("ROI", f"{round(roi * 100, 2)}%")
        c8.metric("Win Rate", f"{round(win_rate * 100, 2)}%")

        if total_profit > 0:
            st.success(f"L IA est en gain de +{round(total_profit, 2)} EUR")
        elif total_profit < 0:
            st.error(f"L IA est en perte de {round(total_profit, 2)} EUR")
        else:
            st.info("L IA est a l equilibre pour le moment.")

        if not finished.empty:
            finished["cumulative_profit"] = finished["profit"].cumsum()

            st.subheader("Courbe de rentabilite")
            st.line_chart(finished["cumulative_profit"])

            st.subheader("Gain / perte par pari")
            st.bar_chart(finished["profit"])

        st.subheader("Paris gagnes")
        if wins.empty:
            st.info("Aucun pari gagne pour le moment.")
        else:
            premium_table(
                wins.sort_values("profit", ascending=False).head(100),
                height=450,
                key="wins_grid_unique"
            )

        st.subheader("Paris perdus")
        if losses.empty:
            st.info("Aucun pari perdu pour le moment.")
        else:
            premium_table(
                losses.sort_values("profit", ascending=True).head(100),
                height=450,
                key="losses_grid_unique"
            )

        st.subheader("Paris en attente")
        if pending.empty:
            st.info("Aucun pari en attente.")
        else:
            premium_table(
                pending.sort_values("date", ascending=False).head(100),
                height=450,
                key="pending_grid_unique"
            )

        st.subheader("Historique complet IA")
        premium_table(
            tracking.sort_values("date", ascending=False).head(300),
            height=650,
            key="tracking_full_grid_unique"
        )

with tabs[7]:
    st.subheader("Toutes les predictions")

    premium_table(
        filtered.sort_values("value", ascending=False).head(300),
        height=750,
        key="all_predictions_grid_unique"
    )

