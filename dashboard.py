import streamlit as st
import pandas as pd
from pathlib import Path
from html import escape

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
    --bg: #0b0d10;
    --surface: #11161c;
    --surface-2: #171f28;
    --surface-3: #1d2732;
    --line: rgba(223, 231, 238, 0.12);
    --line-strong: rgba(223, 231, 238, 0.22);
    --text: #f4f7fa;
    --muted: #9aa8b4;
    --green: #29c36a;
    --blue: #4c8dff;
    --cyan: #38c6d9;
    --amber: #f1b84b;
    --red: #ee5d67;
}

.stApp {
    background:
        linear-gradient(180deg, rgba(76,141,255,0.10) 0%, rgba(11,13,16,0) 330px),
        linear-gradient(135deg, #0b0d10 0%, #0f151b 48%, #121318 100%);
    color: var(--text);
    font-family: Inter, "Segoe UI", system-ui, sans-serif;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image: linear-gradient(rgba(255,255,255,0.028) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(255,255,255,0.028) 1px, transparent 1px);
    background-size: 28px 28px;
    mask-image: linear-gradient(to bottom, black, transparent 70%);
    pointer-events: none;
    z-index: -1;
}

.block-container {
    max-width: 1600px;
    padding-top: 22px;
    padding-bottom: 48px;
}

section[data-testid="stSidebar"] {
    background: #0d1116;
    border-right: 1px solid var(--line-strong);
}

section[data-testid="stSidebar"] > div {
    padding-top: 18px;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #dce5ec !important;
}

.stTextInput input,
.stMultiSelect [data-baseweb="select"],
.stSelectbox [data-baseweb="select"] {
    background: #151c24 !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

.stTextInput input {
    background: #151c24;
    border: 1px solid var(--line-strong);
    border-radius: 8px;
    color: var(--text);
}

h1, h2, h3 {
    color: var(--text);
    letter-spacing: 0;
    text-shadow: none;
}

h1 {
    font-size: 36px !important;
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

.app-hero {
    display: grid;
    grid-template-columns: minmax(0, 1.55fr) minmax(280px, 0.45fr);
    gap: 18px;
    align-items: stretch;
    margin-bottom: 18px;
}

.hero-main {
    background: linear-gradient(135deg, rgba(76,141,255,0.18), rgba(41,195,106,0.08)), #11161c;
    border: 1px solid var(--line-strong);
    border-radius: 8px;
    padding: 22px 24px;
    box-shadow: 0 18px 45px rgba(0,0,0,0.25);
}

.hero-main .eyebrow,
.section-eyebrow {
    color: var(--cyan);
    font-size: 12px;
    text-transform: uppercase;
    font-weight: 850;
    letter-spacing: 0;
    margin-bottom: 8px;
}

.hero-title {
    font-size: 36px;
    line-height: 1.05;
    font-weight: 900;
    margin: 0 0 8px;
}

.hero-copy {
    color: var(--muted);
    font-size: 15px;
    margin: 0;
}

.hero-side {
    background: #151c24;
    border: 1px solid var(--line-strong);
    border-radius: 8px;
    padding: 18px;
}

.status-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid var(--line);
}

.status-row:last-child {
    border-bottom: 0;
}

.status-label {
    color: var(--muted);
    font-size: 12px;
}

.status-value {
    color: var(--text);
    font-weight: 800;
    text-align: right;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 12px 0 22px;
}

.kpi-card {
    background: linear-gradient(180deg, #171f28, #11161c);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 15px 16px;
    min-height: 92px;
}

.kpi-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 750;
    text-transform: uppercase;
}

.kpi-value {
    display: block;
    margin-top: 9px;
    color: var(--text);
    font-size: 26px;
    font-weight: 900;
}

.kpi-note {
    color: var(--muted);
    font-size: 12px;
    margin-top: 5px;
}

.section-head {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 18px;
    margin: 20px 0 12px;
}

.section-title {
    margin: 0;
    font-size: 22px;
    font-weight: 900;
}

.section-subtitle {
    margin: 4px 0 0;
    color: var(--muted);
    font-size: 13px;
}

.match-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin: 0 0 14px;
}

.match-card {
    background: #121820;
    border: 1px solid var(--line);
    border-left: 4px solid var(--blue);
    border-radius: 8px;
    padding: 14px;
    min-height: 172px;
}

.match-card.value-bet {
    border-left-color: var(--green);
    background: linear-gradient(135deg, rgba(41,195,106,0.10), rgba(18,24,32,1) 46%);
}

.match-top {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 10px;
}

.match-teams {
    font-size: 17px;
    line-height: 1.25;
    font-weight: 900;
    color: var(--text);
    margin-bottom: 10px;
}

.match-market {
    color: #d8e2ea;
    font-size: 13px;
    margin-bottom: 12px;
}

.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
}

.pill {
    border: 1px solid var(--line);
    background: #19222c;
    border-radius: 999px;
    color: #dce5ec;
    font-size: 12px;
    font-weight: 800;
    padding: 5px 8px;
}

.pill.good {
    color: #b9f8cd;
    background: rgba(41,195,106,0.14);
    border-color: rgba(41,195,106,0.34);
}

.pill.warn {
    color: #ffe1a3;
    background: rgba(241,184,75,0.14);
    border-color: rgba(241,184,75,0.34);
}

.value-card {
    padding: 0;
    border-radius: 8px;
    margin-bottom: 12px;
    background: transparent;
    border: 0;
    box-shadow: none;
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
    gap: 4px;
    border-bottom: 1px solid var(--line);
    background: #11161c;
    padding: 8px 8px 0;
    border-radius: 8px 8px 0 0;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px 8px 0 0;
    color: #aebbc6;
    border: 1px solid transparent;
    padding: 10px 14px;
    font-weight: 700;
}

.stTabs [aria-selected="true"] {
    background: #1d2732;
    color: #ffffff;
    border-color: var(--line);
    border-bottom-color: var(--blue);
}

.stAlert {
    border-radius: 8px;
}

div[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

@media (max-width: 1100px) {
    .app-hero,
    .kpi-grid,
    .match-grid {
        grid-template-columns: 1fr;
    }
}

/* Clean full redesign override */
:root {
    --bg: #f3f6f8;
    --surface: #ffffff;
    --surface-2: #f8fafc;
    --surface-3: #eef3f7;
    --line: #d9e2ea;
    --line-strong: #c6d2dd;
    --text: #111827;
    --muted: #64748b;
    --green: #16a34a;
    --blue: #2563eb;
    --cyan: #0891b2;
    --amber: #d97706;
    --red: #dc2626;
}

.stApp {
    background: #f3f6f8 !important;
    color: var(--text) !important;
}

.stApp::before {
    display: none;
}

.block-container {
    max-width: 1480px;
    padding: 24px 28px 48px;
}

section[data-testid="stSidebar"] {
    background: #101827 !important;
    border-right: 0 !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #f8fafc !important;
}

section[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #2563eb !important;
    color: white !important;
    border-radius: 6px !important;
    max-width: 100% !important;
}

.stTextInput input,
.stMultiSelect [data-baseweb="select"],
.stSelectbox [data-baseweb="select"] {
    background: #ffffff !important;
    border: 1px solid #d7e0e8 !important;
    color: #111827 !important;
}

section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] {
    background: #172235 !important;
    border-color: rgba(255,255,255,0.16) !important;
    color: #f8fafc !important;
}

h1, h2, h3 {
    color: #111827 !important;
}

.app-hero {
    grid-template-columns: 1fr;
    margin-bottom: 14px;
}

.hero-main {
    background: #ffffff !important;
    border: 1px solid #d9e2ea !important;
    box-shadow: 0 12px 28px rgba(15,23,42,0.08) !important;
}

.hero-title {
    color: #111827;
    font-size: 34px;
}

.hero-copy,
.section-subtitle,
.kpi-note,
.status-label {
    color: #64748b !important;
}

.hero-side {
    display: none;
}

.kpi-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
}

.kpi-card {
    background: #ffffff !important;
    border: 1px solid #d9e2ea !important;
    box-shadow: 0 10px 22px rgba(15,23,42,0.06);
}

.kpi-label {
    color: #64748b !important;
}

.kpi-value {
    color: #0f172a !important;
}

.section-head {
    background: #ffffff;
    border: 1px solid #d9e2ea;
    border-radius: 8px;
    padding: 16px 18px;
    margin-top: 18px;
}

.section-eyebrow {
    color: #2563eb !important;
}

.section-title {
    color: #111827 !important;
}

.match-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
}

.match-card {
    background: #ffffff !important;
    border: 1px solid #d9e2ea !important;
    border-left: 5px solid #2563eb !important;
    box-shadow: 0 10px 22px rgba(15,23,42,0.06);
}

.match-card.value-bet {
    border-left-color: #16a34a !important;
    background: #ffffff !important;
}

.match-teams,
.match-market,
.status-value {
    color: #111827 !important;
}

.pill {
    background: #f1f5f9 !important;
    border-color: #d9e2ea !important;
    color: #334155 !important;
}

.pill.good {
    background: #dcfce7 !important;
    color: #166534 !important;
    border-color: #bbf7d0 !important;
}

.pill.warn {
    background: #fef3c7 !important;
    color: #92400e !important;
    border-color: #fde68a !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border: 1px solid #d9e2ea !important;
    border-radius: 8px !important;
    padding: 6px !important;
    gap: 4px !important;
}

.stTabs [data-baseweb="tab"] {
    color: #475569 !important;
    border-radius: 6px !important;
}

.stTabs [aria-selected="true"] {
    background: #2563eb !important;
    color: #ffffff !important;
    border-color: #2563eb !important;
}

@media (max-width: 1180px) {
    .kpi-grid,
    .match-grid {
        grid-template-columns: 1fr;
    }
}

/* Final dark theme */
:root {
    --bg: #080d12;
    --surface: #101820;
    --surface-2: #141f2a;
    --surface-3: #1b2936;
    --line: rgba(226, 232, 240, 0.12);
    --line-strong: rgba(226, 232, 240, 0.20);
    --text: #f8fafc;
    --muted: #9fb0c3;
    --green: #22c55e;
    --blue: #38bdf8;
    --cyan: #2dd4bf;
    --amber: #f59e0b;
    --red: #fb7185;
}

.stApp {
    background:
        linear-gradient(180deg, rgba(56,189,248,0.10), rgba(8,13,18,0) 340px),
        #080d12 !important;
    color: var(--text) !important;
}

.block-container {
    max-width: 1500px;
}

section[data-testid="stSidebar"] {
    background: #0b1118 !important;
    border-right: 1px solid rgba(226,232,240,0.12) !important;
}

section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] {
    background: #121b25 !important;
    border-color: rgba(226,232,240,0.16) !important;
    color: #f8fafc !important;
}

section[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #1e3a8a !important;
    border: 1px solid rgba(147,197,253,0.30) !important;
    color: #dbeafe !important;
}

h1, h2, h3,
.hero-title,
.section-title,
.match-teams,
.match-market,
.status-value {
    color: #f8fafc !important;
}

.hero-main,
.hero-side,
.kpi-card,
.section-head,
.match-card {
    background: #101820 !important;
    border: 1px solid rgba(226,232,240,0.12) !important;
    box-shadow: 0 16px 35px rgba(0,0,0,0.26) !important;
}

.hero-main {
    background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(34,197,94,0.06)), #101820 !important;
}

.hero-copy,
.section-subtitle,
.kpi-note,
.status-label,
.match-top {
    color: #9fb0c3 !important;
}

.kpi-label {
    color: #7dd3fc !important;
}

.kpi-value {
    color: #ffffff !important;
}

.section-head {
    background: #0f1720 !important;
}

.section-eyebrow {
    color: #2dd4bf !important;
}

.match-card {
    border-left: 5px solid #38bdf8 !important;
}

.match-card.value-bet {
    border-left-color: #22c55e !important;
    background: linear-gradient(135deg, rgba(34,197,94,0.09), rgba(16,24,32,1) 46%) !important;
}

.pill {
    background: #172333 !important;
    border-color: rgba(226,232,240,0.13) !important;
    color: #dbe7f3 !important;
}

.pill.good {
    background: rgba(34,197,94,0.16) !important;
    color: #bbf7d0 !important;
    border-color: rgba(34,197,94,0.28) !important;
}

.pill.warn {
    background: rgba(245,158,11,0.16) !important;
    color: #fde68a !important;
    border-color: rgba(245,158,11,0.28) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #101820 !important;
    border: 1px solid rgba(226,232,240,0.12) !important;
}

.stTabs [data-baseweb="tab"] {
    color: #cbd5e1 !important;
}

.stTabs [aria-selected="true"] {
    background: #1b2936 !important;
    color: #ffffff !important;
    border-color: rgba(56,189,248,0.34) !important;
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


def fmt_percent(value):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "N/A"


def fmt_number(value, suffix="", decimals=2):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.{decimals}f}{suffix}"
    except Exception:
        return "N/A"


def safe_text(value, default="N/A"):
    if value is None or pd.isna(value):
        return default
    text = str(value)
    if not text.strip():
        return default
    return escape(text)


def kpi_cards(items):
    html = ['<div class="kpi-grid">']
    for label, value, note in items:
        html.append(
            '<div class="kpi-card">'
            f'<div class="kpi-label">{escape(str(label))}</div>'
            f'<span class="kpi-value">{escape(str(value))}</span>'
            f'<div class="kpi-note">{escape(str(note))}</div>'
            '</div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def section_header(title, subtitle="", right=""):
    html = (
        '<div class="section-head">'
        '<div>'
        '<div class="section-eyebrow">Analyse</div>'
        f'<h2 class="section-title">{escape(str(title))}</h2>'
        f'<p class="section-subtitle">{escape(str(subtitle))}</p>'
        '</div>'
        f'<div class="status-value">{escape(str(right))}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def match_cards(dataframe, limit=6):
    if dataframe.empty:
        return

    cards = ['<div class="match-grid">']
    for _, row in dataframe.head(limit).iterrows():
        decision = str(row.get("decision", ""))
        card_class = "match-card value-bet" if decision == "VALUE BET" else "match-card"
        teams = f"{safe_text(row.get('home_team'))} vs {safe_text(row.get('away_team'))}"
        date = safe_text(row.get("date"))
        sport = safe_text(row.get("sport"))
        market = safe_text(row.get("market"))
        confidence = safe_text(row.get("confidence"))
        badge = safe_text(row.get("ia_badge", ""))
        score = safe_text(row.get("score_exact_1", "N/A"))

        cards.append(
            f'<div class="{card_class}">'
            '<div class="match-top">'
            f'<span>{sport}</span>'
            f'<span>{date}</span>'
            '</div>'
            f'<div class="match-teams">{teams}</div>'
            f'<div class="match-market">{market}</div>'
            '<div class="pill-row">'
            f'<span class="pill good">IA {fmt_percent(row.get("ai_probability"))}</span>'
            f'<span class="pill warn">Value {fmt_percent(row.get("value"))}</span>'
            f'<span class="pill">Cote {fmt_number(row.get("bookmaker_odds"), "", 2)}</span>'
            f'<span class="pill">Mise {fmt_number(row.get("suggested_stake"), " EUR", 2)}</span>'
            f'<span class="pill">{confidence}</span>'
            f'<span class="pill">{badge}</span>'
            f'<span class="pill">Score {score}</span>'
            '</div>'
            '</div>'
        )
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)


CORE_COLS = [
    "date", "sport", "home_team", "away_team", "market", "ai_probability",
    "bookmaker_odds", "value", "confidence", "ia_badge", "decision",
    "suggested_stake"
]

SCORE_COLS = [
    "date", "sport", "home_team", "away_team", "score_exact_1",
    "score_exact_1_proba", "score_exact_2", "score_exact_2_proba",
    "score_exact_3", "score_exact_3_proba", "draw_hunter"
]

GOALS_COLS = [
    "date", "sport", "home_team", "away_team", "over_25", "under_25",
    "home_xg", "away_xg"
]

BTTS_COLS = [
    "date", "sport", "home_team", "away_team", "btts_yes", "btts_no",
    "home_xg", "away_xg"
]


def prediction_board(dataframe, height=650, key="table", cards=6, columns=None):
    if dataframe.empty:
        st.warning("Aucune donnee a afficher.")
        return
    match_cards(dataframe, limit=cards)
    premium_table(dataframe, height=height, key=key, columns=columns)


def premium_table(dataframe, height=650, key="table", columns=None):
    if dataframe.empty:
        st.warning("Aucune donnee a afficher.")
        return

    data = dataframe.copy()

    preferred_cols = columns or [
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
            editable=False,
            minWidth=105
        )

        gb.configure_grid_options(
            rowHeight=54,
            headerHeight=50,
            floatingFiltersHeight=38,
            animateRows=True,
            enableCellTextSelection=True,
            suppressCellFocus=True,
            rowSelection="single",
            suppressRowClickSelection=False,
            suppressHorizontalScroll=False,
            tooltipShowDelay=250
        )
        gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=25)

        column_ui = {
            "last_update": {"header_name": "MAJ", "width": 150},
            "date": {"header_name": "Date", "width": 160, "pinned": "left"},
            "sport": {"header_name": "Sport", "width": 135},
            "home_team": {"header_name": "Domicile", "width": 230, "pinned": "left"},
            "away_team": {"header_name": "Exterieur", "width": 230, "pinned": "left"},
            "market": {"header_name": "Marche", "width": 190},
            "implied_probability": {"header_name": "Proba book", "width": 130},
            "confidence": {"header_name": "Confiance", "width": 135},
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
                    tooltipField=col,
                    cellStyle={"display": "flex", "alignItems": "center", "whiteSpace": "normal"}
                )

        if "value" in data.columns:
            gb.configure_column(
                "value",
                width=118,
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
                width=126,
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
                width=145,
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
                width=155,
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
                width=100,
                header_name="Cote",
                valueFormatter="x == null ? '' : Number(x).toFixed(2)"
            )

        if "suggested_stake" in data.columns:
            gb.configure_column(
                "suggested_stake",
                width=110,
                header_name="Mise",
                valueFormatter="x == null ? '' : Number(x).toFixed(2) + ' EUR'"
            )

        grid_options = gb.build()

        custom_css = {
            ".ag-root-wrapper": {
                "background-color": "#101820 !important",
                "border": "1px solid rgba(226,232,240,0.14) !important",
                "border-radius": "8px !important",
                "box-shadow": "0 16px 35px rgba(0,0,0,0.26) !important",
                "overflow": "hidden !important",
            },
            ".ag-header": {
                "background": "#172333 !important",
                "color": "#f8fafc !important",
                "border-bottom": "1px solid rgba(226,232,240,0.14) !important",
                "font-weight": "800 !important",
            },
            ".ag-header-cell-label": {
                "justify-content": "flex-start !important",
            },
            ".ag-header-cell-text": {
                "font-size": "12px !important",
                "text-transform": "uppercase !important",
                "letter-spacing": "0 !important",
            },
            ".ag-row": {
                "background-color": "#101820 !important",
                "color": "#e5edf6 !important",
                "border-bottom": "1px solid rgba(226,232,240,0.08) !important",
            },
            ".ag-row-odd": {
                "background-color": "#121d28 !important",
            },
            ".ag-row-hover": {
                "background-color": "rgba(56,189,248,0.12) !important",
            },
            ".ag-cell": {
                "border-right": "1px solid rgba(226,232,240,0.07) !important",
                "font-size": "13px !important",
                "line-height": "1.25 !important",
                "padding-left": "12px !important",
                "padding-right": "12px !important",
            },
            ".ag-floating-filter-input": {
                "background-color": "#0b1118 !important",
                "color": "#f8fafc !important",
                "border": "1px solid rgba(226,232,240,0.16) !important",
                "border-radius": "6px !important",
            },
            ".ag-pinned-left-header, .ag-pinned-left-cols-container": {
                "box-shadow": "6px 0 14px rgba(0,0,0,0.18) !important",
            },
            ".ag-paging-panel": {
                "background": "#101820 !important",
                "color": "#cbd5e1 !important",
                "border-top": "1px solid rgba(226,232,240,0.12) !important",
            },
            ".ag-menu": {
                "background": "#101820 !important",
                "color": "#f8fafc !important",
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


last_update = df["last_update"].iloc[0] if "last_update" in df.columns and not df.empty else "N/A"
best_value = f"{round(filtered['value'].max() * 100, 2) if len(filtered) else 0}%"
max_stake = f"{round(filtered['suggested_stake'].max(), 2) if len(filtered) else 0} EUR"
avg_probability = f"{round(filtered['ai_probability'].mean() * 100, 1) if len(filtered) else 0}%"

hero_html = (
    '<div class="app-hero">'
    '<div class="hero-main">'
    '<div class="eyebrow">Dashboard IA</div>'
    '<div class="hero-title">Paris sportifs, decisions et value bets</div>'
    '<p class="hero-copy">Vue operationnelle pour lire rapidement les matchs, comparer les probabilites IA, '
    'isoler les value bets et suivre le ROI sans perdre les options existantes.</p>'
    '</div>'
    '<div class="hero-side">'
    '<div class="status-row"><span class="status-label">Derniere MAJ</span>'
    f'<span class="status-value">{safe_text(last_update)}</span></div>'
    '<div class="status-row"><span class="status-label">Sports actifs</span>'
    f'<span class="status-value">{len(sports)}</span></div>'
    '<div class="status-row"><span class="status-label">Marches actifs</span>'
    f'<span class="status-value">{len(markets)}</span></div>'
    '</div>'
    '</div>'
)
st.markdown(hero_html, unsafe_allow_html=True)

kpi_cards([
    ("Lignes analysees", len(filtered), "Apres filtres actifs"),
    ("Value Bets", len(value_bets), "Opportunites detectees"),
    ("Meilleure value", best_value, "Ecart IA vs bookmaker"),
    ("Proba IA moyenne", avg_probability, f"Mise max {max_stake}"),
])


if mobile_mode:
    section_header("Mode telephone rapide", "Les meilleurs matchs en format compact.", f"{len(filtered)} lignes")

    mobile_cols = [
        "date", "sport", "home_team", "away_team", "market",
        "ai_probability", "bookmaker_odds", "value",
        "confidence", "ia_badge", "suggested_stake",
        "score_exact_1", "score_exact_1_proba",
        "draw_hunter", "scorer_prediction"
    ]

    mobile_cols = [c for c in mobile_cols if c in filtered.columns]

    prediction_board(
        filtered[mobile_cols].sort_values("value", ascending=False).head(50),
        height=550,
        key="mobile_grid",
        cards=4,
        columns=mobile_cols
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
    section_header("Football", "Matchs classes par value, avec les equipes et le marche visibles tout de suite.")

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
        prediction_board(
            football_df.sort_values("value", ascending=False).head(300),
            height=700,
            key="football_grid_unique",
            cards=6,
            columns=CORE_COLS
        )


with tabs[1]:
    section_header("Tennis", "Selection tennis avec probabilites IA, cotes et value en lecture rapide.")

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
        prediction_board(
            tennis_df.sort_values("value", ascending=False).head(300),
            height=700,
            key="tennis_grid_unique",
            cards=6,
            columns=CORE_COLS
        )


with tabs[2]:
    section_header("Paris recommandes", "Les meilleurs value bets ressortent en cartes, puis restent disponibles dans la grille complete.")

    if value_bets.empty:
        st.warning("Aucun VALUE BET actuellement.")
    else:
        match_cards(value_bets.sort_values("value", ascending=False), limit=9)
        section_header("Grille VALUE BETS", "Toutes les colonnes restent disponibles, filtrables et triables.", f"{len(value_bets)} bets")

        premium_table(
            value_bets.sort_values("value", ascending=False).head(100),
            height=620,
            key="value_bets_grid_unique",
            columns=CORE_COLS + ["score_exact_1", "score_exact_1_proba", "scorer_prediction"]
        )


with tabs[3]:
    section_header("Scores exacts probables", "Priorite aux matchs avec les probabilites de score les plus fortes.")

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

        prediction_board(
            score_df.head(150),
            height=650,
            key="score_grid_unique",
            cards=6,
            columns=SCORE_COLS
        )


with tabs[4]:
    section_header("Over / Under", "Lecture orientee buts avec over, under et xG au meme endroit.")

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

        prediction_board(
            goals_df.head(150),
            height=650,
            key="overunder_grid_unique",
            cards=6,
            columns=GOALS_COLS
        )


with tabs[5]:
    section_header("BTTS", "Selection Both Teams To Score avec xG et probabilites comparees.")

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

        prediction_board(
            btts_df.head(150),
            height=650,
            key="btts_grid_unique",
            cards=6,
            columns=BTTS_COLS
        )

with tabs[6]:
    section_header("Resultats IA / ROI / Rentabilite", "Suivi de performance des paris envoyes et historiques.")

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

        kpi_cards([
            ("Paris suivis", total_bets, "Historique complet"),
            ("Gagnes", total_wins, "Bets termines en WIN"),
            ("Perdus", total_losses, "Bets termines en LOSS"),
            ("En attente", len(pending), "Bets encore ouverts"),
        ])

        kpi_cards([
            ("Mise totale", f"{round(total_staked, 2)} EUR", "Sur les bets termines"),
            ("Profit / Perte", f"{round(total_profit, 2)} EUR", "Resultat net"),
            ("ROI", f"{round(roi * 100, 2)}%", "Profit / mise"),
            ("Win Rate", f"{round(win_rate * 100, 2)}%", "WIN / bets termines"),
        ])

        if total_profit > 0:
            st.success(f"L IA est en gain de +{round(total_profit, 2)} EUR")
        elif total_profit < 0:
            st.error(f"L IA est en perte de {round(total_profit, 2)} EUR")
        else:
            st.info("L IA est a l equilibre pour le moment.")

        if not finished.empty:
            finished["cumulative_profit"] = finished["profit"].cumsum()

            section_header("Courbe de rentabilite", "Evolution cumulee du profit.")
            st.line_chart(finished["cumulative_profit"])

            section_header("Gain / perte par pari", "Chaque barre represente un pari termine.")
            st.bar_chart(finished["profit"])

        section_header("Paris gagnes", "Historique des paris gagnants.", f"{len(wins)} lignes")
        if wins.empty:
            st.info("Aucun pari gagne pour le moment.")
        else:
            premium_table(
                wins.sort_values("profit", ascending=False).head(100),
                height=450,
                key="wins_grid_unique"
            )

        section_header("Paris perdus", "Historique des paris perdants.", f"{len(losses)} lignes")
        if losses.empty:
            st.info("Aucun pari perdu pour le moment.")
        else:
            premium_table(
                losses.sort_values("profit", ascending=True).head(100),
                height=450,
                key="losses_grid_unique"
            )

        section_header("Paris en attente", "Paris ouverts ou non encore renseignes.", f"{len(pending)} lignes")
        if pending.empty:
            st.info("Aucun pari en attente.")
        else:
            premium_table(
                pending.sort_values("date", ascending=False).head(100),
                height=450,
                key="pending_grid_unique"
            )

        section_header("Historique complet IA", "Toutes les decisions suivies dans le tracking.", f"{len(tracking)} lignes")
        premium_table(
            tracking.sort_values("date", ascending=False).head(300),
            height=650,
            key="tracking_full_grid_unique"
        )

with tabs[7]:
    section_header("Toutes les predictions", "Vue complete triee par value, avec cartes rapides et grille detaillee.", f"{len(filtered)} lignes")

    prediction_board(
        filtered.sort_values("value", ascending=False).head(300),
        height=750,
        key="all_predictions_grid_unique",
        cards=9,
        columns=CORE_COLS
    )

