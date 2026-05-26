import streamlit as st
import pandas as pd
from pathlib import Path
from html import escape

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


st.set_page_config(page_title="IA Paris Sportifs Ultime", layout="wide", page_icon="PS")

APP_PASSWORD = "29052007"
PRED_PATH = Path("data/predictions/predictions_today.csv")
TRACK_PATH = Path("tracking_results.csv")
TELEGRAM_SENT_PATH = Path("data/telegram_sent.csv")


# =========================
# STYLE
# =========================

st.markdown("""
<style>
:root {
    --bg: #10131f;
    --panel: #171b2b;
    --panel-2: #20263a;
    --line: rgba(255,255,255,.12);
    --text: #f7f8ff;
    --muted: #9ea8c4;
    --pink: #f472d0;
    --purple: #a855f7;
    --cyan: #22d3ee;
    --blue: #60a5fa;
    --green: #34d399;
    --amber: #fbbf24;
    --red: #fb7185;
}
.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(168,85,247,.28), transparent 30%),
        radial-gradient(circle at 100% 100%, rgba(34,211,238,.18), transparent 28%),
        linear-gradient(135deg, #111827 0%, #10131f 45%, #0d1020 100%);
    color: var(--text);
    font-family: Inter, "Segoe UI", system-ui, sans-serif;
}
.block-container { max-width: 1580px; padding-top: 28px; padding-bottom: 48px; }
section[data-testid="stSidebar"] { background: #141827; border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] * { color: #eef2ff !important; }
h1, h2, h3 { color: var(--text) !important; }
.shell {
    background: rgba(18,22,36,.88);
    border: 1px solid var(--line);
    border-radius: 26px;
    padding: 26px;
    box-shadow: 0 24px 60px rgba(0,0,0,.35);
}
.hero {
    display: grid;
    grid-template-columns: 1.2fr .8fr;
    gap: 18px;
    margin-bottom: 20px;
}
.hero-card, .panel-card, .kpi-card, .match-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.03);
}
.hero-card {
    padding: 24px;
    background: linear-gradient(135deg, rgba(244,114,208,.18), rgba(34,211,238,.08)), var(--panel);
}
.eyebrow { color: var(--pink); font-size: 12px; font-weight: 900; text-transform: uppercase; margin-bottom: 10px; }
.hero-title { font-size: 34px; line-height: 1.08; font-weight: 950; margin-bottom: 10px; }
.hero-copy { color: var(--muted); font-size: 14px; max-width: 780px; }
.status-card { padding: 18px; }
.status-line { display: flex; justify-content: space-between; gap: 14px; border-bottom: 1px solid var(--line); padding: 10px 0; }
.status-line:last-child { border-bottom: 0; }
.status-label { color: var(--muted); font-size: 12px; }
.status-value { font-weight: 800; text-align: right; }
.kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin: 18px 0; }
.kpi-card { min-height: 122px; padding: 18px; overflow: hidden; position: relative; }
.kpi-card:after {
    content: "";
    position: absolute;
    width: 84px;
    height: 84px;
    right: -20px;
    bottom: -28px;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(244,114,208,.25), rgba(34,211,238,.18));
}
.kpi-label { color: var(--muted); font-size: 12px; font-weight: 800; text-transform: uppercase; }
.kpi-value { display: block; color: var(--text); font-size: 32px; font-weight: 950; margin-top: 12px; }
.kpi-note { color: var(--muted); font-size: 12px; margin-top: 6px; }
.match-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-bottom: 16px; }
.match-card { padding: 16px; min-height: 174px; border-left: 4px solid var(--blue); }
.match-card.value {
    border-left-color: var(--green);
    background: linear-gradient(135deg, rgba(52,211,153,.12), rgba(23,27,43,0) 50%), var(--panel);
}
.match-top { display: flex; justify-content: space-between; gap: 10px; color: var(--muted); font-size: 12px; margin-bottom: 12px; }
.match-teams { font-size: 18px; font-weight: 950; line-height: 1.22; margin-bottom: 8px; }
.badge {
    display: inline-block;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 900;
    border: 1px solid var(--line);
}
.badge-green { background: rgba(52,211,153,.15); color: var(--green); }
.badge-red { background: rgba(251,113,133,.15); color: var(--red); }
.badge-amber { background: rgba(251,191,36,.15); color: var(--amber); }
.badge-blue { background: rgba(96,165,250,.15); color: var(--blue); }
.table-wrap {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 14px;
}
.stDataFrame { border-radius: 14px !important; overflow: hidden !important; }
@media (max-width: 1000px) {
    .hero { grid-template-columns: 1fr; }
    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .match-grid { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)


# =========================
# LOGIN
# =========================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <div class="shell">
        <div class="hero-card">
            <div class="eyebrow">Accès privé</div>
            <div class="hero-title">IA Paris Sportifs Ultime</div>
            <div class="hero-copy">Plateforme privée : football, tennis, value betting, ROI, tracking, Telegram et analyse IA.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    password = st.text_input("Mot de passe", type="password")

    if password == APP_PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
    elif password:
        st.error("Mot de passe incorrect.")

    st.stop()


# =========================
# LOAD
# =========================

@st.cache_data(ttl=300)
def load_predictions():
    if PRED_PATH.exists():
        return pd.read_csv(PRED_PATH, low_memory=False)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_tracking():
    if TRACK_PATH.exists():
        return pd.read_csv(TRACK_PATH, low_memory=False)
    return pd.DataFrame()


df = load_predictions()
tracking = load_tracking()

if df.empty:
    st.error("Aucune prédiction trouvée.")
    st.stop()


# =========================
# HELPERS
# =========================

def to_num(data, col, default=0):
    if col in data.columns:
        return pd.to_numeric(data[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(data), index=data.index)


def category_of_sport(sport):
    s = str(sport).lower()
    if "tennis" in s:
        return "tennis"
    if "soccer" in s or "football" in s:
        return "football"
    return "autre"


def format_table(data):
    out = data.copy()

    cols = [
        "date", "sport", "category", "home_team", "away_team", "market", "selection",
        "ai_probability", "bookmaker_odds", "implied_probability", "value",
        "confidence", "ia_badge", "decision", "suggested_stake",
        "score_exact_1", "score_exact_1_proba", "draw_hunter",
        "scorer_prediction", "tennis_engine_score", "tennis_edge"
    ]

    cols = [c for c in cols if c in out.columns]
    out = out[cols]

    for col in ["ai_probability", "implied_probability", "value", "tennis_edge"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").apply(
                lambda x: "" if pd.isna(x) else f"{x*100:.2f}%"
            )

    for col in ["bookmaker_odds", "suggested_stake", "tennis_engine_score"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    return out


def render_table(data, height=480):
    st.markdown('<div class="table-wrap">', unsafe_allow_html=True)
    st.dataframe(
        format_table(data),
        use_container_width=True,
        hide_index=True,
        height=height
    )
    st.markdown('</div>', unsafe_allow_html=True)


def kpi(label, value, note=""):
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{escape(str(label))}</div>
        <span class="kpi-value">{escape(str(value))}</span>
        <div class="kpi-note">{escape(str(note))}</div>
    </div>
    """


def render_match_cards(data, limit=6):
    if data.empty:
        st.info("Aucun pari à afficher.")
        return

    html = '<div class="match-grid">'

    for _, row in data.head(limit).iterrows():
        decision = row.get("decision", "")
        cls = "match-card value" if decision == "VALUE BET" else "match-card"

        value = pd.to_numeric(pd.Series([row.get("value", 0)]), errors="coerce").fillna(0).iloc[0]
        proba = pd.to_numeric(pd.Series([row.get("ai_probability", 0)]), errors="coerce").fillna(0).iloc[0]

        html += f"""
        <div class="{cls}">
            <div class="match-top">
                <span>{escape(str(row.get("sport", "")))}</span>
                <span>{escape(str(row.get("date", "")))}</span>
            </div>
            <div class="match-teams">{escape(str(row.get("home_team", "")))}<br>vs {escape(str(row.get("away_team", "")))}</div>
            <div><span class="badge badge-blue">{escape(str(row.get("market", "")))}</span></div>
            <br>
            <div>Proba IA : <b>{proba*100:.1f}%</b></div>
            <div>Cote : <b>{escape(str(row.get("bookmaker_odds", "")))}</b></div>
            <div>Value : <b>{value*100:.1f}%</b></div>
            <div style="margin-top:8px;"><span class="badge badge-green">{escape(str(row.get("ia_badge", "")))}</span></div>
        </div>
        """

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def plot_line(data, x, y, title):
    if data.empty or y not in data.columns:
        st.info("Pas assez de données pour cette courbe.")
        return

    chart = data.copy()
    chart[y] = pd.to_numeric(chart[y], errors="coerce")

    if PLOTLY_OK:
        fig = px.line(chart, x=x, y=y, title=title, markers=True)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f7f8ff"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(chart.set_index(x)[y])


def plot_bar(data, x, y, title):
    if data.empty or y not in data.columns:
        st.info("Pas assez de données.")
        return

    chart = data.copy()
    chart[y] = pd.to_numeric(chart[y], errors="coerce")

    if PLOTLY_OK:
        fig = px.bar(chart, x=x, y=y, title=title)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f7f8ff"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(chart.set_index(x)[y])


df["category"] = df["sport"].apply(category_of_sport)
df["value"] = to_num(df, "value")
df["ai_probability"] = to_num(df, "ai_probability")
df["suggested_stake"] = to_num(df, "suggested_stake")


# =========================
# SIDEBAR FILTERS
# =========================

st.sidebar.title("Filtres")

sports = st.sidebar.multiselect(
    "Compétitions",
    options=sorted(df["sport"].dropna().unique()),
    default=sorted(df["sport"].dropna().unique())
)

categories = st.sidebar.multiselect(
    "Catégorie",
    options=sorted(df["category"].dropna().unique()),
    default=sorted(df["category"].dropna().unique())
)

markets = st.sidebar.multiselect(
    "Marchés",
    options=sorted(df["market"].dropna().unique()),
    default=sorted(df["market"].dropna().unique())
)

only_value = st.sidebar.checkbox("Seulement VALUE BETS")
search = st.sidebar.text_input("Recherche équipe / joueur")

min_value = st.sidebar.slider("Value minimum", 0.0, 1.5, 0.0, 0.01)
min_prob = st.sidebar.slider("Probabilité IA minimum", 0.0, 1.0, 0.0, 0.01)

filtered = df[
    df["sport"].isin(sports)
    & df["category"].isin(categories)
    & df["market"].isin(markets)
    & (df["value"] >= min_value)
    & (df["ai_probability"] >= min_prob)
].copy()

if only_value:
    filtered = filtered[filtered["decision"] == "VALUE BET"]

if search:
    s = search.lower()
    filtered = filtered[
        filtered.astype(str).apply(lambda r: s in " ".join(r.values).lower(), axis=1)
    ]

football_df = filtered[filtered["category"] == "football"].copy()
tennis_df = filtered[filtered["category"] == "tennis"].copy()
value_bets = filtered[filtered["decision"] == "VALUE BET"].copy()


# =========================
# HEADER
# =========================

last_update = df["last_update"].iloc[0] if "last_update" in df.columns and not df.empty else "Inconnue"
telegram_count = 0

if TELEGRAM_SENT_PATH.exists():
    try:
        telegram_count = len(pd.read_csv(TELEGRAM_SENT_PATH))
    except Exception:
        telegram_count = 0

st.markdown('<div class="shell">', unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
    <div class="hero-card">
        <div class="eyebrow">Plateforme privée</div>
        <div class="hero-title">IA Paris Sportifs Ultime</div>
        <div class="hero-copy">
            Dashboard complet : football, tennis, value betting, Telegram, tracking ROI, rentabilité, courbes et historique.
        </div>
    </div>
    <div class="status-card">
        <div class="status-line"><span class="status-label">Dernière actualisation</span><span class="status-value">{escape(str(last_update))}</span></div>
        <div class="status-line"><span class="status-label">Alertes Telegram mémorisées</span><span class="status-value">{telegram_count}</span></div>
        <div class="status-line"><span class="status-label">Moteurs actifs</span><span class="status-value">Football + Tennis</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

best_value = filtered["value"].max() if not filtered.empty else 0
max_stake = filtered["suggested_stake"].max() if not filtered.empty else 0

st.markdown(
    '<div class="kpi-grid">'
    + kpi("Lignes analysées", len(filtered), "après filtres")
    + kpi("Value Bets", len(value_bets), "paris détectés")
    + kpi("Meilleure value", f"{best_value*100:.2f}%", "edge maximal")
    + kpi("Mise max", f"{max_stake:.2f}€", "stake IA conseillé")
    + "</div>",
    unsafe_allow_html=True
)


# =========================
# TABS
# =========================

tabs = st.tabs([
    "Vue globale",
    "Football",
    "Tennis",
    "Value Bets",
    "Résultats / ROI",
    "Tech IA",
    "Toutes les prédictions"
])


with tabs[0]:
    st.subheader("Vue globale")

    render_match_cards(value_bets.sort_values("value", ascending=False), limit=6)

    c1, c2 = st.columns(2)

    with c1:
        chart = filtered.sort_values("value", ascending=False).head(30)
        plot_bar(chart, "home_team", "value", "Top value par match")

    with c2:
        chart = filtered.sort_values("ai_probability", ascending=False).head(30)
        plot_bar(chart, "home_team", "ai_probability", "Top probabilités IA")

    st.subheader("Répartition des décisions")
    if PLOTLY_OK:
        decision_counts = filtered["decision"].value_counts().reset_index()
        decision_counts.columns = ["decision", "count"]
        fig = px.pie(decision_counts, names="decision", values="count", hole=.45)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#f7f8ff")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(filtered["decision"].value_counts())


with tabs[1]:
    st.subheader("Football")

    render_match_cards(football_df[football_df["decision"] == "VALUE BET"].sort_values("value", ascending=False), limit=6)

    c1, c2 = st.columns(2)

    with c1:
        plot_bar(football_df.sort_values("value", ascending=False).head(25), "home_team", "value", "Football - value")

    with c2:
        if "over_25" in football_df.columns:
            plot_bar(football_df.sort_values("over_25", ascending=False).head(25), "home_team", "over_25", "Football - Over 2.5")

    st.subheader("Tableau football")
    render_table(football_df.sort_values("value", ascending=False), height=560)


with tabs[2]:
    st.subheader("Tennis")

    render_match_cards(tennis_df[tennis_df["decision"] == "VALUE BET"].sort_values("value", ascending=False), limit=6)

    c1, c2 = st.columns(2)

    with c1:
        plot_bar(tennis_df.sort_values("value", ascending=False).head(25), "home_team", "value", "Tennis - value")

    with c2:
        if "tennis_engine_score" in tennis_df.columns:
            plot_bar(tennis_df.sort_values("tennis_engine_score", ascending=False).head(25), "home_team", "tennis_engine_score", "Tennis - score IA")

    st.subheader("Tableau tennis")
    render_table(tennis_df.sort_values("value", ascending=False), height=560)


with tabs[3]:
    st.subheader("Value Bets")

    render_match_cards(value_bets.sort_values("value", ascending=False), limit=9)

    st.subheader("Tableau value bets")
    render_table(value_bets.sort_values("value", ascending=False), height=620)


with tabs[4]:
    st.subheader("Résultats IA / ROI / Rentabilité")

    if tracking.empty:
        st.warning("Aucun tracking disponible.")
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

        total_staked = finished["stake"].sum() if len(finished) else 0
        profit = finished["profit"].sum() if len(finished) else 0
        roi = profit / total_staked if total_staked > 0 else 0
        win_rate = len(wins) / len(finished) if len(finished) else 0

        st.markdown(
            '<div class="kpi-grid">'
            + kpi("Paris terminés", len(finished), "WIN + LOSS")
            + kpi("Gagnés", len(wins), "paris validés")
            + kpi("Perdus", len(losses), "paris perdus")
            + kpi("ROI", f"{roi*100:.2f}%", f"profit {profit:.2f}€")
            + "</div>",
            unsafe_allow_html=True
        )

        if profit > 0:
            st.success(f"L’IA est en gain : +{profit:.2f}€")
        elif profit < 0:
            st.error(f"L’IA est en perte : {profit:.2f}€")
        else:
            st.info("L’IA est à l’équilibre.")

        if not finished.empty:
            finished = finished.sort_values("date").copy()
            finished["cumulative_profit"] = finished["profit"].cumsum()
            finished["bet_number"] = range(1, len(finished) + 1)

            c1, c2 = st.columns(2)

            with c1:
                plot_line(finished, "bet_number", "cumulative_profit", "Courbe de rentabilité")

            with c2:
                plot_bar(finished, "bet_number", "profit", "Gain / perte par pari")

            if "sport" in finished.columns:
                sport_profit = finished.groupby("sport")["profit"].sum().reset_index()
                plot_bar(sport_profit, "sport", "profit", "Profit par compétition")

            if "market" in finished.columns:
                market_profit = finished.groupby("market")["profit"].sum().reset_index()
                plot_bar(market_profit, "market", "profit", "Profit par marché")

        st.subheader("Paris gagnés")
        render_table(wins.sort_values("profit", ascending=False), height=360)

        st.subheader("Paris perdus")
        render_table(losses.sort_values("profit", ascending=True), height=360)

        st.subheader("Paris en attente")
        render_table(pending.sort_values("date", ascending=False), height=360)


with tabs[5]:
    st.subheader("Tech IA")

    st.markdown("""
    <div class="panel-card" style="padding:18px;">
        <b>Moteur Football</b><br>
        xG, Poisson, ELO équipes, calibration, value betting, score exact, BTTS, Over/Under, buteurs probables.
        <br><br>
        <b>Moteur Tennis</b><br>
        ELO joueurs, forme récente, winrate, probabilité marché, edge/value, score IA tennis, marchés Player 1 / Player 2.
        <br><br>
        <b>Automatisation</b><br>
        GitHub Actions, mise à jour données, tracking, anti-spam Telegram, dashboard Streamlit.
    </div>
    """, unsafe_allow_html=True)

    tech_cols = [
        "category", "sport", "market", "ai_probability", "bookmaker_odds",
        "value", "confidence", "ia_badge", "tennis_engine_score", "tennis_edge"
    ]
    tech_cols = [c for c in tech_cols if c in filtered.columns]

    render_table(filtered[tech_cols].sort_values("value", ascending=False), height=560)


with tabs[6]:
    st.subheader("Toutes les prédictions")
    render_table(filtered.sort_values("value", ascending=False), height=720)

st.markdown("</div>", unsafe_allow_html=True)