import ast
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

st.set_page_config(
    page_title="IA Paris Sportifs Ultime",
    layout="wide",
    page_icon="PS",
)

APP_PASSWORD = "29052007"
PRED_PATH = Path("data/predictions/predictions_today.csv")
TRACK_PATH = Path("tracking_results.csv")
TELEGRAM_SENT_PATH = Path("data/telegram_sent.csv")

# ============================================================
# STYLE / DESIGN
# ============================================================

st.markdown(
    """
<style>
@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes floatCard {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
    100% { transform: translateY(0px); }
}

.stApp {
    color: #f8fafc;
    background:
        linear-gradient(rgba(8, 12, 24, .88), rgba(8, 12, 24, .94)),
        url("https://images.unsplash.com/photo-1540747913346-19e32dc3e97e?auto=format&fit=crop&w=2400&q=80");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
}

.stApp:before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background: linear-gradient(120deg, rgba(236,72,153,.20), rgba(59,130,246,.20), rgba(16,185,129,.16));
    background-size: 300% 300%;
    animation: gradientMove 12s ease infinite;
    z-index: 0;
}

.block-container {
    max-width: 1540px;
    padding-top: 26px;
    padding-bottom: 42px;
    position: relative;
    z-index: 1;
}

section[data-testid="stSidebar"] {
    background: rgba(12, 18, 34, .88);
    border-right: 1px solid rgba(255,255,255,.12);
    backdrop-filter: blur(16px);
}

h1, h2, h3, h4 { color: #f8fafc !important; }

[data-testid="stMetric"] {
    background: rgba(15, 23, 42, .78);
    border: 1px solid rgba(255,255,255,.14);
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 18px 44px rgba(0,0,0,.30);
    backdrop-filter: blur(14px);
}

.hero-box {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 8% 0%, rgba(244,114,182,.30), transparent 34%),
        radial-gradient(circle at 95% 20%, rgba(34,211,238,.24), transparent 36%),
        rgba(15, 23, 42, .76);
    border: 1px solid rgba(255,255,255,.14);
    border-radius: 30px;
    padding: 30px;
    margin-bottom: 22px;
    box-shadow: 0 26px 70px rgba(0,0,0,.38);
    backdrop-filter: blur(18px);
}

.hero-title {
    font-size: 42px;
    line-height: 1.04;
    font-weight: 950;
    margin-bottom: 10px;
}

.hero-sub {
    color: #cbd5e1;
    font-size: 15px;
    line-height: 1.6;
}

.pro-card {
    background: rgba(15, 23, 42, .78);
    border: 1px solid rgba(255,255,255,.13);
    border-radius: 22px;
    padding: 18px;
    margin-bottom: 15px;
    box-shadow: 0 16px 42px rgba(0,0,0,.28);
    backdrop-filter: blur(14px);
    animation: floatCard 7s ease-in-out infinite;
}

.pick-card {
    border-left: 5px solid #22c55e;
}

.risky-card {
    border-left: 5px solid #fb7185;
}

.watch-card {
    border-left: 5px solid #f59e0b;
}

.badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    font-weight: 900;
    font-size: 12px;
    margin-right: 6px;
}

.badge-green { background: rgba(34,197,94,.18); color: #86efac; }
.badge-yellow { background: rgba(251,191,36,.18); color: #fde68a; }
.badge-red { background: rgba(251,113,133,.18); color: #fecdd3; }
.badge-blue { background: rgba(96,165,250,.18); color: #bfdbfe; }
.badge-purple { background: rgba(168,85,247,.20); color: #e9d5ff; }

.small-muted { color:#94a3b8; font-size: 13px; }
.big-number { font-size: 28px; font-weight: 950; }

.stDataFrame {
    border-radius: 18px !important;
    overflow: hidden !important;
}

hr { border-color: rgba(255,255,255,.12); }

@media (max-width: 900px) {
    .block-container { padding: 16px 10px 32px; }
    .hero-title { font-size: 28px; }
    .hero-box { padding: 18px; border-radius: 20px; }
    [data-testid="stMetric"] { padding: 12px; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# AUTH
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">IA Paris Sportifs Ultime</div>
            <div class="hero-sub">Accès privé • Football • Tennis • Value Betting • ROI • Tracking</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    password = st.text_input("Mot de passe", type="password")

    if password == APP_PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
    elif password:
        st.error("Mot de passe incorrect.")

    st.stop()

# ============================================================
# DATA
# ============================================================

@st.cache_data(ttl=300)
def load_csv(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()


df = load_csv(PRED_PATH)
tracking = load_csv(TRACK_PATH)

if df.empty:
    st.error("Aucune prédiction trouvée. Lance d'abord le pipeline.")
    st.stop()


def num_col(data, col, default=0):
    if col in data.columns:
        return pd.to_numeric(data[col], errors="coerce").fillna(default)
    return pd.Series([default] * len(data), index=data.index)


def sport_category(sport):
    s = str(sport).lower()
    if "tennis" in s:
        return "tennis"
    if "soccer" in s or "football" in s:
        return "football"
    return "autre"


def prepare_data(data):
    data = data.copy()

    if "sport" not in data.columns:
        data["sport"] = ""

    data["category"] = data["sport"].apply(sport_category)

    for col in [
        "value",
        "ai_probability",
        "suggested_stake",
        "bookmaker_odds",
        "implied_probability",
        "safety_score",
        "score_exact_1_proba",
        "score_exact_2_proba",
        "score_exact_3_proba",
        "tennis_engine_score",
        "tennis_edge",
    ]:
        data[col] = num_col(data, col)

    if "bet_mode" not in data.columns:
        data["bet_mode"] = "NON CLASSÉ"

    if "decision" not in data.columns:
        data["decision"] = "NO BET"

    return data


df = prepare_data(df)

# ============================================================
# HELPERS AFFICHAGE
# ============================================================

RECOMMENDED_MODES = ["MEGA VALUE", "SAFE PICK", "VALUE BET", "RISKY VALUE"]


def sort_recommendations(data):
    if data.empty:
        return data

    rank = {
        "MEGA VALUE": 0,
        "SAFE PICK": 1,
        "VALUE BET": 2,
        "RISKY VALUE": 3,
        "WATCHLIST": 4,
        "NO BET": 5,
    }

    out = data.copy()
    out["_rank"] = out["bet_mode"].map(rank).fillna(9)
    out = out.sort_values(
        ["_rank", "safety_score", "value", "ai_probability"],
        ascending=[True, False, False, False],
    )
    return out.drop(columns=["_rank"])


def format_pct(x):
    try:
        if pd.isna(x):
            return ""
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return ""


def clean_table(data, compact=True):
    core_cols = [
        "date",
        "sport",
        "category",
        "home_team",
        "away_team",
        "market",
        "selection",
        "bet_mode",
        "safety_level",
        "safety_score",
        "ai_probability",
        "bookmaker_odds",
        "value",
        "suggested_stake",
        "score_exact_1",
        "score_exact_1_proba",
        "score_exact_2",
        "score_exact_2_proba",
        "score_exact_3",
        "score_exact_3_proba",
        "tennis_engine_score",
        "tennis_edge",
        "football_trap_signal",
        "confidence",
        "ia_badge",
        "decision",
    ]

    tracking_cols = [
        "result",
        "stake",
        "profit",
        "final_winner",
        "status_detail",
    ]

    cols = core_cols + ([] if compact else tracking_cols)
    cols = [c for c in cols if c in data.columns]
    out = data[cols].copy()

    for col in ["ai_probability", "implied_probability", "value", "tennis_edge"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").apply(format_pct)

    for col in [
        "bookmaker_odds",
        "suggested_stake",
        "stake",
        "profit",
        "safety_score",
        "tennis_engine_score",
        "score_exact_1_proba",
        "score_exact_2_proba",
        "score_exact_3_proba",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    return out


def show_table(data, height=520, compact=True):
    if data.empty:
        st.info("Aucune donnée à afficher.")
        return

    st.dataframe(
        clean_table(data, compact=compact),
        use_container_width=True,
        hide_index=True,
        height=height,
    )


def plot_bar(data, x, y, title):
    if data.empty or x not in data.columns or y not in data.columns:
        st.info("Pas assez de données.")
        return

    chart = data.copy()
    chart[y] = pd.to_numeric(chart[y], errors="coerce").fillna(0)

    if PLOTLY_OK:
        fig = px.bar(chart, x=x, y=y, title=title)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8fafc",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(chart.set_index(x)[y])


def plot_line(data, x, y, title):
    if data.empty or x not in data.columns or y not in data.columns:
        st.info("Pas assez de données.")
        return

    chart = data.copy()
    chart[y] = pd.to_numeric(chart[y], errors="coerce").fillna(0)

    if PLOTLY_OK:
        fig = px.line(chart, x=x, y=y, title=title, markers=True)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8fafc",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(chart.set_index(x)[y])


def mode_class(mode):
    m = str(mode).upper()
    if "RISKY" in m:
        return "risky-card", "badge-red"
    if "WATCH" in m:
        return "watch-card", "badge-yellow"
    return "pick-card", "badge-green"


def render_cards(data, limit=6):
    if data.empty:
        st.info("Aucun pari à afficher.")
        return

    cols = st.columns(3)

    for i, (_, row) in enumerate(data.head(limit).iterrows()):
        with cols[i % 3]:
            value = float(row.get("value", 0) or 0) * 100
            proba = float(row.get("ai_probability", 0) or 0) * 100
            stake = float(row.get("suggested_stake", 0) or 0)
            odds = row.get("bookmaker_odds", "")
            mode = row.get("bet_mode", "")
            card_class, badge_class = mode_class(mode)

            st.markdown(
                f"""
                <div class="pro-card {card_class}">
                    <span class="badge {badge_class}">{mode}</span>
                    <span class="badge badge-blue">{row.get("market", "")}</span><br><br>
                    <b>{row.get("home_team", "")} vs {row.get("away_team", "")}</b><br>
                    <span class="small-muted">{row.get("sport", "")}</span><br><br>
                    Proba IA : <b>{proba:.1f}%</b><br>
                    Cote : <b>{odds}</b><br>
                    Value : <b>{value:.1f}%</b><br>
                    Mise conseillée : <b>{stake:.2f}€</b><br>
                    Score / Set probable : <b>{row.get("score_exact_1", "")}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )


def parse_top_scores(value):
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        result = []
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                result.append((str(item[0]), float(item[1]) * 100))
        return result[:5]
    except Exception:
        return []


def football_score_table(data):
    rows = []
    for _, row in data.iterrows():
        top_scores = parse_top_scores(row.get("top_scores", ""))

        if not top_scores:
            for i in [1, 2, 3]:
                score = row.get(f"score_exact_{i}", "")
                proba = row.get(f"score_exact_{i}_proba", "")
                if str(score).strip():
                    top_scores.append((score, float(proba or 0)))

        best = top_scores[0] if top_scores else ("", 0)
        second = top_scores[1] if len(top_scores) > 1 else ("", 0)
        third = top_scores[2] if len(top_scores) > 2 else ("", 0)

        rows.append({
            "match": f"{row.get('home_team', '')} vs {row.get('away_team', '')}",
            "marché": row.get("market", ""),
            "mode": row.get("bet_mode", ""),
            "score le + probable": best[0],
            "proba score 1": round(best[1], 2),
            "alternative 1": second[0],
            "proba score 2": round(second[1], 2),
            "alternative 2": third[0],
            "proba score 3": round(third[1], 2),
            "draw signal": row.get("draw_hunter", ""),
            "piège bookmaker": row.get("football_trap_signal", ""),
            "mise": row.get("suggested_stake", 0),
        })

    return pd.DataFrame(rows)


def tennis_sets_table(data):
    rows = []
    for _, row in data.iterrows():
        p = float(row.get("ai_probability", 0) or 0)
        score1 = str(row.get("score_exact_1", "2-0") or "2-0")
        score2 = str(row.get("score_exact_2", "2-1") or "2-1")
        proba1 = float(row.get("score_exact_1_proba", 0) or 0)
        proba2 = float(row.get("score_exact_2_proba", 0) or 0)

        # Si ton pipeline fournit déjà les probas, on les garde.
        # Sinon, on estime simplement selon la probabilité IA.
        if proba1 <= 0 and proba2 <= 0:
            if p >= 0.68:
                proba1 = round(p * 58, 2)
                proba2 = round(p * 42, 2)
            else:
                proba1 = round(p * 48, 2)
                proba2 = round(p * 52, 2)

        likely_sets = score1 if proba1 >= proba2 else score2
        likely_sets_proba = max(proba1, proba2)

        rows.append({
            "match": f"{row.get('home_team', '')} vs {row.get('away_team', '')}",
            "sélection IA": row.get("selection", ""),
            "mode": row.get("bet_mode", ""),
            "nombre de sets probable": likely_sets,
            "proba set probable": round(likely_sets_proba, 2),
            "option 2-0": score1,
            "proba 2-0": round(proba1, 2),
            "option 2-1": score2,
            "proba 2-1": round(proba2, 2),
            "score IA tennis": row.get("tennis_engine_score", 0),
            "edge tennis": row.get("tennis_edge", 0),
            "mise": row.get("suggested_stake", 0),
        })

    return pd.DataFrame(rows)

# ============================================================
# FILTERS
# ============================================================

st.sidebar.title("Filtres")

sports = st.sidebar.multiselect(
    "Compétitions",
    sorted(df["sport"].dropna().unique()),
    default=sorted(df["sport"].dropna().unique()),
)

categories = st.sidebar.multiselect(
    "Sport",
    sorted(df["category"].dropna().unique()),
    default=sorted(df["category"].dropna().unique()),
)

modes = st.sidebar.multiselect(
    "Mode IA",
    sorted(df["bet_mode"].dropna().unique()),
    default=sorted(df["bet_mode"].dropna().unique()),
)

only_recommended = st.sidebar.checkbox("Seulement les paris conseillés", value=False)
search = st.sidebar.text_input("Recherche équipe / joueur")
min_stake = st.sidebar.slider("Mise minimum", 0.0, 10.0, 0.0, 0.1)
min_prob = st.sidebar.slider("Probabilité IA minimum", 0.0, 1.0, 0.0, 0.01)

filtered = df[
    df["sport"].isin(sports)
    & df["category"].isin(categories)
    & df["bet_mode"].isin(modes)
    & (df["suggested_stake"] >= min_stake)
    & (df["ai_probability"] >= min_prob)
].copy()

if only_recommended:
    filtered = filtered[filtered["bet_mode"].isin(RECOMMENDED_MODES)]

if search:
    s = search.lower()
    filtered = filtered[
        filtered.astype(str).apply(lambda r: s in " ".join(r.values).lower(), axis=1)
    ]

football_df = filtered[filtered["category"] == "football"].copy()
tennis_df = filtered[filtered["category"] == "tennis"].copy()
recommended = filtered[
    filtered["bet_mode"].isin(RECOMMENDED_MODES)
    & (filtered["suggested_stake"] > 0)
].copy()
watchlist = filtered[filtered["bet_mode"].astype(str).str.upper().str.contains("WATCH")].copy()

last_update = df["last_update"].iloc[0] if "last_update" in df.columns else "Inconnue"
telegram_count = 0
if TELEGRAM_SENT_PATH.exists():
    try:
        telegram_count = len(pd.read_csv(TELEGRAM_SENT_PATH))
    except Exception:
        telegram_count = 0

# ============================================================
# HERO / KPI
# ============================================================

st.markdown(
    f"""
    <div class="hero-box">
        <div class="hero-title">IA Paris Sportifs Ultime</div>
        <div class="hero-sub">
            Dashboard propre • Tennis sets probables • Score exact football • Mises conseillées • ROI<br>
            Dernière actualisation : <b>{last_update}</b> | Alertes Telegram : <b>{telegram_count}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Lignes analysées", len(filtered))
c2.metric("Paris conseillés", len(recommended))
c3.metric("Football", len(football_df))
c4.metric("Tennis", len(tennis_df))
c5.metric("Mise max", f"{filtered['suggested_stake'].max() if not filtered.empty else 0:.2f}€")

# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "Accueil",
    "Paris conseillés",
    "Football score exact",
    "Tennis sets",
    "Mises / bankroll",
    "Résultats / ROI",
    "Toutes les lignes",
])

with tabs[0]:
    st.subheader("Meilleurs spots du jour")
    render_cards(sort_recommendations(recommended), limit=6)

    a, b = st.columns(2)
    with a:
        if not recommended.empty:
            mode_counts = recommended["bet_mode"].value_counts().reset_index()
            mode_counts.columns = ["mode", "nombre"]
            plot_bar(mode_counts, "mode", "nombre", "Répartition des paris conseillés")
    with b:
        plot_bar(sort_recommendations(filtered).head(20), "home_team", "safety_score", "Top sécurité IA")

    st.subheader("Watchlist intelligente")
    show_table(sort_recommendations(watchlist).head(30), height=360)

with tabs[1]:
    st.subheader("Paris conseillés avec mise")
    render_cards(sort_recommendations(recommended), limit=9)
    show_table(sort_recommendations(recommended), height=620)

with tabs[2]:
    st.subheader("Football — score exact plus lisible")
    st.caption("Le score exact reste un marché très difficile. Ici, le dashboard affiche les 3 scores les plus probables + signal piège bookmaker.")

    football_reco = sort_recommendations(football_df)
    score_df = football_score_table(football_reco)

    if score_df.empty:
        st.info("Aucun match football disponible dans les données actuelles.")
    else:
        st.dataframe(score_df, use_container_width=True, hide_index=True, height=520)

        a, b = st.columns(2)
        with a:
            plot_bar(score_df.head(20), "match", "proba score 1", "Probabilité du score exact principal")
        with b:
            if "suggested_stake" in football_reco.columns:
                plot_bar(football_reco.head(20), "home_team", "suggested_stake", "Mises conseillées football")

with tabs[3]:
    st.subheader("Tennis — nombre de sets probable")
    st.caption("Affiche le score en sets le plus probable : 2-0 ou 2-1 selon les probabilités du pipeline.")

    tennis_reco = sort_recommendations(tennis_df)
    sets_df = tennis_sets_table(tennis_reco)

    if sets_df.empty:
        st.info("Aucun match tennis disponible.")
    else:
        st.dataframe(sets_df, use_container_width=True, hide_index=True, height=620)

        a, b = st.columns(2)
        with a:
            plot_bar(sets_df.head(25), "match", "proba set probable", "Probabilité du nombre de sets")
        with b:
            plot_bar(tennis_reco.head(25), "home_team", "tennis_engine_score", "Score IA tennis")

with tabs[4]:
    st.subheader("Mises conseillées / bankroll")

    stake_cols = [
        "date",
        "sport",
        "home_team",
        "away_team",
        "selection",
        "bet_mode",
        "ai_probability",
        "bookmaker_odds",
        "value",
        "suggested_stake",
        "stake_percent",
        "kelly_fraction",
        "bankroll",
        "safety_score",
    ]
    stake_cols = [c for c in stake_cols if c in recommended.columns]

    if recommended.empty:
        st.info("Aucune mise conseillée pour le moment.")
    else:
        total_stake = recommended["suggested_stake"].sum()
        avg_stake = recommended["suggested_stake"].mean()
        m1, m2, m3 = st.columns(3)
        m1.metric("Mise totale conseillée", f"{total_stake:.2f}€")
        m2.metric("Mise moyenne", f"{avg_stake:.2f}€")
        m3.metric("Nombre de tickets", len(recommended))
        st.dataframe(clean_table(recommended[stake_cols]), use_container_width=True, hide_index=True, height=540)

with tabs[5]:
    st.subheader("Résultats IA / ROI")

    if tracking.empty:
        st.warning("Aucun tracking disponible.")
    else:
        tr = tracking.copy()
        if "result" not in tr.columns:
            tr["result"] = "PENDING"

        tr["result"] = tr["result"].fillna("PENDING").astype(str).str.upper().replace({"": "PENDING", "NAN": "PENDING"})
        tr["stake"] = pd.to_numeric(tr.get("stake", tr.get("suggested_stake", 0)), errors="coerce").fillna(0)
        tr["profit"] = pd.to_numeric(tr.get("profit", 0), errors="coerce").fillna(0)
        tr["bookmaker_odds"] = pd.to_numeric(tr.get("bookmaker_odds", 0), errors="coerce").fillna(0)
        if "category" not in tr.columns:
            tr["category"] = tr["sport"].apply(sport_category)

        finished = tr[tr["result"].isin(["WIN", "LOSS"])].copy()
        pending = tr[tr["result"] == "PENDING"].copy()
        wins = tr[tr["result"] == "WIN"].copy()
        losses = tr[tr["result"] == "LOSS"].copy()

        total_staked = finished["stake"].sum() if not finished.empty else 0
        total_profit = finished["profit"].sum() if not finished.empty else 0
        roi = total_profit / total_staked if total_staked > 0 else 0
        winrate = len(wins) / len(finished) if len(finished) else 0

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Terminés", len(finished))
        r2.metric("En attente", len(pending))
        r3.metric("ROI", f"{roi * 100:.2f}%")
        r4.metric("Winrate", f"{winrate * 100:.2f}%")

        r5, r6, r7 = st.columns(3)
        r5.metric("Misé total", f"{total_staked:.2f}€")
        r6.metric("Profit net", f"{total_profit:.2f}€")
        r7.metric("Cote moyenne", f"{finished['bookmaker_odds'].mean() if not finished.empty else 0:.2f}")

        if not finished.empty:
            chart = finished.sort_values("date").copy()
            chart["cumulative_profit"] = chart["profit"].cumsum()
            chart["bet_number"] = range(1, len(chart) + 1)
            plot_line(chart, "bet_number", "cumulative_profit", "Courbe profit cumulé")

        result_tabs = st.tabs(["Gagnés", "Perdus", "En attente", "Historique"])
        with result_tabs[0]:
            show_table(wins.sort_values("profit", ascending=False), height=460, compact=False)
        with result_tabs[1]:
            show_table(losses.sort_values("profit", ascending=True), height=460, compact=False)
        with result_tabs[2]:
            show_table(pending.sort_values("date"), height=460, compact=False)
        with result_tabs[3]:
            show_table(tr.sort_values("date", ascending=False), height=620, compact=False)

with tabs[6]:
    st.subheader("Toutes les lignes analysées")
    show_table(sort_recommendations(filtered), height=720)
