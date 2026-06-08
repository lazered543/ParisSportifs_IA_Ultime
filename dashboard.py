import ast
import re
import unicodedata
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
UPCOMING_PATH = Path("data/processed/upcoming_odds.csv")
TRACK_PATH = Path("tracking_results.csv")
TELEGRAM_SENT_PATH = Path("data/telegram_sent.csv")
ARCHIVE_PATH = Path("data/archive/finished_bets_archive.csv")
BANKROLL_STATE_PATH = Path("data/bankroll_state.csv")
BACKTEST_SUMMARY_PATH = Path("data/learning/backtest_summary.csv")
CALIBRATION_PATH = Path("data/learning/probability_calibration.csv")
THRESHOLD_PROFILE_PATH = Path("data/learning/threshold_optimizer.csv")
LOCAL_TZ = "Europe/Paris"

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
upcoming_odds = load_csv(UPCOMING_PATH)
tracking = load_csv(TRACK_PATH)
archive = load_csv(ARCHIVE_PATH)
bankroll_state = load_csv(BANKROLL_STATE_PATH)
backtest_summary = load_csv(BACKTEST_SUMMARY_PATH)
calibration = load_csv(CALIBRATION_PATH)
threshold_profiles = load_csv(THRESHOLD_PROFILE_PATH)

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

    data["potential_profit"] = (
        data["suggested_stake"] * (data["bookmaker_odds"] - 1)
    ).clip(lower=0).round(2)
    data["potential_return"] = (
        data["suggested_stake"] * data["bookmaker_odds"]
    ).clip(lower=0).round(2)
    data["benefice_avec_mise"] = data["potential_return"].round(2)

    return data


df = prepare_data(df)

if not bankroll_state.empty and "current_bankroll" in bankroll_state.columns:
    current_bankroll_display = float(pd.to_numeric(bankroll_state["current_bankroll"], errors="coerce").fillna(10.0).iloc[0])
    initial_bankroll_display = float(pd.to_numeric(bankroll_state.get("initial_bankroll", pd.Series([10.0])), errors="coerce").fillna(10.0).iloc[0])
else:
    current_bankroll_display = 10.0
    initial_bankroll_display = 10.0

profit_net_display = current_bankroll_display - initial_bankroll_display
roi_display = (profit_net_display / initial_bankroll_display * 100) if initial_bankroll_display > 0 else 0.0

# Objectifs de progression visibles dans le dashboard
GOALS = [20, 50, 100, 500, 1000]
next_goal_display = next((g for g in GOALS if current_bankroll_display < g), GOALS[-1])
previous_goal_display = initial_bankroll_display
for goal in GOALS:
    if current_bankroll_display >= goal:
        previous_goal_display = goal
progress_display = 100.0
if next_goal_display > previous_goal_display:
    progress_display = ((current_bankroll_display - previous_goal_display) / (next_goal_display - previous_goal_display)) * 100
progress_display = max(0.0, min(100.0, progress_display))

max_single_bet_display = min(current_bankroll_display * 0.50, 5.00)
max_daily_exposure_display = current_bankroll_display * 0.50


# ============================================================
# HELPERS AFFICHAGE
# ============================================================

SAFE_MODES = ["MEGA VALUE", "SAFE PICK", "FAVORI SOLIDE", "VALUE BET", "NUL POSSIBLE"]
RISKY_MODES = ["RISKY VALUE"]
RECOMMENDED_MODES = SAFE_MODES


def sort_recommendations(data):
    if data.empty:
        return data

    rank = {
        "MEGA VALUE": 0,
        "SAFE PICK": 1,
        "FAVORI SOLIDE": 2,
        "VALUE BET": 3,
        "NUL POSSIBLE": 4,
        "RISKY VALUE": 5,
        "WATCHLIST": 6,
        "NO BET": 7,
    }

    out = data.copy()
    out["_rank"] = out["bet_mode"].map(rank).fillna(9)
    if "priority" not in out.columns:
        out["priority"] = 0

    out = out.sort_values(
        ["priority", "_rank", "safety_score", "value", "ai_probability"],
        ascending=[False, True, False, False, False],
    )
    return out.drop(columns=["_rank"])


def format_pct(x):
    try:
        if pd.isna(x):
            return ""
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return ""

def parse_display_datetime(value):
    try:
        return pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def display_today():
    return pd.Timestamp.now(tz=LOCAL_TZ).date()


def today_only(data):
    if data.empty or "date" not in data.columns: return data
    out = data.copy(); out["_dt"] = out["date"].apply(parse_display_datetime)
    out["_dt"] = pd.to_datetime(out["_dt"], utc=True, errors="coerce")
    out["_local_day"] = out["_dt"].dt.tz_convert(LOCAL_TZ).dt.date
    valid = out["_dt"].notna()
    today_mask = valid & (out["_local_day"] == display_today())
    if today_mask.any():
        out = out[today_mask].copy()
    else:
        now = pd.Timestamp.now(tz="UTC")
        future = valid & (out["_dt"] >= now - pd.Timedelta(hours=2))
        if future.any():
            next_day = out.loc[future, "_local_day"].min()
            out = out[future & (out["_local_day"] == next_day)].copy()
        else:
            out = out.iloc[0:0].copy()
    return out.drop(columns=["_dt", "_local_day"], errors="ignore")


def odds_freshness_message(data):
    if data.empty:
        return "warning", "Aucune cote en memoire. Relance la mise a jour des donnees avant de chercher des matchs."
    if "commence_time" not in data.columns:
        return "warning", "Le fichier de cotes ne contient pas les horaires des matchs. Les donnees sont incompletes."

    out = data.copy()
    out["_dt"] = pd.to_datetime(out["commence_time"], utc=True, errors="coerce")
    out["_source"] = out.get("odds_source", pd.Series("", index=out.index)).astype(str).str.lower()
    out["_sport"] = out.get("sport", pd.Series("", index=out.index)).astype(str).str.lower()

    now = pd.Timestamp.now(tz="UTC")
    fresh = out[out["_dt"].isna() | (out["_dt"] >= now - pd.Timedelta(hours=4))].copy()
    live_api = fresh[fresh["_source"].eq("the-odds-api")]
    live_tennis = live_api[live_api["_sport"].str.contains("tennis")]

    def fmt_dt(value):
        if pd.isna(value):
            return "inconnue"
        try:
            return pd.Timestamp(value).tz_convert("Europe/Paris").strftime("%d/%m %H:%M")
        except Exception:
            return str(value)

    if fresh.empty:
        latest = out["_dt"].max()
        return "warning", f"Aucune cote future dans les donnees actuelles. Derniere cote trouvee : {fmt_dt(latest)}."

    if live_api.empty:
        next_match = fresh["_dt"].min()
        return "warning", (
            "Aucune cote API fraiche dans les donnees actuelles. "
            f"Prochaine ligne locale : {fmt_dt(next_match)}. "
            "Les lignes fallback ne doivent pas etre jouees en argent reel."
        )

    if live_tennis.empty:
        next_match = fresh["_dt"].min()
        return "info", (
            "Aucune cote tennis fraiche dans le flux API actuel. "
            f"Prochaine cote sportive disponible : {fmt_dt(next_match)}."
        )

    return "", ""


def match_display_label(row):
    sport = str(row.get("category", "")).lower()
    if sport == "tennis":
        return f"{row.get('home_team', '')} vs {row.get('away_team', '')}"
    return f"{row.get('home_team', '')} vs {row.get('away_team', '')}"

  
def clean_table(data, compact=True):
    core_cols = [
        "date",
        "sport",
        "odds_source",
        "category",
        "home_team",
        "away_team",
        "market",
        "selection",
        "decision_status",
        "refusal_reason",
        "bet_mode",
        "safety_level",
        "safety_score",
        "ai_probability",
        "bookmaker_odds",
        "value",
        "suggested_stake",
        "potential_profit",
        "potential_return",
        "benefice_avec_mise",
        "score_exact_1",
        "score_exact_1_proba",
        "score_exact_2",
        "score_exact_2_proba",
        "score_exact_3",
        "score_exact_3_proba",
        "tennis_engine_score",
        "tennis_edge",
        "football_trap_signal",
        "learning_adjustment",
        "calibration_adjustment",
        "threshold_profile",
        "decision_reason",
        "confidence",
        "ia_badge",
        "decision",
    ]

    tracking_cols = [
        "result",
        "stake",
        "profit",
        "potential_profit",
        "potential_return",
        "return_paid",
        "benefice_avec_mise",
        "benefice_avec_mise_encaisse",
        "final_winner",
        "final_score_home",
        "final_score_away",
        "status_detail",
        "resolved_at",
        "bet_mode",
    ]

    cols = core_cols + ([] if compact else tracking_cols)
    cols = list(dict.fromkeys(cols))
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
        "potential_profit",
        "potential_return",
        "return_paid",
        "benefice_avec_mise",
        "benefice_avec_mise_encaisse",
        "safety_score",
        "tennis_engine_score",
        "score_exact_1_proba",
        "score_exact_2_proba",
        "score_exact_3_proba",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    # Sécurité Streamlit / PyArrow : évite les crashs d'affichage
    for col in out.columns:
        out[col] = out[col].astype(str)

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


def plot_gain_loss_bars(data):
    if data.empty:
        st.info("Aucun pari termine pour afficher le graphique gains / pertes.")
        return

    chart = data.copy()
    chart["profit"] = pd.to_numeric(chart.get("profit", 0), errors="coerce").fillna(0)
    chart["stake"] = pd.to_numeric(chart.get("stake", 0), errors="coerce").fillna(0)
    chart["bookmaker_odds"] = pd.to_numeric(chart.get("bookmaker_odds", 0), errors="coerce").fillna(0)
    chart["result"] = chart.get("result", pd.Series("", index=chart.index)).astype(str).str.upper()

    chart["gain_loss"] = chart.apply(
        lambda row: abs(row["profit"]) if row["result"] == "WIN" else -abs(row["profit"] or row["stake"]),
        axis=1,
    )
    if "date" in chart.columns:
        chart = chart.sort_values("date").copy()
    chart["bet_number"] = range(1, len(chart) + 1)
    chart["match"] = chart.apply(match_display_label, axis=1)
    chart["gain_loss_label"] = chart["gain_loss"].apply(lambda value: f"{value:+.2f} EUR")
    chart["bar_color"] = chart["gain_loss"].apply(lambda value: "#22c55e" if value >= 0 else "#ef4444")

    if PLOTLY_OK:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=chart["bet_number"],
                    y=chart["gain_loss"],
                    text=chart["gain_loss_label"],
                    textposition="outside",
                    marker_color=chart["bar_color"],
                    customdata=chart[["match", "result", "stake", "bookmaker_odds"]],
                    hovertemplate=(
                        "Pari %{x}<br>"
                        "%{customdata[0]}<br>"
                        "Resultat : %{customdata[1]}<br>"
                        "Mise : %{customdata[2]:.2f} EUR<br>"
                        "Cote : %{customdata[3]:.2f}<br>"
                        "Gain / perte : %{text}<extra></extra>"
                    ),
                )
            ]
        )
        fig.add_hline(y=0, line_color="rgba(255,255,255,.35)", line_width=1)
        fig.update_layout(
            title="Gains / pertes par pari",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8fafc",
            xaxis_title="Pari",
            yaxis_title="Montant",
            bargap=0.26,
            margin=dict(l=20, r=20, t=56, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(chart.set_index("bet_number")["gain_loss"])


def normalize_lookup_text(value):
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def to_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def fair_odds(probability):
    probability = to_float(probability)
    if probability <= 0:
        return 0.0
    return round(1 / probability, 2)


def machine_mode(probability, value, market=""):
    probability = to_float(probability)
    value = to_float(value)
    market_l = str(market or "").strip().lower()
    if market_l == "draw":
        if probability >= 0.285 and value >= 0:
            return "NUL POSSIBLE"
        if probability >= 0.30 and value >= -0.08:
            return "NUL POSSIBLE"
        if probability >= 0.265 or value > 0:
            return "WATCHLIST"
        return "NO BET"
    if probability >= 0.70 and value >= 0.03:
        return "MEGA VALUE"
    if probability >= 0.63 and value >= 0.01:
        return "SAFE PICK"
    if probability >= 0.58 and value >= 0.03:
        return "VALUE BET"
    if value > 0:
        return "WATCHLIST"
    return "NO BET"


def machine_stake(mode, bankroll):
    if mode not in RECOMMENDED_MODES:
        return 0.0
    bankroll = max(to_float(bankroll, 10.0), 0.0)
    growth_factor = max(1.0, min(bankroll / 10.0, 6.0))
    floors = {
        "MEGA VALUE": min(3.00 * growth_factor, 5.00),
        "SAFE PICK": min(2.00 * growth_factor, 5.00),
        "FAVORI SOLIDE": min(1.50 * growth_factor, 5.00),
        "VALUE BET": min(1.00 * growth_factor, 5.00),
        "NUL POSSIBLE": min(1.00 * growth_factor, 2.50, 5.00),
    }
    return round(min(floors.get(mode, 0.0), bankroll * 0.50, 5.00, bankroll), 2)


def find_machine_match_rows(data, category, first_name, second_name):
    if data.empty:
        return data

    first_key = normalize_lookup_text(first_name)
    second_key = normalize_lookup_text(second_name)
    if not first_key and not second_key:
        return pd.DataFrame(columns=data.columns)

    out = data.copy()
    out["_home_key"] = out.get("home_team", "").astype(str).apply(normalize_lookup_text)
    out["_away_key"] = out.get("away_team", "").astype(str).apply(normalize_lookup_text)

    if category != "all" and "category" in out.columns:
        out = out[out["category"].astype(str).str.lower().eq(category)]

    mask = pd.Series(True, index=out.index)
    if first_key:
        mask &= out["_home_key"].str.contains(first_key, na=False) | out["_away_key"].str.contains(first_key, na=False)
    if second_key:
        mask &= out["_home_key"].str.contains(second_key, na=False) | out["_away_key"].str.contains(second_key, na=False)

    return out[mask].drop(columns=["_home_key", "_away_key"], errors="ignore")


def value_min_odds(probability, target_value=0.03):
    probability = to_float(probability)
    if probability <= 0:
        return 0.0
    return round((1 + target_value) / probability, 2)


def add_machine_prediction_row(rows, source, market, selection, probability, market_odds, bankroll, note):
    probability = to_float(probability)
    market_odds = to_float(market_odds)
    implied = (1 / market_odds) if market_odds > 1 else ""
    value = probability * market_odds - 1 if market_odds > 1 else ""
    mode = machine_mode(probability, value, market) if market_odds > 1 else "INFO IA"
    rows.append({
        "source": source,
        "market": market,
        "selection": selection,
        "proba_ia": probability,
        "proba_marche": implied,
        "cote_marche_dispo": market_odds if market_odds > 1 else "",
        "cote_ia_estimee": fair_odds(probability),
        "cote_min_value": value_min_odds(probability),
        "value": value,
        "mode": mode,
        "mise": machine_stake(mode, bankroll),
        "note": note,
    })


def build_ai_machine_rows(match_rows, bankroll, sport_category_value):
    rows = []

    if not match_rows.empty:
        local = sort_recommendations(match_rows).copy()
        for _, row in local.head(18).iterrows():
            add_machine_prediction_row(
                rows,
                "IA",
                row.get("market", ""),
                row.get("selection", ""),
                row.get("ai_probability", 0),
                row.get("bookmaker_odds", 0),
                bankroll,
                row.get("decision_reason", "Analyse IA locale"),
            )

        first = local.iloc[0]
        if sport_category_value == "football":
            derived = [
                ("Total buts", "Over 2.5", "over_25", "over_25"),
                ("Total buts", "Under 2.5", "under_25", "under_25"),
                ("BTTS", "Oui", "btts_yes", "btts_yes"),
                ("BTTS", "Non", "btts_no", "btts_no"),
            ]
            for market, selection, probability_key, odds_key in derived:
                probability = to_float(first.get(probability_key, 0))
                if probability > 0:
                    add_machine_prediction_row(
                        rows,
                        "IA",
                        market,
                        selection,
                        probability,
                        0,
                        bankroll,
                        "Marche derive du modele football - cote IA estimee",
                    )

    if rows:
        return pd.DataFrame(rows)

    return pd.DataFrame(columns=[
        "source",
        "market",
        "selection",
        "proba_ia",
        "proba_marche",
        "cote_marche_dispo",
        "cote_ia_estimee",
        "cote_min_value",
        "value",
        "mode",
        "mise",
        "note",
    ])


def format_machine_table(data):
    if data.empty:
        return data
    out = data.copy()
    for col in ["proba_ia", "proba_marche", "value"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda value: format_pct(value) if value != "" else "")
    for col in ["cote_marche_dispo", "cote_ia_estimee", "cote_min_value", "mise"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda value: "" if value == "" else round(to_float(value), 2))
    return out.astype(str)


def mode_class(mode):
    m = str(mode).upper()
    if "RISKY" in m:
        return "risky-card", "badge-red"
    if "WATCH" in m:
        return "watch-card", "badge-yellow"
    return "pick-card", "badge-green"



def ensure_match_key(data):
    if data.empty: return data
    out = data.copy()
    if "match_key" not in out.columns:
        out["match_key"] = out["sport"].astype(str).str.lower()+"|"+out["date"].astype(str).str[:10]+"|"+out["home_team"].astype(str).str.lower()+"|"+out["away_team"].astype(str).str.lower()
    return out

def best_card_rows(data):
    if data.empty: return data
    out = ensure_match_key(data).copy()
    out["_stake"] = pd.to_numeric(out.get("suggested_stake", 0), errors="coerce").fillna(0)
    out["_value"] = pd.to_numeric(out.get("value", 0), errors="coerce").fillna(-9)
    out["_safety"] = pd.to_numeric(out.get("safety_score", 0), errors="coerce").fillna(0)
    out["_prob"] = pd.to_numeric(out.get("ai_probability", 0), errors="coerce").fillna(0)
    out["_priority"] = pd.to_numeric(out.get("priority", 0), errors="coerce").fillna(0)
    out = out[
        out["bet_mode"].isin(RECOMMENDED_MODES)
        & (out["_stake"] > 0)
        & ((out["_value"] > 0) | (out["bet_mode"].isin(["FAVORI SOLIDE", "NUL POSSIBLE"])))
    ].copy()
    if out.empty: return out.drop(columns=["_stake", "_value", "_safety", "_prob", "_priority"], errors="ignore")
    out["_card_score"] = out["_priority"]*1000 + out["_stake"]*100 + out["_value"]*100 + out["_safety"] + out["_prob"]*10
    out = out.sort_values("_card_score", ascending=False).drop_duplicates("match_key", keep="first")
    return out.drop(columns=["_stake", "_value", "_safety", "_prob", "_priority", "_card_score"], errors="ignore")

def probable_detail(row):
    category = str(row.get("category", "")).lower()
    score = row.get("score_exact_1", "")
    proba = row.get("score_exact_1_proba", "")
    try:
        proba_text = f" ({float(proba):.2f}%)"
    except Exception:
        proba_text = ""

    if category == "football":
        return f"Score exact probable : <b>{score}{proba_text}</b>"
    if category == "tennis":
        return f"Set probable : <b>{score}{proba_text}</b>"
    return ""

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
            odds_value = float(row.get("bookmaker_odds", 0) or 0)
            potential_profit = float(row.get("potential_profit", stake * (odds_value - 1)) or 0)
            potential_return = float(row.get("potential_return", stake * odds_value) or 0)
            benefice_avec_mise = float(row.get("benefice_avec_mise", potential_return) or 0)
            odds = row.get("bookmaker_odds", "")
            mode = row.get("bet_mode", "")
            source = row.get("odds_source", "")
            reason = row.get("decision_reason", "")
            card_class, badge_class = mode_class(mode)
            detail_line = probable_detail(row)

            st.markdown(
                f"""
                <div class="pro-card {card_class}">
                    <span class="badge {badge_class}">{mode}</span>
                    <span class="badge badge-blue">{row.get("market", "")}</span><br><br>
                    <b>{row.get("home_team", "")} vs {row.get("away_team", "")}</b><br>
                    <span class="small-muted">{row.get("sport", "")}</span><br><br>
                    Proba IA : <b>{proba:.1f}%</b><br>
                    Cote : <b>{odds}</b><br>
                    Source : <b>{source}</b><br>
                    Value : <b>{value:.1f}%</b><br>
                    Profit net potentiel : <b>+{potential_profit:.2f} EUR</b><br>
                    Retour total potentiel : <b>{potential_return:.2f} EUR</b><br>
                    Benefice avec mise : <b>{benefice_avec_mise:.2f} EUR</b><br>
                    Mise conseillée : <b>{stake:.2f}€</b><br>
                    <span class="small-muted">
                    {reason}
                    </span><br>
                    {detail_line}
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
    if data.empty:
        return pd.DataFrame()

    data = ensure_match_key(data).copy()
    if "category" in data.columns:
        data = data[data["category"].astype(str).str.lower() == "football"].copy()
    if data.empty:
        return pd.DataFrame()
    data["_priority"] = pd.to_numeric(data.get("priority", 0), errors="coerce").fillna(0)
    data = data.sort_values("_priority", ascending=False).drop_duplicates("match_key", keep="first")

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
            "source": row.get("odds_source", ""),
            "alerte": row.get("score_exact_alert", ""),
            "mise": row.get("suggested_stake", 0),
        })

    return pd.DataFrame(rows)


def tennis_sets_table(data):
    if data.empty:
        return pd.DataFrame()
    if "category" in data.columns:
        data = data[data["category"].astype(str).str.lower() == "tennis"].copy()
    if data.empty:
        return pd.DataFrame()

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

# Les mises restent visibles en priorité, mais les onglets sport affichent tout l'univers analysé.
filtered_today = today_only(filtered)

football_df = filtered[filtered["category"] == "football"].copy()
tennis_df = filtered[filtered["category"] == "tennis"].copy()
recommended = filtered[
    filtered["bet_mode"].isin(RECOMMENDED_MODES)
    & (filtered["suggested_stake"] > 0)
].copy()

# ============================================================
# REBALANCE DES MISES
# ============================================================

def rebalance_stake(row):
    # LEVEL MAX : le dashboard n'augmente plus les mises calculées par le pipeline.
    return float(row.get("suggested_stake", 0) or 0)

recommended["suggested_stake"] = recommended.apply(
    rebalance_stake,
    axis=1
)
recommended = best_card_rows(recommended)
risky_list = filtered[
    filtered["bet_mode"].astype(str).str.upper().str.contains("RISKY")
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
            Balance évolutive • Départ 10€ • Les mises montent avec les gains • Scores auto<br>
            Dernière actualisation : <b>{last_update}</b> | Alertes Telegram : <b>{telegram_count}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Lignes analysees", len(filtered))
c2.metric("Paris avec mise", len(recommended))
c3.metric("Football", len(football_df))
c4.metric("Tennis", len(tennis_df))
c5.metric("Mise max", f"{filtered_today['suggested_stake'].max() if not filtered_today.empty else 0:.2f}€")

b1, b2, b3, b4 = st.columns(4)
b1.metric("Balance actuelle", f"{current_bankroll_display:.2f}€")
b2.metric("Capital de départ", f"{initial_bankroll_display:.2f}€")
b3.metric("Profit net", f"{profit_net_display:+.2f}€")
b4.metric("ROI balance", f"{roi_display:+.1f}%")

st.progress(progress_display / 100)
st.caption(
    f"Objectif suivant : {next_goal_display:.0f}€ | Progression : {progress_display:.1f}% | "
    f"Mise max par pari : {max_single_bet_display:.2f}€ | Exposition max jour : {max_daily_exposure_display:.2f}€"
)

odds_level, odds_message = odds_freshness_message(upcoming_odds)
if odds_message:
    if odds_level == "warning":
        st.warning(odds_message)
    else:
        st.info(odds_message)

# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    "Accueil",
    "⚽ Football",
    "🎾 Tennis",
    "Paris conseillés",
    "Football score exact",
    "Tennis sets",
    "Mises / bankroll",
    "Résultats / ROI",
    "Backtest IA",
    "Toutes les lignes",
    "Machine IA",
])

with tabs[0]:
    st.subheader("Meilleurs spots a venir")
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
    st.subheader("⚽ Football — matchs du jour")
    st.caption("Uniquement les matchs du jour, triés par priorité puis sécurité IA.")

    football_reco = best_card_rows(sort_recommendations(football_df))
    football_analysis = sort_recommendations(football_df)

    if football_df.empty:
        st.info("Aucun match football du jour trouvé dans les données actuelles.")
    else:
        render_cards(football_reco, limit=9)

        st.subheader("Analyse football")
        score_df = football_score_table(football_analysis)
        st.dataframe(score_df, use_container_width=True, hide_index=True, height=420)

        show_table(football_analysis, height=520)


with tabs[2]:
    st.subheader("🎾 Tennis — matchs du jour")
    st.caption("ATP/WTA du jour uniquement. Les gros joueurs et gros tournois sont priorisés.")

    tennis_reco = best_card_rows(sort_recommendations(tennis_df))
    tennis_analysis = sort_recommendations(tennis_df)

    if tennis_df.empty:
        st.info("Aucun match tennis du jour trouvé dans les données actuelles.")
    else:
        render_cards(tennis_reco, limit=9)

        st.subheader("Analyse tennis")
        sets_df = tennis_sets_table(tennis_analysis)
        st.dataframe(sets_df, use_container_width=True, hide_index=True, height=420)

        show_table(tennis_analysis, height=520)



with tabs[3]:
    st.subheader("Paris conseillés avec mise")
    render_cards(sort_recommendations(recommended), limit=9)
    show_table(sort_recommendations(recommended), height=620)

with tabs[4]:
    st.subheader("Football — score exact plus lisible")
    st.caption("Le score exact reste un marché très difficile. Ici, le dashboard affiche les 3 scores les plus probables + signal piège bookmaker.")

    football_reco = best_card_rows(sort_recommendations(football_df))
    football_analysis = sort_recommendations(football_df)
    score_df = football_score_table(football_analysis)

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

with tabs[5]:
    st.subheader("Tennis — nombre de sets probable")
    st.caption("Affiche le score en sets le plus probable : 2-0 ou 2-1 selon les probabilités du pipeline.")

    tennis_reco = best_card_rows(sort_recommendations(tennis_df))
    tennis_analysis = sort_recommendations(tennis_df)
    sets_df = tennis_sets_table(tennis_analysis)

    if sets_df.empty:
        st.info("Aucun match tennis disponible.")
    else:
        st.dataframe(sets_df, use_container_width=True, hide_index=True, height=620)

        a, b = st.columns(2)
        with a:
            plot_bar(sets_df.head(25), "match", "proba set probable", "Probabilité du nombre de sets")
        with b:
            plot_bar(tennis_reco.head(25), "home_team", "tennis_engine_score", "Score IA tennis")

with tabs[6]:
    st.subheader("Mises conseillées / bankroll")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Balance actuelle", f"{current_bankroll_display:.2f}€")
    k2.metric("Profit net", f"{profit_net_display:+.2f}€")
    k3.metric("Objectif suivant", f"{next_goal_display:.0f}€")
    k4.metric("Exposition max jour", f"{max_daily_exposure_display:.2f}€")
    st.progress(progress_display / 100)
    st.caption(f"Progression vers {next_goal_display:.0f}€ : {progress_display:.1f}%")

    stake_cols = [
        "date",
        "sport",
        "odds_source",
        "home_team",
        "away_team",
        "selection",
        "bet_mode",
        "ai_probability",
        "bookmaker_odds",
        "value",
        "suggested_stake",
        "potential_profit",
        "potential_return",
        "benefice_avec_mise",
        "stake_percent",
        "kelly_fraction",
        "bankroll",
        "safety_score",
        "calibration_adjustment",
        "decision_reason",
    ]
    stake_cols = [c for c in stake_cols if c in recommended.columns]

    if recommended.empty:
        st.info("Aucune mise conseillée pour le moment.")
    else:
        total_stake = recommended["suggested_stake"].sum()
        avg_stake = recommended["suggested_stake"].mean()
        total_potential_profit = recommended["potential_profit"].sum() if "potential_profit" in recommended.columns else 0
        total_potential_return = recommended["potential_return"].sum() if "potential_return" in recommended.columns else 0
        total_benefice_avec_mise = recommended["benefice_avec_mise"].sum() if "benefice_avec_mise" in recommended.columns else total_potential_return
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Mise totale conseillée", f"{total_stake:.2f}€")
        m2.metric("Mise moyenne", f"{avg_stake:.2f}€")
        m3.metric("Nombre de tickets", len(recommended))
        m4.metric("Profit potentiel", f"+{total_potential_profit:.2f} EUR")
        m5.metric("Retour potentiel", f"{total_potential_return:.2f} EUR")
        m6.metric("Benefice avec mise", f"{total_benefice_avec_mise:.2f} EUR")
        st.dataframe(clean_table(recommended[stake_cols]), use_container_width=True, hide_index=True, height=540)

with tabs[7]:
    st.subheader("Résultats IA / ROI / Archive")

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
        tr["potential_profit"] = (tr["stake"] * (tr["bookmaker_odds"] - 1)).clip(lower=0).round(2)
        tr["potential_return"] = (tr["stake"] * tr["bookmaker_odds"]).clip(lower=0).round(2)
        tr["return_paid"] = 0.0
        tr.loc[tr["result"] == "WIN", "return_paid"] = tr.loc[tr["result"] == "WIN", "potential_return"]
        tr["benefice_avec_mise"] = tr["potential_return"]
        tr["benefice_avec_mise_encaisse"] = tr["return_paid"]
        if "category" not in tr.columns:
            tr["category"] = tr["sport"].apply(sport_category)
        if "resolved_at" not in tr.columns:
            tr["resolved_at"] = ""

        finished = tr[tr["result"].isin(["WIN", "LOSS"])].copy()
        pending = tr[tr["result"] == "PENDING"].copy()
        wins = tr[tr["result"] == "WIN"].copy()
        losses = tr[tr["result"] == "LOSS"].copy()
        archive_view = archive.copy() if not archive.empty else finished.copy()
        if "resolved_at" not in archive_view.columns:
            archive_view["resolved_at"] = ""
        archive_view["_resolved_dt"] = pd.to_datetime(
            archive_view["resolved_at"].replace("", pd.NA),
            utc=True,
            errors="coerce",
        )
        finished_week = archive_view[
            archive_view["_resolved_dt"].isna()
            | (archive_view["_resolved_dt"] >= pd.Timestamp.utcnow() - pd.Timedelta(days=7))
        ].drop(columns=["_resolved_dt"], errors="ignore")
        archive_view = archive_view.drop(columns=["_resolved_dt"], errors="ignore")

        total_staked = finished["stake"].sum() if not finished.empty else 0
        total_profit = finished["profit"].sum() if not finished.empty else 0
        total_return_paid = finished["return_paid"].sum() if "return_paid" in finished.columns and not finished.empty else 0
        total_benefice_avec_mise_paid = finished["benefice_avec_mise_encaisse"].sum() if "benefice_avec_mise_encaisse" in finished.columns and not finished.empty else 0
        roi = total_profit / total_staked if total_staked > 0 else 0
        winrate = len(wins) / len(finished) if len(finished) else 0

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Terminés", len(finished))
        r2.metric("En attente", len(pending))
        r3.metric("ROI", f"{roi * 100:.2f}%")
        r4.metric("Winrate", f"{winrate * 100:.2f}%")

        total_won = wins["profit"].sum() if not wins.empty else 0
        total_lost = abs(losses["profit"].sum()) if not losses.empty else 0

        if len(finished) < 50:
            st.info(
                "Échantillon encore trop faible : il faut idéalement 50 à 100 paris terminés pour juger la fiabilité réelle."
            )

        r5, r6, r7, r8, r9 = st.columns(5)
        r5.metric("Misé total", f"{total_staked:.2f}€")
        r6.metric("Profit net", f"{total_profit:.2f}€")
        r7.metric("Cote moyenne", f"{finished['bookmaker_odds'].mean() if not finished.empty else 0:.2f}")
        r8.metric("Retour encaisse", f"{total_return_paid:.2f} EUR")
        r9.metric("Benefice avec mise", f"{total_benefice_avec_mise_paid:.2f} EUR")

        g1, g2 = st.columns(2)
        g1.metric(
            "💰 Argent gagné",
            f"+{total_won:.2f}€"
        )
        g2.metric(
            "📉 Argent perdu",
            f"-{total_lost:.2f}€"
        )

        if not wins.empty:
            st.success(
                f"Les gains viennent principalement des paris WIN avec bonnes cotes et value positive."
            )

        if not losses.empty:
            st.warning(
                f"Les pertes viennent surtout des paris plus volatils ou des scores exacts difficiles."
            )

        if not finished.empty:
            by_sport = (
                finished.groupby("category")
                .agg(
                    paris=("result", "count"),
                    wins=("result", lambda x: (x == "WIN").sum()),
                    pertes=("result", lambda x: (x == "LOSS").sum()),
                    mise=("stake", "sum"),
                    profit=("profit", "sum"),
                )
                .reset_index()
            )
            by_sport["roi"] = by_sport.apply(
                lambda r: r["profit"] / r["mise"] if r["mise"] > 0 else 0,
                axis=1,
            )
            st.subheader("Topo des paris terminés")
            st.dataframe(by_sport, use_container_width=True, hide_index=True, height=180)

            st.subheader("Gains / pertes")
            plot_gain_loss_bars(finished)

            chart = finished.sort_values("date").copy()
            chart["cumulative_profit"] = chart["profit"].cumsum()
            chart["bet_number"] = range(1, len(chart) + 1)
            plot_line(chart, "bet_number", "cumulative_profit", "Courbe profit cumulé")

        result_tabs = st.tabs(["En attente", "Terminés semaine", "Archive", "Gagnés", "Perdus"])
        with result_tabs[0]:
            show_table(pending.sort_values("date"), height=460, compact=False)
        with result_tabs[1]:
            show_table(finished_week.sort_values("date", ascending=False), height=520, compact=False)
        with result_tabs[2]:
            show_table(archive_view.sort_values("date", ascending=False), height=620, compact=False)
        with result_tabs[3]:
            show_table(wins.sort_values("profit", ascending=False), height=460, compact=False)
        with result_tabs[4]:
            show_table(losses.sort_values("profit", ascending=True), height=460, compact=False)

with tabs[8]:
    st.subheader("Backtest, calibration et auto-apprentissage")

    if backtest_summary.empty and calibration.empty and threshold_profiles.empty:
        st.info("Aucun backtest disponible pour le moment. Lance le pipeline automatique pour générer la calibration.")
    else:
        if not threshold_profiles.empty:
            st.markdown("**Seuils optimisés**")
            show_table(threshold_profiles, height=180, compact=False)

        if not backtest_summary.empty:
            bt = backtest_summary.copy()
            st.markdown("**Segments testés**")
            show_table(bt.sort_values("bets", ascending=False) if "bets" in bt.columns else bt, height=360, compact=False)

        if not calibration.empty:
            cal = calibration.copy()
            st.markdown("**Calibration des probabilités**")
            show_table(cal.sort_values("bets", ascending=False) if "bets" in cal.columns else cal, height=420, compact=False)

with tabs[9]:
    st.subheader("Toutes les lignes analysées")
    show_table(sort_recommendations(filtered), height=720)

with tabs[10]:
    st.subheader("Machine IA")

    sport_machine = st.selectbox(
        "Sport",
        ["football", "tennis"],
        key="machine_ia_sport",
    )

    n1, n2 = st.columns(2)
    with n1:
        machine_first = st.text_input(
            "Equipe / joueur 1",
            placeholder="ex: PSG, Alcaraz...",
            key="machine_ia_first",
        )
    with n2:
        machine_second = st.text_input(
            "Equipe / joueur 2",
            placeholder="ex: Marseille, Sinner...",
            key="machine_ia_second",
        )

    has_input = bool(machine_first.strip() or machine_second.strip())
    if not has_input:
        st.info("Entre simplement le match pour lancer l'analyse IA.")
    else:
        machine_matches = find_machine_match_rows(filtered, sport_machine, machine_first, machine_second)
        machine_table = build_ai_machine_rows(
            machine_matches,
            current_bankroll_display,
            sport_machine,
        )

        if machine_table.empty:
            st.warning("Match non trouve dans les predictions disponibles. Relance la mise a jour des donnees puis le pipeline pour l'analyser.")
        else:
            proba_ia_sort = pd.to_numeric(machine_table["proba_ia"].replace("", pd.NA), errors="coerce").fillna(0)
            value_sort = pd.to_numeric(machine_table["value"].replace("", 0), errors="coerce").fillna(0)
            machine_table["_sort_prob"] = proba_ia_sort
            machine_table["_sort_value"] = value_sort
            machine_table = machine_table.sort_values(
                ["_sort_prob", "_sort_value"],
                ascending=[False, False],
            ).drop(columns=["_sort_prob", "_sort_value"], errors="ignore")

            top = machine_table.iloc[0]
            top_probability = top.get("proba_ia", "")
            top_value = top.get("value", "")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Plus probable", str(top.get("selection", "")))
            m2.metric("Probabilite IA", format_pct(top_probability) if top_probability != "" else "")
            m3.metric("Cote IA estimee", f"{to_float(top.get('cote_ia_estimee', 0)):.2f}")
            m4.metric("Cote min value", f"{to_float(top.get('cote_min_value', 0)):.2f}")

            if top_value != "":
                st.caption(f"Value sur la cote marche disponible : {format_pct(top_value)}")

            st.success("Analyse IA calculee depuis les predictions disponibles.")
            st.dataframe(
                format_machine_table(machine_table),
                use_container_width=True,
                hide_index=True,
                height=420,
            )

            if sport_machine == "football":
                score_machine = football_score_table(machine_matches)
                if not score_machine.empty:
                    st.subheader("Scores exacts les plus probables")
                    st.dataframe(score_machine.head(5), use_container_width=True, hide_index=True, height=220)

            if sport_machine == "tennis":
                sets_machine = tennis_sets_table(machine_matches)
                if not sets_machine.empty:
                    st.subheader("Sets les plus probables")
                    st.dataframe(sets_machine.head(5), use_container_width=True, hide_index=True, height=220)
