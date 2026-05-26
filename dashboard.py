import streamlit as st
import pandas as pd
from pathlib import Path

try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


st.set_page_config(
    page_title="IA Paris Sportifs Ultime",
    layout="wide",
    page_icon="PS"
)

APP_PASSWORD = "29052007"
PRED_PATH = Path("data/predictions/predictions_today.csv")
TRACK_PATH = Path("tracking_results.csv")
TELEGRAM_SENT_PATH = Path("data/telegram_sent.csv")


st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(168,85,247,.28), transparent 30%),
        radial-gradient(circle at 100% 100%, rgba(34,211,238,.18), transparent 28%),
        linear-gradient(135deg, #111827 0%, #10131f 45%, #0d1020 100%);
    color: #f7f8ff;
}

.block-container {
    max-width: 1550px;
    padding-top: 28px;
}

section[data-testid="stSidebar"] {
    background: #141827;
    border-right: 1px solid rgba(255,255,255,.12);
}

h1, h2, h3 {
    color: #f7f8ff !important;
}

[data-testid="stMetric"] {
    background: #171b2b;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 14px 40px rgba(0,0,0,.25);
}

.hero-box {
    background: linear-gradient(135deg, rgba(244,114,208,.18), rgba(34,211,238,.08)), #171b2b;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 24px;
    padding: 28px;
    margin-bottom: 22px;
}

.hero-title {
    font-size: 38px;
    font-weight: 950;
    margin-bottom: 8px;
}

.hero-sub {
    color: #9ea8c4;
    font-size: 15px;
}

.card {
    background: #171b2b;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 18px;
    padding: 16px;
    margin-bottom: 14px;
}

.value-card {
    background: linear-gradient(135deg, rgba(52,211,153,.13), rgba(23,27,43,1));
    border-left: 5px solid #34d399;
}

.badge {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 12px;
}

.badge-green {
    background: rgba(52,211,153,.18);
    color: #34d399;
}

.badge-red {
    background: rgba(251,113,133,.18);
    color: #fb7185;
}

.badge-yellow {
    background: rgba(251,191,36,.18);
    color: #fbbf24;
}

.badge-blue {
    background: rgba(96,165,250,.18);
    color: #60a5fa;
}

.stDataFrame {
    border-radius: 18px !important;
    overflow: hidden !important;
}

@media (max-width: 900px) {
    .block-container {
        padding-top: 18px;
        padding-left: 12px;
        padding-right: 12px;
    }

    .hero-title {
        font-size: 27px;
    }

    .hero-box {
        padding: 18px;
        border-radius: 18px;
    }

    [data-testid="stMetric"] {
        padding: 12px;
    }
}
</style>
""", unsafe_allow_html=True)


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False


if not st.session_state.authenticated:
    st.markdown("""
    <div class="hero-box">
        <div class="hero-title">IA Paris Sportifs Ultime</div>
        <div class="hero-sub">
            Accès privé • Football • Tennis • Value Betting • ROI • Telegram
        </div>
    </div>
    """, unsafe_allow_html=True)

    password = st.text_input(
        "Mot de passe",
        type="password"
    )

    if password == APP_PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
    elif password:
        st.error("Mot de passe incorrect.")

    st.stop()


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


def prepare_base_data(data):
    data = data.copy()

    data["category"] = data["sport"].apply(sport_category)
    data["value"] = num_col(data, "value")
    data["ai_probability"] = num_col(data, "ai_probability")
    data["suggested_stake"] = num_col(data, "suggested_stake")
    data["bookmaker_odds"] = num_col(data, "bookmaker_odds")
    data["implied_probability"] = num_col(data, "implied_probability")

    return data


def clean_table(data):
    cols = [
        "date",
        "sport",
        "category",
        "home_team",
        "away_team",
        "market",
        "selection",
        "bet_mode",
        "mode_category",
        "result",
        "stake",
        "stake_percent",
        "kelly_fraction",
        "bankroll",
        "profit",
        "final_winner",
        "status_detail",
        "ai_probability",
        "bookmaker_odds",
        "implied_probability",
        "value",
        "confidence",
        "ia_badge",
        "decision",
        "suggested_stake",
        "score_exact_1",
        "score_exact_1_proba",
        "draw_hunter",
        "scorer_prediction",
        "tennis_engine_score",
        "tennis_edge",
    ]

    cols = [c for c in cols if c in data.columns]
    out = data[cols].copy()

    for col in ["ai_probability", "implied_probability", "value", "tennis_edge"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].apply(lambda x: "" if pd.isna(x) else f"{x * 100:.2f}%")

    for col in ["bookmaker_odds", "suggested_stake", "stake", "stake_percent", "kelly_fraction", "bankroll", "profit", "tennis_engine_score", "score_exact_1_proba"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    return out


def show_table(data, height=520):
    if data.empty:
        st.info("Aucune donnée à afficher.")
        return

    st.dataframe(
        clean_table(data),
        use_container_width=True,
        hide_index=True,
        height=height
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
            font_color="#f7f8ff"
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
            font_color="#f7f8ff"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(chart.set_index(x)[y])


def render_cards(data, limit=6):
    if data.empty:
        st.info("Aucun pari à afficher.")
        return

    cols = st.columns(3)

    for i, (_, row) in enumerate(data.head(limit).iterrows()):
        with cols[i % 3]:
            value = float(row.get("value", 0) or 0) * 100
            proba = float(row.get("ai_probability", 0) or 0) * 100
            odds = row.get("bookmaker_odds", "")

            st.markdown(
                f"""
                <div class="card value-card">
                    <b>{row.get("home_team", "")} vs {row.get("away_team", "")}</b><br>
                    <span style="color:#9ea8c4;">{row.get("sport", "")}</span><br><br>
                    <span class="badge badge-blue">{row.get("market", "")}</span><br><br>
                    Proba IA : <b>{proba:.1f}%</b><br>
                    Cote : <b>{odds}</b><br>
                    Value : <b>{value:.1f}%</b><br>
                    Confiance : <b>{row.get("confidence", "")}</b><br>
                    Badge : <b>{row.get("ia_badge", "")}</b>
                </div>
                """,
                unsafe_allow_html=True
            )


df = prepare_base_data(df)


st.sidebar.title("Filtres")

sports = st.sidebar.multiselect(
    "Compétitions",
    sorted(df["sport"].dropna().unique()),
    default=sorted(df["sport"].dropna().unique())
)

categories = st.sidebar.multiselect(
    "Catégorie",
    sorted(df["category"].dropna().unique()),
    default=sorted(df["category"].dropna().unique())
)

markets = st.sidebar.multiselect(
    "Marchés",
    sorted(df["market"].dropna().unique()),
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
        filtered.astype(str).apply(
            lambda r: s in " ".join(r.values).lower(),
            axis=1
        )
    ]

football_df = filtered[filtered["category"] == "football"].copy()
tennis_df = filtered[filtered["category"] == "tennis"].copy()
value_bets = filtered[filtered["decision"] == "VALUE BET"].copy()

last_update = df["last_update"].iloc[0] if "last_update" in df.columns else "Inconnue"

telegram_count = 0
if TELEGRAM_SENT_PATH.exists():
    try:
        telegram_count = len(pd.read_csv(TELEGRAM_SENT_PATH))
    except Exception:
        telegram_count = 0


st.markdown(
    f"""
    <div class="hero-box">
        <div class="hero-title">IA Paris Sportifs Ultime</div>
        <div class="hero-sub">
            Football + Tennis • Value Betting • Telegram • ROI • Tracking • Courbes
            <br><br>
            Dernière actualisation : <b>{last_update}</b> |
            Alertes Telegram mémorisées : <b>{telegram_count}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


c1, c2, c3, c4 = st.columns(4)

best_value = filtered["value"].max() if not filtered.empty else 0
max_stake = filtered["suggested_stake"].max() if not filtered.empty else 0

c1.metric("Lignes analysées", len(filtered))
c2.metric("Value Bets", len(value_bets))
c3.metric("Meilleure value", f"{best_value * 100:.2f}%")
c4.metric("Mise max", f"{max_stake:.2f}€")


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

    render_cards(value_bets.sort_values("value", ascending=False), limit=6)

    c1, c2 = st.columns(2)

    with c1:
        plot_bar(
            filtered.sort_values("value", ascending=False).head(25),
            "home_team",
            "value",
            "Top value"
        )

    with c2:
        plot_bar(
            filtered.sort_values("ai_probability", ascending=False).head(25),
            "home_team",
            "ai_probability",
            "Top probabilités IA"
        )

    st.subheader("Toutes les meilleures lignes")
    show_table(filtered.sort_values("value", ascending=False).head(80), height=520)


with tabs[1]:
    st.subheader("Football")

    render_cards(
        football_df[football_df["decision"] == "VALUE BET"].sort_values("value", ascending=False),
        limit=6
    )

    c1, c2 = st.columns(2)

    with c1:
        plot_bar(
            football_df.sort_values("value", ascending=False).head(25),
            "home_team",
            "value",
            "Football - Value"
        )

    with c2:
        if "over_25" in football_df.columns:
            plot_bar(
                football_df.sort_values("over_25", ascending=False).head(25),
                "home_team",
                "over_25",
                "Football - Over 2.5"
            )

    show_table(football_df.sort_values("value", ascending=False), height=620)


with tabs[2]:
    st.subheader("Tennis")

    render_cards(
        tennis_df[tennis_df["decision"] == "VALUE BET"].sort_values("value", ascending=False),
        limit=6
    )

    c1, c2 = st.columns(2)

    with c1:
        plot_bar(
            tennis_df.sort_values("value", ascending=False).head(25),
            "home_team",
            "value",
            "Tennis - Value"
        )

    with c2:
        if "tennis_engine_score" in tennis_df.columns:
            plot_bar(
                tennis_df.sort_values("tennis_engine_score", ascending=False).head(25),
                "home_team",
                "tennis_engine_score",
                "Tennis - Score IA"
            )

    show_table(tennis_df.sort_values("value", ascending=False), height=620)


with tabs[3]:
    st.subheader("Value Bets")

    render_cards(value_bets.sort_values("value", ascending=False), limit=9)
    show_table(value_bets.sort_values("value", ascending=False), height=650)

with tabs[4]:
    st.subheader("Résultats IA / ROI")

    if tracking.empty:
        st.warning("Aucun tracking disponible.")
    else:
        tracking = tracking.copy()

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

        tracking["profit"] = pd.to_numeric(
            tracking.get("profit", 0),
            errors="coerce"
        ).fillna(0)

        if "category" not in tracking.columns:
            tracking["category"] = tracking["sport"].apply(sport_category)

        if "bet_mode" not in tracking.columns:
            tracking["bet_mode"] = "NON CLASSÉ"

        def mode_group(mode):
            mode = str(mode).upper().strip()

            if "MEGA" in mode or "MONSTER" in mode:
                return "💎 MEGA VALUE"

            if "ULTRA SAFE" in mode or mode == "SAFE":
                return "🟢 SAFE"

            if "VALUE" in mode:
                return "🟡 MEDIUM"

            if "AGGRESSIVE" in mode or "RISKY" in mode:
                return "🔴 RISKY"

            return "⚪ AUTRE"

        tracking["mode_category"] = tracking["bet_mode"].apply(mode_group)

        finished = tracking[tracking["result"].isin(["WIN", "LOSS"])].copy()
        wins = tracking[tracking["result"] == "WIN"].copy()
        losses = tracking[tracking["result"] == "LOSS"].copy()
        pending = tracking[tracking["result"] == "PENDING"].copy()

        finished = finished.sort_values("date", ascending=False)
        wins = wins.sort_values("profit", ascending=False)
        losses = losses.sort_values("profit", ascending=True)
        pending = pending.sort_values("date", ascending=True)

        total_staked = finished["stake"].sum() if len(finished) else 0
        total_profit = finished["profit"].sum() if len(finished) else 0
        total_won = wins["profit"].sum() if len(wins) else 0
        total_lost = losses["profit"].sum() if len(losses) else 0
        roi = total_profit / total_staked if total_staked > 0 else 0
        win_rate = len(wins) / len(finished) if len(finished) else 0

        avg_odds = finished["bookmaker_odds"].mean() if len(finished) else 0
        best_win = wins["profit"].max() if len(wins) else 0
        worst_loss = losses["profit"].min() if len(losses) else 0

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Paris terminés", len(finished))
        r2.metric("Gagnés", len(wins))
        r3.metric("Perdus", len(losses))
        r4.metric("En attente", len(pending))

        r5, r6, r7, r8 = st.columns(4)
        r5.metric("Misé total", f"{total_staked:.2f}€")
        r6.metric("Profit net", f"{total_profit:.2f}€")
        r7.metric("ROI", f"{roi * 100:.2f}%")
        r8.metric("Win Rate", f"{win_rate * 100:.2f}%")

        r9, r10, r11, r12 = st.columns(4)
        r9.metric("Cote moyenne", f"{avg_odds:.2f}")
        r10.metric("Argent gagné", f"+{total_won:.2f}€")
        r11.metric("Argent perdu", f"{total_lost:.2f}€")
        r12.metric("Rentabilité", "Positive" if total_profit > 0 else "Négative" if total_profit < 0 else "Neutre")

        if total_profit > 0:
            st.success(f"L’IA est rentable : +{total_profit:.2f}€")
        elif total_profit < 0:
            st.error(f"L’IA est en perte : {total_profit:.2f}€")
        else:
            st.info("L’IA est à l’équilibre.")

        st.divider()

        st.subheader("📊 Performance par catégorie de pari")

        finished_modes = tracking[tracking["result"].isin(["WIN", "LOSS"])].copy()

        if finished_modes.empty:
            st.info("Pas encore assez de résultats pour analyser les catégories.")
        else:
            mode_stats = (
                finished_modes
                .groupby("mode_category")
                .agg(
                    paris=("result", "count"),
                    gagnes=("result", lambda x: (x == "WIN").sum()),
                    perdus=("result", lambda x: (x == "LOSS").sum()),
                    mise_totale=("stake", "sum"),
                    profit_total=("profit", "sum"),
                    cote_moyenne=("bookmaker_odds", "mean"),
                )
                .reset_index()
            )

            mode_stats["winrate"] = mode_stats["gagnes"] / mode_stats["paris"]
            mode_stats["roi"] = mode_stats.apply(
                lambda r: r["profit_total"] / r["mise_totale"] if r["mise_totale"] > 0 else 0,
                axis=1
            )

            mode_order = {
                "💎 MEGA VALUE": 0,
                "🟢 SAFE": 1,
                "🟡 MEDIUM": 2,
                "🔴 RISKY": 3,
                "⚪ AUTRE": 4,
            }

            mode_stats["order"] = mode_stats["mode_category"].map(mode_order).fillna(99)
            mode_stats = mode_stats.sort_values("order")

            mode_stats_display = mode_stats.copy()
            mode_stats_display["mise_totale"] = mode_stats_display["mise_totale"].round(2).astype(str) + "€"
            mode_stats_display["profit_total"] = mode_stats_display["profit_total"].round(2).astype(str) + "€"
            mode_stats_display["cote_moyenne"] = mode_stats_display["cote_moyenne"].round(2)
            mode_stats_display["winrate"] = (mode_stats_display["winrate"] * 100).round(2).astype(str) + "%"
            mode_stats_display["roi"] = (mode_stats_display["roi"] * 100).round(2).astype(str) + "%"
            mode_stats_display = mode_stats_display.drop(columns=["order"])

            st.dataframe(
                mode_stats_display,
                use_container_width=True,
                hide_index=True
            )

            c1, c2 = st.columns(2)

            with c1:
                plot_bar(
                    mode_stats,
                    "mode_category",
                    "profit_total",
                    "Profit par catégorie"
                )

            with c2:
                plot_bar(
                    mode_stats,
                    "mode_category",
                    "winrate",
                    "Winrate par catégorie"
                )

        st.divider()

        if not finished.empty:
            finished_chart = finished.sort_values("date").copy()
            finished_chart["cumulative_profit"] = finished_chart["profit"].cumsum()
            finished_chart["bet_number"] = range(1, len(finished_chart) + 1)

            c1, c2 = st.columns(2)

            with c1:
                plot_line(
                    finished_chart,
                    "bet_number",
                    "cumulative_profit",
                    "Courbe bankroll / profit cumulé"
                )

            with c2:
                plot_bar(
                    finished_chart,
                    "bet_number",
                    "profit",
                    "Gain / perte par pari"
                )

            c3, c4 = st.columns(2)

            with c3:
                if "category" in finished.columns:
                    cat_profit = (
                        finished.groupby("category")["profit"]
                        .sum()
                        .reset_index()
                        .sort_values("profit", ascending=False)
                    )
                    plot_bar(cat_profit, "category", "profit", "Profit par sport")

            with c4:
                if "market" in finished.columns:
                    market_profit = (
                        finished.groupby("market")["profit"]
                        .sum()
                        .reset_index()
                        .sort_values("profit", ascending=False)
                    )
                    plot_bar(market_profit, "market", "profit", "Profit par marché")

            if "sport" in finished.columns:
                sport_profit = (
                    finished.groupby("sport")
                    .agg(
                        paris=("result", "count"),
                        gains=("profit", "sum"),
                        mise=("stake", "sum"),
                    )
                    .reset_index()
                )

                sport_profit["roi"] = sport_profit.apply(
                    lambda r: r["gains"] / r["mise"] if r["mise"] > 0 else 0,
                    axis=1
                )

                st.subheader("Rentabilité par compétition")
                show_table(
                    sport_profit.sort_values("gains", ascending=False),
                    height=360
                )

        else:
            st.info("Aucun pari terminé pour le moment.")

        st.divider()

        result_tabs = st.tabs([
            "💰 Argent gagné",
            "📉 Argent perdu",
            "✅ Paris gagnés",
            "❌ Paris perdus",
            "⏳ En attente",
            "📋 Tous les résultats"
        ])

        with result_tabs[0]:
            st.subheader("💰 Argent gagné")
            st.metric("Bénéfice brut gagné", f"+{total_won:.2f}€")

            if not wins.empty:
                top_gains = wins.sort_values("profit", ascending=False)
                plot_bar(top_gains.head(20), "home_team", "profit", "Top gains")
                show_table(top_gains, height=520)
            else:
                st.info("Aucun pari gagnant pour le moment.")

        with result_tabs[1]:
            st.subheader("📉 Argent perdu")
            st.metric("Perte totale", f"{total_lost:.2f}€")

            if not losses.empty:
                top_losses = losses.sort_values("profit", ascending=True)
                plot_bar(top_losses.head(20), "home_team", "profit", "Top pertes")
                show_table(top_losses, height=520)
            else:
                st.info("Aucun pari perdant pour le moment.")

        with result_tabs[2]:
            st.subheader("✅ Paris gagnés triés par profit")
            show_table(wins.sort_values("profit", ascending=False), height=520)

        with result_tabs[3]:
            st.subheader("❌ Paris perdus triés par perte")
            show_table(losses.sort_values("profit", ascending=True), height=520)

        with result_tabs[4]:
            st.subheader("⏳ Paris en attente")
            show_table(pending.sort_values("date", ascending=True), height=520)

        with result_tabs[5]:
            st.subheader("📋 Historique complet")

            order = {"WIN": 0, "LOSS": 1, "PENDING": 2}
            all_results = tracking.copy()
            all_results["result_order"] = all_results["result"].map(order).fillna(3)

            show_table(
                all_results.sort_values(["result_order", "date"], ascending=[True, False]),
                height=650
            )


with tabs[5]:
    st.subheader("Tech IA")

    st.info(
        "Football : xG, Poisson, ELO équipes, calibration, score exact, BTTS, Over/Under, buteurs probables.\n\n"
        "Tennis : ELO joueurs, forme récente, winrate, probabilité marché, edge/value, score IA tennis.\n\n"
        "Modes de pari : ULTRA SAFE, SAFE, VALUE, AGGRESSIVE, MEGA VALUE.\n\n"
        "Automatisation : GitHub Actions, Telegram anti-spam, tracking ROI, dashboard Streamlit."
    )

    tech_cols = [
        "category",
        "sport",
        "bet_mode",
        "market",
        "ai_probability",
        "bookmaker_odds",
        "value",
        "confidence",
        "ia_badge",
        "tennis_engine_score",
        "tennis_edge",
    ]

    tech_cols = [c for c in tech_cols if c in filtered.columns]

    show_table(filtered[tech_cols].sort_values("value", ascending=False), height=560)


with tabs[6]:
    st.subheader("Toutes les prédictions")
    show_table(filtered.sort_values("value", ascending=False), height=720)
