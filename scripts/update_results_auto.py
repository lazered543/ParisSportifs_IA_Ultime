import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz

try:
    from sofascore_wrapper import SofaScore
    SOFASCORE_OK = True
except Exception:
    SOFASCORE_OK = False

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
FOOTBALL_DATA_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN")
API_TENNIS_KEY = os.getenv("API_TENNIS_KEY")

TRACK_PATH = Path("tracking_results.csv")
MANUAL_RESULTS_PATH = Path("manual_results.csv")
LEARNING_PATH = Path("data/learning/ai_learning_profile.csv")
LEARNING_SUMMARY_PATH = Path("data/learning/ai_learning_summary.csv")
BASE_URL = "https://api.the-odds-api.com/v4"

sofa = SofaScore() if SOFASCORE_OK else None


def safe_str(x):
    text = str(x).lower().strip()

    if text in ["nan", "none"]:
        return ""

    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9 ]", " ", text)

    replacements = {
        "paris saint germain": "psg",
        "paris sg": "psg",
        "st etienne": "saint etienne",
        "as saint etienne": "saint etienne",
        "manchester united": "man utd",
        "manchester city": "man city",
        "tottenham hotspur": "tottenham",
        "internazionale": "inter",
        "fc internazionale milano": "inter",
        "bayern munich": "bayern",
        "real madrid cf": "real madrid",
        "atletico madrid": "atl madrid",
        "borussia dortmund": "dortmund",
        "sporting cp": "sporting",
        "rb leipzig": "leipzig",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return " ".join(text.split())


def last_name(text):
    parts = safe_str(text).split()
    if not parts:
        return ""

    # tennis : "H. Hurkacz", "Hurkacz Hubert", etc.
    if len(parts) >= 2:
        return parts[-1]

    return parts[0]


def tennis_name_variants(name):
    clean = safe_str(name)
    parts = clean.split()

    variants = {clean}

    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]
        variants.add(last)
        variants.add(f"{first[0]} {last}")
        variants.add(f"{last} {first[0]}")
        variants.add(f"{last} {first}")

    return {v for v in variants if v}


def names_match(a, b, threshold=82):
    a_clean = safe_str(a)
    b_clean = safe_str(b)

    if not a_clean or not b_clean:
        return False

    if a_clean == b_clean or a_clean in b_clean or b_clean in a_clean:
        return True

    score = max(
        fuzz.token_sort_ratio(a_clean, b_clean),
        fuzz.partial_ratio(a_clean, b_clean),
        fuzz.WRatio(a_clean, b_clean),
    )

    if score >= threshold:
        return True

    # Cas tennis : nom de famille suffisant si assez distinctif
    a_last = last_name(a_clean)
    b_last = last_name(b_clean)

    if len(a_last) >= 5 and len(b_last) >= 5:
        if fuzz.ratio(a_last, b_last) >= 88:
            return True

    return False


def match_pair_score(home, away, event_home, event_away):
    home_clean = safe_str(home)
    away_clean = safe_str(away)
    eh = safe_str(event_home)
    ea = safe_str(event_away)

    direct = (
        fuzz.WRatio(home_clean, eh)
        + fuzz.WRatio(away_clean, ea)
    ) / 2

    reverse = (
        fuzz.WRatio(home_clean, ea)
        + fuzz.WRatio(away_clean, eh)
    ) / 2

    return direct, reverse


def get_score_number(score):
    try:
        return float(score)
    except Exception:
        return None


def parse_datetime(value):
    if value is None or pd.isna(value):
        return None

    try:
        text = str(value).strip()

        if re.fullmatch(r"\d{8}", text):
            parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce", utc=True)
        elif re.match(r"\d{4}-\d{2}-\d{2}", text):
            parsed = pd.to_datetime(text, errors="coerce", utc=True)
        else:
            parsed = pd.to_datetime(value, errors="coerce", utc=True, dayfirst=True)
    except Exception:
        return None

    if pd.isna(parsed):
        return None

    return parsed.to_pydatetime()


def event_datetime(event):
    if not event:
        return None

    for key in ["commence_time", "date", "start_time"]:
        parsed = parse_datetime(event.get(key))

        if parsed is not None:
            return parsed

    return None


def dates_are_close(row_date, other_date, max_days=2):
    row_dt = parse_datetime(row_date)
    other_dt = parse_datetime(other_date)

    if row_dt is None or other_dt is None:
        return True

    return abs(row_dt - other_dt) <= timedelta(days=max_days)


def pending_status(row, event=None):
    start = parse_datetime(row.get("date"))
    now = datetime.now(timezone.utc)

    if event is not None:
        return "MATCH_FOUND_NOT_COMPLETED_YET"

    if start is None:
        return "RESULT_NOT_FOUND_VERIFY_MANUALLY"

    if start > now:
        return "MATCH_NOT_STARTED_YET"

    sport = str(row.get("sport", "")).lower()
    grace = timedelta(hours=8 if "tennis" in sport else 3)

    if now - start <= grace:
        return "MATCH_IN_PROGRESS_OR_RESULT_NOT_AVAILABLE_YET"

    return "RESULT_NOT_FOUND_VERIFY_MANUALLY"


def ensure_manual_results_template():
    if MANUAL_RESULTS_PATH.exists():
        return

    template = pd.DataFrame(
        columns=[
            "date",
            "sport",
            "home_team",
            "away_team",
            "winner",
            "score_home",
            "score_away",
            "source",
            "notes",
        ]
    )

    template.to_csv(MANUAL_RESULTS_PATH, index=False)


def load_manual_results():
    ensure_manual_results_template()

    try:
        df = pd.read_csv(MANUAL_RESULTS_PATH, low_memory=False)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    for col in ["home_team", "away_team", "winner"]:
        if col not in df.columns:
            df[col] = ""

    return df


def find_manual_result(row, manual_df):
    if manual_df.empty:
        return None

    home = row.get("home_team", "")
    away = row.get("away_team", "")

    for _, manual in manual_df.iterrows():
        m_home = manual.get("home_team", "")
        m_away = manual.get("away_team", "")

        same_match = names_match(home, m_home) and names_match(away, m_away)
        reversed_match = names_match(home, m_away) and names_match(away, m_home)

        if same_match or reversed_match:
            if not dates_are_close(row.get("date", ""), manual.get("date", "")):
                continue

            winner = manual.get("winner", "")

            if pd.notna(winner) and str(winner).strip():
                score_home = manual.get("score_home", None)
                score_away = manual.get("score_away", None)

                if reversed_match:
                    score_home, score_away = score_away, score_home

                return str(winner), score_home, score_away

    return None



def fetch_sofascore_events_for_date(sport, target_date=None):
    """
    Remplacé par API-FOOTBALL / API-TENNIS.
    On garde le nom pour compatibilité avec le reste du script.
    """

    if target_date is None:
        target_date = datetime.now(timezone.utc)

    sport_lower = str(sport).lower()

    try:
        if "tennis" in sport_lower:
            return fetch_api_tennis_events(target_date)

        return fetch_api_football_events(target_date)

    except Exception as e:
        print("Erreur API events :", e)
        return []


def fetch_api_football_events(target_date):
    if not FOOTBALL_DATA_TOKEN:
        print("FOOTBALL_DATA_TOKEN manquante.")
        return []

    date_str = target_date.strftime("%Y-%m-%d")

    url = "https://api.football-data.org/v4/matches"

    headers = {
        "X-Auth-Token": FOOTBALL_DATA_TOKEN,
    }

    params = {
        "dateFrom": date_str,
        "dateTo": date_str,
    }

    try:
        r = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30,
        )

        r.raise_for_status()

        data = r.json()

        events = []

        for item in data.get("matches", []):
            status = item.get("status", "")

            home_score = (
                item.get("score", {})
                .get("fullTime", {})
                .get("home")
            )

            away_score = (
                item.get("score", {})
                .get("fullTime", {})
                .get("away")
            )

            events.append({
                "homeTeam": {
                    "name": item.get("homeTeam", {}).get("name", "")
                },
                "awayTeam": {
                    "name": item.get("awayTeam", {}).get("name", "")
                },
                "homeScore": {
                    "current": home_score
                },
                "awayScore": {
                    "current": away_score
                },
                "status": {
                    "type": (
                        "finished"
                        if status == "FINISHED"
                        else "notfinished"
                    )
                },
                "date": item.get("utcDate"),
            })

        return events

    except Exception as e:
        print("Erreur Football-Data :", e)
        return []


def fetch_api_tennis_events(target_date):
    if not API_TENNIS_KEY:
        return []

    date_str = target_date.strftime("%Y-%m-%d")

    url = "https://api.api-tennis.com/tennis/"

    params = {
        "method": "get_fixtures",
        "APIkey": API_TENNIS_KEY,
        "date_start": date_str,
        "date_stop": date_str,
    }

    try:
        r = requests.get(
            url,
            params=params,
            timeout=30,
        )

        r.raise_for_status()

        data = r.json()

        events = []

        for item in data.get("result", []):
            home_name = item.get("event_first_player", "")
            away_name = item.get("event_second_player", "")

            home_score = item.get("event_final_result", "")

            h_score = None
            a_score = None

            if "-" in str(home_score):
                try:
                    split_score = str(home_score).split("-")
                    h_score = float(split_score[0].strip())
                    a_score = float(split_score[1].strip())
                except Exception:
                    pass

            status_raw = str(
                item.get("event_status", "")
            ).lower()

            finished = (
                "finished" in status_raw
                or "after" in status_raw
                or "ended" in status_raw
            )

            events.append({
                "homeTeam": {
                    "name": home_name
                },
                "awayTeam": {
                    "name": away_name
                },
                "homeScore": {
                    "current": h_score
                },
                "awayScore": {
                    "current": a_score
                },
                "status": {
                    "type": "finished" if finished else "notfinished"
                },
                "date": item.get("event_date"),
            })

        return events

    except Exception as e:
        print("Erreur API-TENNIS :", e)
        return []


def find_sofascore_match(row, events):
    home = row.get("home_team", "")
    away = row.get("away_team", "")

    best_event = None
    best_score = 0

    for event in events:
        try:
            event_home = event.get("homeTeam", {}).get("name", "")
            event_away = event.get("awayTeam", {}).get("name", "")

            direct, reverse = match_pair_score(
                home,
                away,
                event_home,
                event_away,
            )

            score = max(direct, reverse)

            if score > best_score:
                best_score = score
                best_event = event

        except Exception:
            continue

    if best_score >= 78:
        return best_event

    return None


def resolve_sofascore_result(row, event):
    if not event:
        return None, None, None

    try:
        status = (
            event.get("status", {})
            .get("type", "")
        )

        if status != "finished":
            return None, None, None

        home_score = (
            event.get("homeScore", {})
            .get("current")
        )

        away_score = (
            event.get("awayScore", {})
            .get("current")
        )

        if home_score is None or away_score is None:
            return None, None, None

        home_score = float(home_score)
        away_score = float(away_score)

        if home_score > away_score:
            winner = row.get("home_team")

        elif away_score > home_score:
            winner = row.get("away_team")

        else:
            winner = "DRAW"

        return winner, home_score, away_score

    except Exception:
        return None, None, None


def fetch_scores(sport):
    if not ODDS_API_KEY:
        return []

    url = f"{BASE_URL}/sports/{sport}/scores"

    params = {
        "apiKey": ODDS_API_KEY,
        "daysFrom": 3,
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    except Exception as e:
        print(f"Erreur scores {sport} :", e)
        return []


def find_matching_score(row, scores):
    home = row.get("home_team", "")
    away = row.get("away_team", "")
    row_dt = parse_datetime(row.get("date", ""))

    candidates = []

    for event in scores:
        event_home = event.get("home_team", "")
        event_away = event.get("away_team", "")

        direct, reverse = match_pair_score(
            home,
            away,
            event_home,
            event_away,
        )

        score = max(direct, reverse)

        if score >= 76:
            event_dt = event_datetime(event)

            if row_dt is not None and event_dt is not None:
                delta = abs(row_dt - event_dt)

                if delta > timedelta(days=2):
                    continue
            else:
                delta = timedelta(days=99)

            candidates.append((score, -delta.total_seconds(), event))

    if candidates:
        candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    return None


def get_score_for_name(scores, name):
    for score in scores:
        if names_match(name, score.get("name", "")):
            return get_score_number(score.get("score"))

    return None


def get_winner_from_event(row, event):
    if not event:
        return None, None, None

    if not event.get("completed", False):
        return None, None, None

    scores = event.get("scores", [])

    if not scores or len(scores) < 2:
        return None, None, None

    parsed_scores = []

    for score in scores:
        score_value = get_score_number(score.get("score"))

        if score_value is not None:
            parsed_scores.append((score.get("name"), score_value))

    if len(parsed_scores) < 2:
        return None, None, None

    parsed_scores = sorted(parsed_scores, key=lambda item: item[1], reverse=True)
    top_name, top_score = parsed_scores[0]
    second_name, second_score = parsed_scores[1]

    home_score = get_score_for_name(scores, row.get("home_team", ""))
    away_score = get_score_for_name(scores, row.get("away_team", ""))

    if home_score is None or away_score is None:
        event_home = event.get("home_team", "")
        event_away = event.get("away_team", "")

        if names_match(row.get("home_team", ""), event_home):
            home_score = get_score_number(scores[0].get("score"))
            away_score = get_score_number(scores[1].get("score"))

        elif names_match(row.get("home_team", ""), event_away):
            home_score = get_score_number(scores[1].get("score"))
            away_score = get_score_number(scores[0].get("score"))

    if top_score > second_score:
        return top_name, home_score, away_score

    return "DRAW", home_score, away_score


def try_resolve_tennis_from_history(row):
    hist_paths = [
        Path("data/processed/tennis_history_all.csv"),
        Path("data/processed/tennis_history.csv"),
        Path("data/raw/tennis_history.csv"),
    ]

    history = None

    for path in hist_paths:
        if path.exists():
            try:
                history = pd.read_csv(path, low_memory=False)
                break
            except Exception:
                pass

    if history is None or history.empty:
        return None, None

    home = safe_str(row.get("home_team", ""))
    away = safe_str(row.get("away_team", ""))

    winner_cols = [
        "winner",
        "Winner",
        "winner_name",
        "player_winner",
        "match_winner",
    ]

    date_cols = [
        "date",
        "Date",
        "match_date",
        "tourney_date",
    ]

    for _, h in history.tail(10000).iterrows():
        line = " ".join([safe_str(v) for v in h.values])

        if home in line and away in line:
            row_date = row.get("date", "")
            history_date = ""

            for date_col in date_cols:
                if date_col in history.columns:
                    history_date = h.get(date_col, "")
                    break

            if history_date and not dates_are_close(row_date, history_date, max_days=3):
                continue

            for col in winner_cols:
                if col in history.columns:
                    winner = h.get(col)

                    if pd.notna(winner):
                        return str(winner), h

    return None, None


def evaluate_bet(row, winner):
    market = safe_str(row.get("market", ""))
    home = safe_str(row.get("home_team", ""))
    away = safe_str(row.get("away_team", ""))
    selection = safe_str(row.get("selection", ""))

    if winner is None:
        return "PENDING"

    winner_clean = safe_str(winner)

    if market == "home win":
        return "WIN" if names_match(winner_clean, home) else "LOSS"

    if market == "away win":
        return "WIN" if names_match(winner_clean, away) else "LOSS"

    if market == "draw":
        return "WIN" if winner_clean == "draw" else "LOSS"

    if market == "player 1 win":
        if selection:
            return "WIN" if names_match(winner_clean, selection) else "LOSS"
        return "WIN" if names_match(winner_clean, home) else "LOSS"

    if market == "player 2 win":
        if selection:
            return "WIN" if names_match(winner_clean, selection) else "LOSS"
        return "WIN" if names_match(winner_clean, away) else "LOSS"

    return "PENDING"


def calc_profit(row):
    result = row.get("result", "PENDING")

    stake = pd.to_numeric(
        row.get("stake", row.get("suggested_stake", 0)),
        errors="coerce",
    )

    odds = pd.to_numeric(
        row.get("bookmaker_odds", 0),
        errors="coerce",
    )

    if pd.isna(stake):
        stake = 0

    if pd.isna(odds):
        odds = 0

    if result == "WIN":
        return round(float(stake) * (float(odds) - 1), 2)

    if result == "LOSS":
        return round(-float(stake), 2)

    return 0.0


def mode_group(mode):
    mode = str(mode).upper().strip()

    if "MEGA" in mode or "MONSTER" in mode:
        return "MEGA VALUE"

    if "ULTRA SAFE" in mode or mode == "SAFE":
        return "SAFE"

    if "VALUE" in mode:
        return "MEDIUM"

    if "AGGRESSIVE" in mode or "RISKY" in mode:
        return "RISKY"

    return "OTHER"


def build_learning_profile(tracking):
    LEARNING_PATH.parent.mkdir(parents=True, exist_ok=True)

    finished = tracking[tracking["result"].isin(["WIN", "LOSS"])].copy()

    if finished.empty:
        pd.DataFrame().to_csv(LEARNING_PATH, index=False)
        pd.DataFrame([{
            "status": "NO_FINISHED_BETS",
            "message": "Pas encore assez de résultats pour apprendre.",
        }]).to_csv(LEARNING_SUMMARY_PATH, index=False)
        return

    if "bet_mode" not in finished.columns:
        finished["bet_mode"] = "UNKNOWN"

    if "category" not in finished.columns:
        finished["category"] = "unknown"

    finished["mode_group"] = finished["bet_mode"].apply(mode_group)
    finished["stake"] = pd.to_numeric(finished.get("stake", 0), errors="coerce").fillna(0)
    finished["profit"] = pd.to_numeric(finished.get("profit", 0), errors="coerce").fillna(0)
    finished["bookmaker_odds"] = pd.to_numeric(finished.get("bookmaker_odds", 0), errors="coerce").fillna(0)

    groups = []

    dimensions = [
        ("mode_group", "Mode de pari"),
        ("category", "Sport"),
        ("sport", "Compétition"),
        ("market", "Marché"),
    ]

    for col, label in dimensions:
        if col not in finished.columns:
            continue

        grouped = (
            finished
            .groupby(col)
            .agg(
                bets=("result", "count"),
                wins=("result", lambda x: (x == "WIN").sum()),
                losses=("result", lambda x: (x == "LOSS").sum()),
                stake=("stake", "sum"),
                profit=("profit", "sum"),
                avg_odds=("bookmaker_odds", "mean"),
            )
            .reset_index()
            .rename(columns={col: "segment"})
        )

        grouped["dimension"] = label
        grouped["winrate"] = grouped["wins"] / grouped["bets"]
        grouped["roi"] = grouped.apply(
            lambda r: r["profit"] / r["stake"] if r["stake"] > 0 else 0,
            axis=1,
        )

        def recommendation(r):
            if r["bets"] < 3:
                return "WAIT_MORE_DATA"
            if r["roi"] > 0.15 and r["winrate"] >= 0.65:
                return "BOOST"
            if r["roi"] < -0.10 or r["winrate"] < 0.45:
                return "REDUCE"
            return "KEEP"

        grouped["ai_recommendation"] = grouped.apply(recommendation, axis=1)
        groups.append(grouped)

    profile = pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()
    profile.to_csv(LEARNING_PATH, index=False)

    total_bets = len(finished)
    total_profit = finished["profit"].sum()
    total_stake = finished["stake"].sum()
    global_roi = total_profit / total_stake if total_stake > 0 else 0
    global_winrate = (finished["result"] == "WIN").mean()

    summary = pd.DataFrame([{
        "finished_bets": total_bets,
        "profit": round(total_profit, 2),
        "stake": round(total_stake, 2),
        "roi": round(global_roi, 4),
        "winrate": round(global_winrate, 4),
        "best_action": "BOOST les segments rentables, REDUCE les segments négatifs.",
    }])

    summary.to_csv(LEARNING_SUMMARY_PATH, index=False)


def main():
    if not ODDS_API_KEY:
        print("ODDS_API_KEY manquante : resultats manuels/historique uniquement.")

    if not TRACK_PATH.exists():
        print("tracking_results.csv introuvable.")
        return

    tracking = pd.read_csv(TRACK_PATH)

    if tracking.empty:
        print("Tracking vide.")
        return

    # ============================================================
    # FORCE LES COLONNES TEXTE
    # ============================================================
    text_cols = [
        "status_detail",
        "final_winner",
        "final_score_home",
        "final_score_away",
        "result",
        "selection",
    ]

    for col in text_cols:
        if col not in tracking.columns:
            tracking[col] = ""

        tracking[col] = (
            tracking[col]
            .fillna("")
            .astype(str)
        )

    if "result" not in tracking.columns:
        tracking["result"] = "PENDING"

    tracking["result"] = (
        tracking["result"]
        .fillna("PENDING")
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": "PENDING", "NAN": "PENDING", "NONE": "PENDING"})
    )

    if "stake" not in tracking.columns:
        tracking["stake"] = pd.to_numeric(
            tracking.get("suggested_stake", 0),
            errors="coerce",
        ).fillna(0.0).astype(float)

    tracking["stake"] = pd.to_numeric(
        tracking.get("stake", 0),
        errors="coerce",
    ).fillna(0.0).astype(float)

    tracking["profit"] = pd.to_numeric(
        tracking.get("profit", 0),
        errors="coerce",
    ).fillna(0.0).astype(float)

    if "category" not in tracking.columns:
        tracking["category"] = ""

    category_blank = tracking["category"].fillna("").astype(str).str.strip() == ""

    if "sport" not in tracking.columns:
        tracking["sport"] = ""

    tracking.loc[category_blank, "category"] = (
        tracking.loc[category_blank, "sport"]
        .fillna("")
        .astype(str)
        .str.lower()
        .apply(
            lambda sport:
            "tennis"
            if "tennis" in sport
            else "football"
            if "soccer" in sport or "football" in sport
            else "autre"
        )
    )

    manual_df = load_manual_results()

    pending = tracking[
        ~tracking["result"].isin(["WIN", "LOSS", "VOID"])
    ].copy()

    if pending.empty:
        print("Aucun pari en attente.")
        build_learning_profile(tracking)
        return

    sports = pending["sport"].dropna().unique()

    scores_by_sport = {}
    api_events_by_key = {}

    for sport in sports:
        print("Recuperation scores :", sport)
        scores_by_sport[sport] = fetch_scores(sport)

    # ============================================================
    # API-FOOTBALL / API-TENNIS PAR DATE REELLE DU PARI
    # ============================================================
    for _, pending_row in pending.iterrows():
        sport = pending_row.get("sport", "")
        row_dt = parse_datetime(pending_row.get("date", ""))

        if row_dt is None:
            row_dt = datetime.now(timezone.utc)

        date_key = row_dt.strftime("%Y-%m-%d")
        cache_key = (str(sport), date_key)

        if cache_key in api_events_by_key:
            continue

        try:
            events = fetch_sofascore_events_for_date(
                sport,
                row_dt,
            )

            api_events_by_key[cache_key] = events

            print(
                "API events trouves :",
                sport,
                date_key,
                len(events),
            )

        except Exception as e:
            print("Erreur API date :", sport, date_key, e)
            api_events_by_key[cache_key] = []

    updated = 0
    manual_updated = 0
    not_found = 0

    for idx, row in tracking.iterrows():
        if row.get("result") in ["WIN", "LOSS", "VOID"]:
            continue

        winner = None
        score_home = None
        score_away = None
        event = None
        status_detail = ""

        manual_result = find_manual_result(row, manual_df)

        if manual_result is not None:
            winner, score_home, score_away = manual_result
            status_detail = "RESOLVED_FROM_MANUAL_RESULTS"
            manual_updated += 1

        else:
            sport = row.get("sport", "")
            scores = scores_by_sport.get(sport, [])

            # ============================================================
            # PRIORITÉ 1 : SOFASCORE
            # ============================================================

            row_dt = parse_datetime(row.get("date", ""))

            if row_dt is None:
                row_dt = datetime.now(timezone.utc)

            date_key = row_dt.strftime("%Y-%m-%d")

            sofa_events = api_events_by_key.get(
                (str(sport), date_key),
                []
            )

            sofa_match = find_sofascore_match(
                row,
                sofa_events
            )

            winner, score_home, score_away = (
                resolve_sofascore_result(
                    row,
                    sofa_match
                )
            )

            if winner is not None:
                status_detail = "RESOLVED_FROM_SOFASCORE"

            # ============================================================
            # PRIORITÉ 2 : ODDS API
            # ============================================================

            if winner is None:
                event = find_matching_score(row, scores)

                winner, score_home, score_away = (
                    get_winner_from_event(
                        row,
                        event
                    )
                )

                if winner is not None:
                    status_detail = "RESOLVED_FROM_ODDS_API"

            # ============================================================
            # PRIORITÉ 3 : HISTORIQUE TENNIS
            # ============================================================

            if winner is None and "tennis" in str(sport).lower():
                winner, _ = try_resolve_tennis_from_history(row)

                if winner is not None:
                    score_home = None
                    score_away = None
                    status_detail = "RESOLVED_FROM_TENNIS_HISTORY"

        if winner is None:
            tracking.at[idx, "status_detail"] = pending_status(row, event)
            not_found += 1
            continue

        result = evaluate_bet(row, winner)

        if result in ["WIN", "LOSS"]:
            tracking.at[idx, "result"] = result
            tracking.at[idx, "final_winner"] = winner
            tracking.at[idx, "final_score_home"] = (
                "" if score_home is None else str(score_home)
            )

            tracking.at[idx, "final_score_away"] = (
                "" if score_away is None else str(score_away)
            )

            tracking.at[idx, "profit"] = calc_profit(tracking.loc[idx])
            tracking.at[idx, "status_detail"] = status_detail or "RESOLVED"

            updated += 1

            print(
                row.get("home_team"),
                "vs",
                row.get("away_team"),
                "=>",
                result,
                "| gagnant :",
                winner,
            )

    tracking.to_csv(TRACK_PATH, index=False)
    build_learning_profile(tracking)

    print("Résultats mis à jour :", updated)
    print("Résultats manuels utilisés :", manual_updated)
    print("Résultats à vérifier manuellement :", not_found)
    print("Auto-learning mis à jour :", LEARNING_PATH)


if __name__ == "__main__":
    main()