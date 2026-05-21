from pathlib import Path
import pandas as pd

# Football-Data codes principaux Europe
LEAGUES = {
    "E0": "Premier League",
    "E1": "Championship",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
    "N1": "Eredivisie",
    "P1": "Portugal Liga",
    "B1": "Belgium",
    "SC0": "Scotland Premiership",
}

# saisons récentes : 2024/25 et 2025/26. Ajoute 2324, 2223 si tu veux plus d'historique.
SEASONS = ["2425", "2526"]

def download_football_history(out_dir="data/raw/football"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in SEASONS:
        for code, league in LEAGUES.items():
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            try:
                df = pd.read_csv(url)
                if df.empty:
                    continue
                df["LeagueCode"] = code
                df["LeagueName"] = league
                df["Season"] = season
                file_path = out / f"{season}_{code}.csv"
                df.to_csv(file_path, index=False)
                frames.append(df)
                print(f"OK {season} {league}: {len(df)} matchs")
            except Exception as e:
                print(f"SKIP {season} {code}: {e}")
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        all_df.to_csv("data/processed/football_history_all.csv", index=False)
        print(f"Historique football total: {len(all_df)} lignes")
    else:
        print("Aucune donnée téléchargée. Vérifie internet.")
