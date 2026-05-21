from pathlib import Path
import pandas as pd

def download_tennis_history(out_dir="data/raw/tennis"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    urls = [
        ("atp_2025.csv", "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2025.csv"),
        ("atp_2026.csv", "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2026.csv"),
        ("wta_2025.csv", "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2025.csv"),
        ("wta_2026.csv", "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_2026.csv"),
    ]
    frames = []
    for name, url in urls:
        try:
            df = pd.read_csv(url)
            df.to_csv(out / name, index=False)
            frames.append(df)
            print(f"OK tennis {name}: {len(df)} matchs")
        except Exception as e:
            print(f"SKIP tennis {name}: {e}")
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        all_df.to_csv("data/processed/tennis_history_all.csv", index=False)
        print(f"Historique tennis total: {len(all_df)} lignes")
