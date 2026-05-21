from src.connectors.football_data_downloader import download_football_history
from src.connectors.tennis_downloader import download_tennis_history
from src.connectors.the_odds_api import fetch_upcoming_odds

if __name__ == "__main__":
    print("=== Update historique football ===")
    download_football_history()
    print("=== Update historique tennis ===")
    download_tennis_history()
    print("=== Update cotes à venir ===")
    fetch_upcoming_odds()
    print("Terminé.")
