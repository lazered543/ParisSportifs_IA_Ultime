import subprocess
import time


def run_command(cmd):
    print(f"\nLancement : {cmd}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"Erreur sur : {cmd}")
        return False

    print(f"Termine : {cmd}")
    return True


print("\n==============================")
print("SYSTEME IA PARIS SPORTIFS")
print("==============================\n")

run_command("python scripts/update_data.py")
time.sleep(2)

run_command("python scripts/update_player_scorers.py")
time.sleep(2)

run_command("python scripts/update_results_auto.py")
time.sleep(2)

run_command("python scripts/backtest_learning.py")
time.sleep(2)

run_command("python scripts/run_pipeline.py")
time.sleep(2)

run_command("python scripts/save_bets_to_tracking.py")
time.sleep(2)

run_command("python scripts/send_telegram_alerts.py")
time.sleep(2)

print("\n==============================")
print("SYSTEME TERMINE")
print("==============================\n")

print("- Dashboard Streamlit pret")
print("- Resultats anciens mis a jour")
print("- Backtest et calibration mis a jour")
print("- tracking_results.csv mis a jour")
print("- value_bets_today.csv mis a jour")
print("- Alertes Telegram envoyees si VALUE BETS disponibles")
