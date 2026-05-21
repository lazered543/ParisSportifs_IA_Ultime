import subprocess
import time


def run_command(cmd):

    print(f"\nLancement : {cmd}\n")

    result = subprocess.run(
        cmd,
        shell=True
    )

    if result.returncode != 0:
        print(f"Erreur sur : {cmd}")

    else:
        print(f"Terminé : {cmd}")


print("\n==============================")
print("SYSTEME IA PARIS SPORTIFS")
print("==============================\n")

# 1. Mise à jour des données
run_command(
    "python scripts/update_data.py"
)

time.sleep(2)

# 2. Génération IA
run_command(
    "python scripts/run_pipeline.py"
)

time.sleep(2)

# 3. Sauvegarde tracking
run_command(
    "python scripts/save_bets_to_tracking.py"
)

time.sleep(2)

print("\n==============================")
print("SYSTEME TERMINE")
print("==============================\n")

print(
    "Tu peux maintenant ouvrir :\n"
)

print(
    "- Dashboard Streamlit"
)

print(
    "- tracking_results.csv"
)

print(
    "- value_bets_today.csv"
)