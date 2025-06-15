import requests
import openpyxl
from datetime import datetime
import os

# Chemin de destination
save_path = r"C:\Users\bhiste\OneDrive - MND\Bureau\Perso\scores_ligue1.xlsx"

# Création du fichier Excel
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Scores Ligue 1"
ws.append(["ID Match", "Ligue", "Saison", "Journée", "Équipe Domicile", "Score", "Équipe Extérieure", "Score"])

# Plage de match ID à tester
start_id = 70000
end_id = 80000

for match_id in range(start_id, end_id + 1):
    url = f"https://ma-api.ligue1.fr/championship-match/l1_championship_match_{match_id}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"⛔ Match {match_id} introuvable.")
            continue

        data = response.json()

        championshipId = data.get("championshipId","N/A")
        season = data.get("season", "N/A")
        gameweek = data.get("gameWeekNumber", "N/A")
        home_team = data["home"]["clubIdentity"]["shortName"]
        home_score = data["home"]["score"]
        away_team = data["away"]["clubIdentity"]["shortName"]
        away_score = data["away"]["score"]

        print(f"✅ Match {match_id} | {championshipId}  S{season} J{gameweek} : {home_team} {home_score} - {away_score} {away_team}")

        ws.append([match_id, championshipId, season, gameweek, home_team, home_score, away_team, away_score])

    except Exception as e:
        print(f"⚠️ Erreur pour le match {match_id} : {e}")

# Sauvegarde du fichier
wb.save(save_path)
print(f"📄 Fichier enregistré ici : {save_path}")
