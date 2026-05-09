import requests
import json

url = "http://127.0.0.1:8000/predict"

payload = {
    "home_team": "Man City",
    "away_team": "Brentford",
    "league": "E0",
    "country": "England",
    "neutral": 0,

    "home_elo": 1980,
    "away_elo": 1700,
    "elo_diff": 280,

    "home_last5_points_avg": 2.2,
    "away_last5_points_avg": 1.4,
    "home_last5_goals_for_avg": 2.1,
    "away_last5_goals_for_avg": 1.3,
    "home_last5_goals_against_avg": 0.9,
    "away_last5_goals_against_avg": 1.5,
    "home_last5_win_rate": 0.7,
    "away_last5_win_rate": 0.4,

    "last5_points_diff": 0.8,
    "last5_goals_for_diff": 0.8,
    "last5_goals_against_diff": -0.6,
    "last5_win_rate_diff": 0.3
}

response = requests.post(url, json=payload)
print(response.status_code)
print(json.dumps(response.json(), indent=2, ensure_ascii=False))
