# ============================================================
# ScoutIQ FastAPI Backend
# Football Match Result Prediction API
# ============================================================

import os
import json
import joblib
import pandas as pd
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "scoutiq_artifacts")

MODEL_PATH = os.path.join(ARTIFACT_DIR, "scoutiq_best_model.pkl")
ENCODER_PATH = os.path.join(ARTIFACT_DIR, "scoutiq_label_encoder.pkl")
FEATURES_PATH = os.path.join(ARTIFACT_DIR, "scoutiq_feature_columns.json")
METADATA_PATH = os.path.join(ARTIFACT_DIR, "scoutiq_metadata.json")


# ------------------------------------------------------------
# Load model artifacts
# ------------------------------------------------------------

def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Please copy scoutiq_artifacts folder beside app.py."
    )

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
feature_config = load_json(FEATURES_PATH)
metadata = load_json(METADATA_PATH)

FEATURE_COLS = feature_config["feature_cols"]
NUMERIC_COLS = feature_config["numeric_cols"]
CATEGORICAL_COLS = feature_config["categorical_cols"]


# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------

app = FastAPI(
    title="ScoutIQ Football Prediction API",
    description="AI backend for predicting football match outcomes: Home Win, Draw, Away Win.",
    version="1.0.0"
)


# ------------------------------------------------------------
# Request schema
# ------------------------------------------------------------

class MatchFeatures(BaseModel):
    # Basic match info
    home_team: str = Field(..., example="Man City")
    away_team: str = Field(..., example="Brentford")
    league: str = Field("E0", example="E0")
    country: str = Field("England", example="England")
    neutral: int = Field(0, example=0)

    # Recent form features
    home_last5_points_avg: float = 0
    home_last5_goals_for_avg: float = 0
    home_last5_goals_against_avg: float = 0
    home_last5_goal_diff_avg: float = 0
    home_last5_win_rate: float = 0
    home_last5_draw_rate: float = 0
    home_last5_loss_rate: float = 0
    home_last5_matches_count: float = 0

    away_last5_points_avg: float = 0
    away_last5_goals_for_avg: float = 0
    away_last5_goals_against_avg: float = 0
    away_last5_goal_diff_avg: float = 0
    away_last5_win_rate: float = 0
    away_last5_draw_rate: float = 0
    away_last5_loss_rate: float = 0
    away_last5_matches_count: float = 0

    home_last10_points_avg: float = 0
    home_last10_goals_for_avg: float = 0
    home_last10_goals_against_avg: float = 0
    home_last10_goal_diff_avg: float = 0
    home_last10_win_rate: float = 0
    home_last10_draw_rate: float = 0
    home_last10_loss_rate: float = 0
    home_last10_matches_count: float = 0

    away_last10_points_avg: float = 0
    away_last10_goals_for_avg: float = 0
    away_last10_goals_against_avg: float = 0
    away_last10_goal_diff_avg: float = 0
    away_last10_win_rate: float = 0
    away_last10_draw_rate: float = 0
    away_last10_loss_rate: float = 0
    away_last10_matches_count: float = 0

    # Venue-specific form
    home_home_last5_points_avg: float = 0
    home_home_last5_goals_for_avg: float = 0
    home_home_last5_goals_against_avg: float = 0
    home_home_last5_goal_diff_avg: float = 0
    home_home_last5_win_rate: float = 0
    home_home_last5_draw_rate: float = 0
    home_home_last5_loss_rate: float = 0
    home_home_last5_matches_count: float = 0

    away_away_last5_points_avg: float = 0
    away_away_last5_goals_for_avg: float = 0
    away_away_last5_goals_against_avg: float = 0
    away_away_last5_goal_diff_avg: float = 0
    away_away_last5_win_rate: float = 0
    away_away_last5_draw_rate: float = 0
    away_away_last5_loss_rate: float = 0
    away_away_last5_matches_count: float = 0

    # Elo
    home_elo: float = 0
    away_elo: float = 0
    elo_diff: float = 0

    # Market odds / probabilities
    home_odds: float = 0
    draw_odds: float = 0
    away_odds: float = 0
    market_home_prob: float = 0
    market_draw_prob: float = 0
    market_away_prob: float = 0
    market_home_minus_away: float = 0

    # Differences
    last5_points_diff: float = 0
    last5_goals_for_diff: float = 0
    last5_goals_against_diff: float = 0
    last5_goal_diff_diff: float = 0
    last5_win_rate_diff: float = 0

    last10_points_diff: float = 0
    last10_goals_for_diff: float = 0
    last10_goals_against_diff: float = 0
    last10_goal_diff_diff: float = 0
    last10_win_rate_diff: float = 0


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

RESULT_NAMES = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win"
}


def build_feature_frame(request: MatchFeatures) -> pd.DataFrame:
    raw = request.model_dump()

    # Remove non-feature fields that are not used by model
    raw.pop("home_team", None)
    raw.pop("away_team", None)

    # Create all expected columns
    row = {}

    for col in FEATURE_COLS:
        if col in raw:
            row[col] = raw[col]
        else:
            if col in NUMERIC_COLS:
                row[col] = 0
            else:
                row[col] = "Unknown"

    # Ensure categorical fields exist
    for col in CATEGORICAL_COLS:
        if col not in row or row[col] is None:
            row[col] = "Unknown"

    # Ensure numeric fields are numeric
    for col in NUMERIC_COLS:
        if col in row:
            try:
                row[col] = float(row[col])
            except Exception:
                row[col] = 0.0

    return pd.DataFrame([row])[FEATURE_COLS]


def confidence_label(confidence: float) -> str:
    if confidence >= 0.60:
        return "High"
    if confidence >= 0.45:
        return "Medium"
    return "Low"


def generate_report(home_team: str, away_team: str, predicted_result: str, confidence: float, probabilities: Dict[str, float]) -> str:
    return f"""
ScoutIQ Match Prediction Report

Match:
{home_team} vs {away_team}

Predicted Result:
{predicted_result}

Confidence:
{confidence_label(confidence)} ({confidence:.2%})

Probabilities:
- {home_team} Win: {probabilities.get("H", 0):.2%}
- Draw: {probabilities.get("D", 0):.2%}
- {away_team} Win: {probabilities.get("A", 0):.2%}

Disclaimer:
This prediction is based on historical data and machine learning.
It is intended for sports analytics and educational use only.
It does not guarantee the actual match result.
""".strip()


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "ScoutIQ Football Prediction API is running.",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": metadata.get("best_model_name", "unknown"),
        "classes": list(label_encoder.classes_),
        "feature_count": len(FEATURE_COLS),
        "test_metrics": metadata.get("test_metrics", {})
    }


@app.post("/predict")
def predict_match(request: MatchFeatures):
    try:
        X = build_feature_frame(request)

        proba = model.predict_proba(X)[0]
        pred_index = int(proba.argmax())
        pred_code = label_encoder.inverse_transform([pred_index])[0]
        predicted_result = RESULT_NAMES.get(pred_code, pred_code)

        probabilities = {
            code: float(prob)
            for code, prob in zip(label_encoder.classes_, proba)
        }

        confidence = float(max(proba))

        response = {
            "home_team": request.home_team,
            "away_team": request.away_team,
            "predicted_code": pred_code,
            "predicted_result": predicted_result,
            "confidence": confidence,
            "confidence_label": confidence_label(confidence),
            "probabilities": {
                "home_win": probabilities.get("H", 0),
                "draw": probabilities.get("D", 0),
                "away_win": probabilities.get("A", 0)
            },
            "raw_probabilities": probabilities,
            "report": generate_report(
                request.home_team,
                request.away_team,
                predicted_result,
                confidence,
                probabilities
            )
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
