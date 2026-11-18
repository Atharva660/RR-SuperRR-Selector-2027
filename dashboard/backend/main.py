
from pathlib import Path
import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(
    title="SupeRR Selector - Pressure Intelligence Engine API"
)

# Works on Windows, Linux, Render, and local development.
BASE = Path(__file__).resolve().parents[2]
RANKINGS_DIR = BASE / "outputs" / "rankings"
REPORTS_DIR = BASE / "outputs" / "reports"
FRONTEND_DIR = BASE / "dashboard" / "frontend"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Public read-only dashboard API
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def read_csv(filename):
    path = RANKINGS_DIR / filename

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Required ranking file not found: {filename}"
        )

    try:
        df = pd.read_csv(path)
        return json.loads(df.to_json(orient="records"))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read {filename}"
        ) from exc


def read_json(filename):
    path = REPORTS_DIR / filename

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Required report file not found: {filename}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read {filename}"
        ) from exc


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/overview")
def get_overview():
    return {
        "total_deliveries": 200999,
        "ipl_deliveries": 87687,
        "t20i_deliveries": 106327,
        "regional_deliveries": 6985,
        "total_matches": 883,
        "played_matches": 867,
        "abandoned_matches": 16,
        "ranking_stability": 0.95,
        "model_validation_auc": 0.5829,
        "bootstrap_iterations": 150
    }


@app.get("/api/batters")
def get_batters():
    return read_csv("batters_final_ranked.csv")


@app.get("/api/bowlers")
def get_bowlers():
    return read_csv("bowlers_final_ranked.csv")


@app.get("/api/reports/ml")
def get_ml_reports():
    return {
        "evaluation": read_json("model_evaluation_report.json"),
        "explainability": read_json("model_explainability_report.json")
    }


@app.get("/api/reports/stability")
def get_stability_report():
    return read_json("ranking_stability_report.json")


@app.get("/api/player/{player_id}")
def get_player_profile(player_id: int):
    batters = get_batters()
    bowlers = get_bowlers()

    batting = next(
        (p for p in batters
         if str(p.get("striker_id")) == str(player_id)),
        None
    )

    bowling = next(
        (p for p in bowlers
         if str(p.get("bowler_id")) == str(player_id)),
        None
    )

    if batting is None and bowling is None:
        raise HTTPException(
            status_code=404,
            detail="Player not found"
        )

    return {
        "player_id": player_id,
        "batting": batting,
        "bowling": bowling
    }


@app.get("/")
def serve_dashboard():
    index = FRONTEND_DIR / "index.html"

    if not index.is_file():
        return {
            "message": "SupeRR Selector API",
            "docs": "/docs"
        }

    return FileResponse(index)
