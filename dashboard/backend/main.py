
from pathlib import Path
import json
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="SupeRR Selector - Pressure Intelligence Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Project root: RR-SuperRR-Selector-2027
BASE = Path(__file__).resolve().parents[2]

RANKINGS_DIR = BASE / "outputs" / "rankings"
REPORTS_DIR = BASE / "outputs" / "reports"
FRONTEND_DIR = BASE / "dashboard" / "frontend"


@app.get("/")
def serve_index():
    index = FRONTEND_DIR / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="Dashboard index.html not found")
    return FileResponse(index)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/overview")
def get_overview():
    return {
        "total_deliveries": 200999,
        "ipl_deliveries": 87687,
        "t20i_deliveries": 106327,
        "total_matches": 883,
        "ipl_matches": 371,
        "high_pressure_deliveries": 24503,
        "extreme_pressure_deliveries": 2109,
        "eligible_batters": 78,
        "eligible_bowlers": 119,
        "avg_pressure_sr": 154.2,
        "avg_pressure_economy": 9.42,
        "model_validation_auc": 0.5846,
        "ranking_stability": 0.80
    }


def read_ranking(filename):
    path = RANKINGS_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Ranking file not found: {filename}")
    return pd.read_csv(path).to_dict(orient="records")


@app.get("/api/batters")
def get_batters():
    return read_ranking("batters_final_ranked.csv")


@app.get("/api/bowlers")
def get_bowlers():
    return read_ranking("bowlers_final_ranked.csv")


def read_report(filename):
    path = REPORTS_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Report not found: {filename}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/reports/ml")
def get_ml_reports():
    return {
        "evaluation": read_report("model_evaluation_report.json"),
        "explainability": read_report("model_explainability_report.json")
    }


@app.get("/api/reports/stability")
def get_stability_report():
    return read_report("ranking_stability_report.json")


@app.get("/api/player/{player_id}")
def get_player_profile(player_id: int):
    batters = get_batters()
    bowlers = get_bowlers()

    batting = next(
        (p for p in batters if str(p.get("striker_id")) == str(player_id)), None
    )
    bowling = next(
        (p for p in bowlers if str(p.get("bowler_id")) == str(player_id)), None
    )

    if batting is None and bowling is None:
        raise HTTPException(status_code=404, detail="Player not found")

    return {
        "player_id": player_id,
        "batting": batting,
        "bowling": bowling
    }
