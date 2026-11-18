"""
FastAPI Backend for SupeRR Selector Dashboard
Serves data for overview, batters, bowlers, pressure engine, ML models, player profiles, and recommendations.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import json
import os

app = FastAPI(title="SupeRR Selector - Pressure Intelligence Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = r"c:\Atharva\RR-SuperRR-Selector-2027"
RANKINGS_DIR = os.path.join(BASE, "outputs", "rankings")
REPORTS_DIR = os.path.join(BASE, "outputs", "reports")
FRONTEND_DIR = os.path.join(BASE, "dashboard", "frontend")

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

@app.get("/api/batters")
def get_batters():
    path = os.path.join(RANKINGS_DIR, "batters_final_ranked.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Batters rankings not found")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/api/bowlers")
def get_bowlers():
    path = os.path.join(RANKINGS_DIR, "bowlers_final_ranked.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Bowlers rankings not found")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

@app.get("/api/reports/ml")
def get_ml_reports():
    eval_path = os.path.join(REPORTS_DIR, "model_evaluation_report.json")
    exp_path = os.path.join(REPORTS_DIR, "model_explainability_report.json")
    
    with open(eval_path, "r") as f:
        ev = json.load(f)
    with open(exp_path, "r") as f:
        exp = json.load(f)
        
    return {"evaluation": ev, "explainability": exp}

@app.get("/api/reports/stability")
def get_stability_report():
    stab_path = os.path.join(REPORTS_DIR, "ranking_stability_report.json")
    with open(stab_path, "r") as f:
        st = json.load(f)
    return st

@app.get("/api/player/{player_id}")
def get_player_profile(player_id: int):
    bat_path = os.path.join(RANKINGS_DIR, "batters_final_ranked.csv")
    bowl_path = os.path.join(RANKINGS_DIR, "bowlers_final_ranked.csv")
    
    bat_df = pd.read_csv(bat_path)
    bowl_df = pd.read_csv(bowl_path)
    
    bat_match = bat_df[bat_df["striker_id"] == player_id]
    bowl_match = bowl_df[bowl_df["bowler_id"] == player_id]
    
    res = {"player_id": player_id}
    if not bat_match.empty:
        res["batting"] = bat_match.iloc[0].to_dict()
    if not bowl_match.empty:
        res["bowling"] = bowl_match.iloc[0].to_dict()
        
    if "batting" not in res and "bowling" not in res:
        raise HTTPException(status_code=404, detail="Player not found in rankings")
        
    return res

# Serve static frontend dashboard if index.html exists
if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

