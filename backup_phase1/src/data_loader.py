"""
Data Loader Module - RR SuperRR Selector Season 2
Loads and merges all raw datasets.
"""
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

RAW_DATA_DIR = r"c:\Atharva\RR-SuperRR-Selector-2027\Data Pack"

IPL_COMP_IDS = [1358, 1715, 2193, 2725, 3298]  # IPL 2022-2026

def load_all_datasets() -> dict:
    """Load all 6 raw CSV files and return as a dict of DataFrames."""
    files = {
        "ball_by_ball": "ball_by_ball.csv",
        "matches": "matches.csv",
        "competitions": "competitions.csv",
        "teams": "teams.csv",
        "playing_eleven": "playing_eleven.csv",
        "player_mapping": "competition_team_player_mapping.csv",
    }
    datasets = {}
    for key, filename in files.items():
        path = os.path.join(RAW_DATA_DIR, filename)
        df = pd.read_csv(path)
        datasets[key] = df
        logger.info(f"Loaded {filename}: {df.shape}")
    return datasets


def build_master_dataset(datasets: dict) -> pd.DataFrame:
    """
    Merge ball_by_ball with matches, competitions to create a full delivery-level frame.
    """
    bbb = datasets["ball_by_ball"].copy()
    matches = datasets["matches"].copy()
    comps = datasets["competitions"].copy()

    # Merge match metadata onto ball-by-ball
    matches_slim = matches[[
        "match_id", "comp_id", "match_date", "stage",
        "home_team_id", "away_team_id",
        "toss_win_team_id", "toss_opted",
        "win_team_id", "result",
        "first_inn_revised_overs", "second_inn_revised_overs", "revised_target"
    ]]
    df = bbb.merge(matches_slim, on="match_id", how="left")

    # Merge competition metadata
    comps_slim = comps[["comp_id", "comp_name", "match_format", "season"]]
    df = df.merge(comps_slim, on="comp_id", how="left")

    # Classify competition type
    df["is_ipl"] = df["comp_id"].isin(IPL_COMP_IDS)
    df["is_t20i"] = df["match_format"] == "T20I"

    return df


def load_player_names(datasets: dict) -> pd.DataFrame:
    """Return unique player_id -> player_name mapping from competition_team_player_mapping."""
    pm = datasets["player_mapping"]
    # Deduplicate: take the most common name per player_id
    name_map = (
        pm.groupby("player_id")["player_name"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )
    return name_map

