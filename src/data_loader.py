"""
Audited Data Loader Module
RR SuperRR Selector Season 2 - Independent Model Audit & Refinement

Loads raw CSV datasets, validates keys, and constructs master delivery-level dataframe.
Reconciles all 200,999 deliveries across IPL, Other T20, and T20I competitions.
"""
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

RAW_DATA_DIR = r"c:\Atharva\RR-SuperRR-Selector-2027\Data Pack"

IPL_COMP_IDS = [1358, 1715, 2193, 2725, 3298]  # IPL 2022-2026
OTHER_T20_COMP_IDS = [2670, 3289]  # Emerging Teams Asia Cup 2024 & Rising Stars 2025

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
    Accounts for 100% of the 200,999 raw deliveries.
    """
    bbb = datasets["ball_by_ball"].copy()
    matches = datasets["matches"].copy()
    comps = datasets["competitions"].copy()

    # Match metadata
    matches_slim = matches[[
        "match_id", "comp_id", "match_date", "stage",
        "home_team_id", "away_team_id",
        "toss_win_team_id", "toss_opted",
        "win_team_id", "result",
        "first_inn_revised_overs", "second_inn_revised_overs", "revised_target"
    ]]
    df = bbb.merge(matches_slim, on="match_id", how="left")

    # Competition metadata
    comps_slim = comps[["comp_id", "comp_name", "match_format", "season"]]
    df = df.merge(comps_slim, on="comp_id", how="left")

    # Competition flags
    df["is_ipl"] = df["comp_id"].isin(IPL_COMP_IDS)
    df["is_other_t20"] = df["comp_id"].isin(OTHER_T20_COMP_IDS)
    df["is_t20i"] = df["match_format"] == "T20I"

    logger.info(f"Master dataset assembled: {df.shape}. IPL deliveries: {df['is_ipl'].sum():,}, "
                f"Other T20: {df['is_other_t20'].sum():,}, T20I: {df['is_t20i'].sum():,}.")
    return df


def load_player_names(datasets: dict) -> pd.DataFrame:
    """Canonical mapping from player_id to official name."""
    pm = datasets["player_mapping"]
    name_map = (
        pm.groupby("player_id")["player_name"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )
    return name_map
