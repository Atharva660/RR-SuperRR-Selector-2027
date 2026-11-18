"""
Feature Engineering Module - RR SuperRR Selector Season 2
Builds player-level pressure and normal performance features.
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

PRESSURE_THRESHOLD = 50.0   # Deliveries with pressure score >= this are "pressure" deliveries
MIN_PRESSURE_BALLS_BATTER = 50  # Minimum legal balls faced under pressure to be eligible
MIN_PRESSURE_OVERS_BOWLER = 20  # Minimum overs bowled under pressure to be eligible
MIN_MATCHES_BATTER = 10
MIN_MATCHES_BOWLER = 10


def build_batting_features(bbb: pd.DataFrame, name_map: pd.DataFrame,
                           ipl_only: bool = False) -> pd.DataFrame:
    """
    Build batting features for every player who faced sufficient deliveries.
    Returns player-level DataFrame with pressure and normal features.
    """
    # Filter to relevant competitions
    if ipl_only:
        df = bbb[bbb["is_ipl"]].copy()
    else:
        df = bbb[bbb["is_ipl"] | bbb["is_t20i"]].copy()

    # Exclude super overs
    df = df[~df["is_super_over"]]

    # Focus on batter (striker)
    # Legal deliveries only for balls faced
    df_legal = df[df["is_legal_ball"]].copy()

    # Pressure flag
    df_legal["is_pressure"] = df_legal["batting_pressure_score"] >= PRESSURE_THRESHOLD

    # --- Per-delivery batter metrics ---
    # For each delivery: runs, is_dismissed, is_boundary (4 or 6)
    df_legal["is_dismissed"] = df_legal.apply(
        lambda r: r["is_wicket"] and r["dismissed_player_id"] == r["striker_id"], axis=1
    )
    df_legal["is_boundary"] = df_legal["is_four"] | df_legal["is_six"]
    df_legal["is_dot"] = df_legal["batter_runs"] == 0

    # ---- Normal (all) stats ----
    all_stats = _aggregate_batting(df_legal, "all")

    # ---- Pressure stats ----
    press_stats = _aggregate_batting(df_legal[df_legal["is_pressure"]], "pressure")

    # ---- Match-level consistency ----
    consistency = _batting_consistency(df_legal, "pressure")

    # Merge
    batting = all_stats.merge(press_stats, on="striker_id", how="inner")
    batting = batting.merge(consistency, on="striker_id", how="left")

    # Merge player names
    batting = batting.merge(name_map, left_on="striker_id", right_on="player_id", how="left")

    # Compute differentials
    batting["sr_differential"] = batting["pressure_strike_rate"] - batting["all_strike_rate"]
    batting["avg_differential"] = batting["pressure_avg"] - batting["all_avg"]

    # Minimum eligibility filter
    batting = batting[
        (batting["pressure_balls"] >= MIN_PRESSURE_BALLS_BATTER) &
        (batting["all_matches"] >= MIN_MATCHES_BATTER)
    ]

    logger.info(f"Eligible batters: {len(batting)}")
    return batting


def _aggregate_batting(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate batting stats per striker_id."""
    agg = df.groupby("striker_id").agg(
        total_runs=("batter_runs", "sum"),
        total_balls=("is_legal_ball", "sum"),
        total_dismissals=("is_dismissed", "sum"),
        total_boundaries=("is_boundary", "sum"),
        total_dots=("is_dot", "sum"),
        total_matches=("match_id", "nunique"),
        total_innings=("match_id", lambda x: x.nunique()),  # innings where faced at least 1 ball
    ).reset_index()

    # Strike rate
    agg["strike_rate"] = (agg["total_runs"] / agg["total_balls"] * 100).round(2)

    # Batting average (runs per dismissal)
    agg["avg"] = (agg["total_runs"] / agg["total_dismissals"].clip(lower=1)).round(2)
    agg.loc[agg["total_dismissals"] == 0, "avg"] = agg.loc[agg["total_dismissals"] == 0, "total_runs"]

    # Boundary %
    agg["boundary_pct"] = (agg["total_boundaries"] / agg["total_balls"] * 100).round(2)

    # Dot %
    agg["dot_pct"] = (agg["total_dots"] / agg["total_balls"] * 100).round(2)

    # Rename columns with prefix
    rename_map = {c: f"{prefix}_{c}" for c in agg.columns if c != "striker_id"}
    rename_map["total_matches"] = f"{prefix}_matches"
    rename_map["total_innings"] = f"{prefix}_innings"
    rename_map["total_runs"] = f"{prefix}_runs"
    rename_map["total_balls"] = f"{prefix}_balls"
    rename_map["total_dismissals"] = f"{prefix}_dismissals"
    rename_map["total_boundaries"] = f"{prefix}_boundaries"
    rename_map["total_dots"] = f"{prefix}_dots"
    rename_map["strike_rate"] = f"{prefix}_strike_rate"
    rename_map["avg"] = f"{prefix}_avg"
    rename_map["boundary_pct"] = f"{prefix}_boundary_pct"
    rename_map["dot_pct"] = f"{prefix}_dot_pct"
    agg = agg.rename(columns=rename_map)

    return agg


def _batting_consistency(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute match-level consistency: std dev and coefficient of variation of runs."""
    press_df = df[df["batting_pressure_score"] >= PRESSURE_THRESHOLD].copy()
    match_runs = press_df.groupby(["striker_id", "match_id"])["batter_runs"].sum().reset_index()
    match_runs.columns = ["striker_id", "match_id", "match_pressure_runs"]

    cons = match_runs.groupby("striker_id").agg(
        pressure_match_std=("match_pressure_runs", "std"),
        pressure_match_mean=("match_pressure_runs", "mean"),
        pressure_matches_count=("match_id", "count"),
        pressure_above_20=("match_pressure_runs", lambda x: (x >= 20).sum()),
    ).reset_index()

    cons["pressure_match_std"] = cons["pressure_match_std"].fillna(0)
    cons["cv"] = (cons["pressure_match_std"] / cons["pressure_match_mean"].clip(lower=0.1)).round(3)
    cons["consistency_score"] = (1 / (1 + cons["cv"])).round(4)  # higher = more consistent
    cons["pressure_success_rate"] = (cons["pressure_above_20"] / cons["pressure_matches_count"]).round(4)

    return cons[["striker_id", "pressure_match_std", "pressure_match_mean",
                 "cv", "consistency_score", "pressure_success_rate", "pressure_above_20"]]


def build_bowling_features(bbb: pd.DataFrame, name_map: pd.DataFrame,
                           ipl_only: bool = False) -> pd.DataFrame:
    """Build bowling features for every eligible bowler."""
    if ipl_only:
        df = bbb[bbb["is_ipl"]].copy()
    else:
        df = bbb[bbb["is_ipl"] | bbb["is_t20i"]].copy()

    df = df[~df["is_super_over"]]

    # Bowling pressure flag
    df["is_pressure"] = df["bowling_pressure_score"] >= PRESSURE_THRESHOLD

    df["is_boundary"] = df["is_four"] | df["is_six"]
    df["is_dot_bowl"] = (df["total_runs"] == 0)

    # All stats
    all_stats = _aggregate_bowling(df, "all")

    # Pressure stats
    press_stats = _aggregate_bowling(df[df["is_pressure"]], "pressure")

    # Consistency
    consistency = _bowling_consistency(df)

    bowling = all_stats.merge(press_stats, on="bowler_id", how="inner")
    bowling = bowling.merge(consistency, on="bowler_id", how="left")
    bowling = bowling.merge(name_map, left_on="bowler_id", right_on="player_id", how="left")

    # Differentials (lower economy under pressure = better -> negative differential is good)
    bowling["eco_differential"] = bowling["pressure_economy"] - bowling["all_economy"]

    # Minimum eligibility
    bowling = bowling[
        (bowling["pressure_legal_balls"] >= MIN_PRESSURE_OVERS_BOWLER * 6) &
        (bowling["all_matches"] >= MIN_MATCHES_BOWLER)
    ]

    logger.info(f"Eligible bowlers: {len(bowling)}")
    return bowling


def _aggregate_bowling(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Aggregate bowling stats per bowler_id."""
    agg = df.groupby("bowler_id").agg(
        total_runs_conceded=("total_runs", "sum"),
        total_legal_balls=("is_legal_ball", "sum"),
        total_wickets=("is_wicket", "sum"),
        total_boundaries=("is_boundary", "sum"),
        total_dots=("is_dot_bowl", "sum"),
        total_matches=("match_id", "nunique"),
    ).reset_index()

    # Overs
    agg["overs"] = (agg["total_legal_balls"] / 6).round(2)

    # Economy
    agg["economy"] = (agg["total_runs_conceded"] / agg["overs"].clip(lower=0.01)).round(2)

    # Bowling average
    agg["bowl_avg"] = (agg["total_runs_conceded"] / agg["total_wickets"].clip(lower=1)).round(2)

    # Strike rate
    agg["bowl_sr"] = (agg["total_legal_balls"] / agg["total_wickets"].clip(lower=1)).round(2)

    # Boundary rate
    agg["boundary_rate"] = (agg["total_boundaries"] / agg["total_legal_balls"].clip(lower=1) * 100).round(2)

    # Dot %
    agg["dot_pct"] = (agg["total_dots"] / agg["total_legal_balls"].clip(lower=1) * 100).round(2)

    rename_map = {
        "total_runs_conceded": f"{prefix}_runs_conceded",
        "total_legal_balls": f"{prefix}_legal_balls",
        "total_wickets": f"{prefix}_wickets",
        "total_boundaries": f"{prefix}_boundaries",
        "total_dots": f"{prefix}_dots",
        "total_matches": f"{prefix}_matches",
        "overs": f"{prefix}_overs",
        "economy": f"{prefix}_economy",
        "bowl_avg": f"{prefix}_bowl_avg",
        "bowl_sr": f"{prefix}_bowl_sr",
        "boundary_rate": f"{prefix}_boundary_rate",
        "dot_pct": f"{prefix}_dot_pct",
    }
    agg = agg.rename(columns=rename_map)
    return agg


def _bowling_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Match-level consistency for bowlers under pressure."""
    press_df = df[df["bowling_pressure_score"] >= PRESSURE_THRESHOLD].copy()
    match_eco = press_df.groupby(["bowler_id", "match_id"]).agg(
        match_runs=("total_runs", "sum"),
        match_balls=("is_legal_ball", "sum"),
        match_wickets=("is_wicket", "sum"),
    ).reset_index()
    match_eco["match_economy"] = match_eco["match_runs"] / (match_eco["match_balls"] / 6).clip(lower=0.01)
    match_eco["good_spell"] = (match_eco["match_economy"] <= 8) | (match_eco["match_wickets"] >= 2)

    cons = match_eco.groupby("bowler_id").agg(
        pressure_eco_std=("match_economy", "std"),
        pressure_eco_mean=("match_economy", "mean"),
        pressure_bowl_matches=("match_id", "count"),
        good_spells=("good_spell", "sum"),
    ).reset_index()

    cons["pressure_eco_std"] = cons["pressure_eco_std"].fillna(0)
    cons["eco_cv"] = (cons["pressure_eco_std"] / cons["pressure_eco_mean"].clip(lower=0.1)).round(3)
    cons["bowl_consistency_score"] = (1 / (1 + cons["eco_cv"])).round(4)
    cons["good_spell_rate"] = (cons["good_spells"] / cons["pressure_bowl_matches"]).round(4)

    return cons[["bowler_id", "pressure_eco_std", "eco_cv",
                 "bowl_consistency_score", "good_spell_rate", "pressure_bowl_matches"]]

