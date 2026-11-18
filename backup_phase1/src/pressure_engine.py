"""
Pressure Situation Engine - RR SuperRR Selector Season 2
Dynamic ball-by-ball pressure identification system.
Core innovation of the project.
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Pressure score weights (domain-informed, validated via sensitivity analysis)
BATTING_PRESSURE_WEIGHTS = {
    "rrr_ratio": 0.30,         # Required run rate vs current run rate
    "wickets_lost_ratio": 0.25, # Proportion of wickets lost
    "balls_remaining_norm": 0.20,  # How late in innings (inverse)
    "runs_required_norm": 0.15,  # Runs still needed (normalised)
    "innings_stage": 0.10,     # Powerplay/middle/death
}

BOWLING_PRESSURE_WEIGHTS = {
    "rrr_ratio": 0.30,
    "wickets_remaining_ratio": 0.25,
    "balls_remaining_norm": 0.20,
    "target_proximity": 0.15,
    "innings_stage": 0.10,
}


def compute_innings_totals(bbb: pd.DataFrame) -> pd.DataFrame:
    """
    For each match+innings compute cumulative totals at each delivery point.
    Returns bbb with added columns:
    - cumulative_runs: runs scored so far in this innings (before this delivery)
    - cumulative_wickets: wickets fallen so far
    - cumulative_balls: legal balls faced so far
    """
    # Sort for cumulative calculations
    bbb = bbb.sort_values(["match_id", "innings", "innings_ball_num"]).copy()

    # Cumulative batter runs (shift by 1 so we have state BEFORE the delivery)
    bbb["cum_runs"] = bbb.groupby(["match_id", "innings"])["total_runs"].cumsum().shift(1).fillna(0)

    # Cumulative wickets fallen (before this delivery)
    bbb["cum_wickets"] = bbb.groupby(["match_id", "innings"])["is_wicket"].cumsum().shift(1).fillna(0)

    # Cumulative legal balls
    bbb["cum_legal_balls"] = bbb.groupby(["match_id", "innings"])["is_legal_ball"].cumsum().shift(1).fillna(0)

    return bbb


def compute_innings_final_totals(bbb: pd.DataFrame) -> pd.DataFrame:
    """Compute first innings final score for each match (for second innings target)."""
    # First innings final score
    inn1 = bbb[bbb["innings"] == 1].groupby("match_id").agg(
        inn1_total_runs=("total_runs", "sum"),
        inn1_wickets=("is_wicket", "sum"),
        inn1_legal_balls=("is_legal_ball", "sum"),
    ).reset_index()
    return inn1


def classify_innings_stage(over_id: pd.Series) -> pd.Series:
    """
    Classify over into phase:
    1 = Powerplay (1-6)
    2 = Middle (7-15)
    3 = Death (16-20)
    """
    stage = pd.Series(2, index=over_id.index)  # default middle
    stage[over_id <= 6] = 1
    stage[over_id >= 16] = 3
    return stage


def compute_pressure_scores(bbb: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    Main pressure computation function.
    Computes a Pressure Intensity Score (0-100) for every delivery.

    For 2nd innings (chase): uses required run rate, wickets, balls remaining
    For 1st innings: uses wickets lost, scoring rate vs expected, innings stage

    Returns bbb with pressure columns added.
    """
    logger.info("Computing innings cumulative totals...")
    bbb = compute_innings_totals(bbb)

    logger.info("Computing first-innings final totals...")
    inn1_totals = compute_innings_final_totals(bbb)

    # Merge match info for second innings target (only if not already present)
    for col in ["revised_target", "second_inn_revised_overs"]:
        if col not in bbb.columns:
            bbb = bbb.merge(
                matches[["match_id", col]].drop_duplicates("match_id"),
                on="match_id", how="left"
            )

    # Merge first innings total
    bbb = bbb.merge(inn1_totals, on="match_id", how="left")

    # For 2nd innings: target = inn1_total_runs + 1 (or revised target)
    bbb["chase_target"] = np.where(
        bbb["revised_target"].notna() & (bbb["innings"] == 2),
        bbb["revised_target"],
        bbb["inn1_total_runs"] + 1
    )

    # Total overs in innings (20 by default, or revised)
    bbb["max_overs"] = 20
    bbb.loc[(bbb["innings"] == 2) & bbb["second_inn_revised_overs"].notna(), "max_overs"] = \
        bbb.loc[(bbb["innings"] == 2) & bbb["second_inn_revised_overs"].notna(), "second_inn_revised_overs"]

    max_legal_balls = bbb["max_overs"] * 6

    # Balls remaining (before this delivery)
    bbb["balls_remaining"] = (max_legal_balls - bbb["cum_legal_balls"]).clip(lower=0)

    # Runs remaining (for chase)
    bbb["runs_required"] = (bbb["chase_target"] - bbb["cum_runs"]).clip(lower=0)

    # Wickets remaining
    bbb["wickets_remaining"] = (10 - bbb["cum_wickets"]).clip(lower=0, upper=10)

    # Innings stage
    bbb["innings_stage"] = classify_innings_stage(bbb["overid"])

    # ---- SECOND INNINGS PRESSURE (Chase) ----
    overs_remaining = (bbb["balls_remaining"] / 6).clip(lower=0.01)
    rrr = (bbb["runs_required"] / overs_remaining)  # required run rate

    # Current run rate
    overs_faced = (bbb["cum_legal_balls"] / 6).clip(lower=0.01)
    crr = (bbb["cum_runs"] / overs_faced)

    # RRR ratio: how much harder is the required rate vs what they've been scoring
    rrr_ratio = (rrr / crr.clip(lower=0.1)).clip(upper=5)
    # Normalise: 1=par, >1=harder, <1=easier; map to 0-1 pressure
    rrr_pressure = ((rrr_ratio - 0.5) / 4.5).clip(0, 1)  # maps 0.5..5 -> 0..1

    # Wickets lost pressure
    wickets_lost = 10 - bbb["wickets_remaining"]
    wickets_lost_ratio = (wickets_lost / 10).clip(0, 1)

    # Balls remaining pressure (more pressure when fewer balls left, non-linearly)
    balls_norm = (1 - bbb["balls_remaining"] / max_legal_balls).clip(0, 1)
    balls_pressure = balls_norm ** 1.5  # accelerate pressure near end

    # Runs required pressure (high required score = high pressure)
    runs_req_norm = (bbb["runs_required"] / bbb["chase_target"].clip(lower=1)).clip(0, 1)

    # Innings stage pressure
    stage_pressure = (bbb["innings_stage"] - 1) / 2  # 0, 0.5, 1 for pp/mid/death

    # Combined 2nd innings batting pressure
    p2 = (
        BATTING_PRESSURE_WEIGHTS["rrr_ratio"] * rrr_pressure +
        BATTING_PRESSURE_WEIGHTS["wickets_lost_ratio"] * wickets_lost_ratio +
        BATTING_PRESSURE_WEIGHTS["balls_remaining_norm"] * balls_pressure +
        BATTING_PRESSURE_WEIGHTS["runs_required_norm"] * runs_req_norm +
        BATTING_PRESSURE_WEIGHTS["innings_stage"] * stage_pressure
    ).clip(0, 1) * 100

    # ---- FIRST INNINGS PRESSURE ----
    # No RRR available. Use:
    # - Wickets lost (collapse risk)
    # - Innings stage (powerplay/death)
    # - Scoring rate vs expected T20 average (~8 rpo = 48 per 6 overs)
    expected_crr = 8.0  # expected scoring rate in T20
    crr_deficit = ((expected_crr - crr) / expected_crr).clip(-1, 1)
    crr_deficit_pressure = crr_deficit.clip(0, 1)  # only if scoring below expected

    # For 1st innings, wickets lost is about collapse risk
    p1 = (
        0.35 * wickets_lost_ratio +
        0.30 * crr_deficit_pressure +
        0.25 * balls_pressure +
        0.10 * stage_pressure
    ).clip(0, 1) * 100

    # ---- BOWLING PRESSURE (2nd innings defender) ----
    # Bowler is under pressure in 2nd innings when opposition is close to target
    target_proximity = (bbb["cum_runs"] / bbb["chase_target"].clip(lower=1)).clip(0, 1)
    target_proximity_sq = target_proximity ** 2  # nonlinear: more pressure when close

    # Wickets remaining (for batting team = bowling team's concern)
    # Fewer wickets remaining = less pressure on bowler
    bowl_wicket_ratio = (bbb["wickets_remaining"] / 10)  # more wickets = more bowling pressure

    p_bowl_2nd = (
        BOWLING_PRESSURE_WEIGHTS["rrr_ratio"] * rrr_pressure +
        BOWLING_PRESSURE_WEIGHTS["wickets_remaining_ratio"] * bowl_wicket_ratio +
        BOWLING_PRESSURE_WEIGHTS["balls_remaining_norm"] * balls_pressure +
        BOWLING_PRESSURE_WEIGHTS["target_proximity"] * target_proximity_sq +
        BOWLING_PRESSURE_WEIGHTS["innings_stage"] * stage_pressure
    ).clip(0, 1) * 100

    # 1st innings bowling pressure: wickets not falling, opposition scoring fast
    crr_excess = ((crr - expected_crr) / expected_crr).clip(-1, 1).clip(0, 1)
    p_bowl_1st = (
        0.35 * crr_excess +
        0.30 * (1 - wickets_lost_ratio) +  # wickets not falling = pressure on bowler
        0.25 * balls_pressure +
        0.10 * stage_pressure
    ).clip(0, 1) * 100

    # Assign pressure scores based on innings
    bbb["batting_pressure_score"] = np.where(bbb["innings"] == 2, p2, p1)
    bbb["bowling_pressure_score"] = np.where(bbb["innings"] == 2, p_bowl_2nd, p_bowl_1st)

    # Zero out super-over deliveries from pressure (treat separately)
    bbb.loc[bbb["is_super_over"], "batting_pressure_score"] = 50.0
    bbb.loc[bbb["is_super_over"], "bowling_pressure_score"] = 80.0

    # ---- PRESSURE CATEGORIES ----
    def categorise(score: pd.Series) -> pd.Series:
        cats = pd.cut(
            score,
            bins=[-0.1, 25, 50, 75, 100.1],
            labels=["Low", "Moderate", "High", "Extreme"]
        )
        return cats

    bbb["batting_pressure_cat"] = categorise(bbb["batting_pressure_score"])
    bbb["bowling_pressure_cat"] = categorise(bbb["bowling_pressure_score"])

    # Store RRR and CRR for reference
    bbb["rrr"] = rrr.where(bbb["innings"] == 2, np.nan)
    bbb["crr"] = crr
    bbb["balls_remaining"] = bbb["balls_remaining"]

    logger.info(f"Pressure scores computed. High/Extreme batting: "
                f"{(bbb['batting_pressure_score'] >= 50).sum():,} deliveries")

    return bbb
