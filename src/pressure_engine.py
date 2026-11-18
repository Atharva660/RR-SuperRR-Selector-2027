"""
Audited Dynamic Pressure Situation Engine
RR SuperRR Selector Season 2 - Independent Model Audit & Refinement

Key Fixes & Methodological Principles:
1. No Data Leakage: Pre-delivery match state computed via vectorised (cumsum - current_val).
   Zero cross-match or cross-innings shift leakage.
2. Separate Physics for 1st vs 2nd Innings:
   - 2nd Innings: Real-time RRR, CRR leverage ratio, wickets lost, balls urgency.
   - 1st Innings: Collapse pressure (wickets lost early), scoring deficit vs par (8.0 RPO), death acceleration.
3. Bowler Pressure Independence: Evaluates containment difficulty, target proximity, and opposition resource advantage.
4. Continuous Pressure Score (0-100) + Categorical Bins (Low, Moderate, High, Extreme).
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def classify_innings_phase(over_id: pd.Series) -> pd.Series:
    """1 = Powerplay (1-6), 2 = Middle (7-15), 3 = Death (16-20)."""
    phase = pd.Series(2, index=over_id.index)
    phase[over_id <= 6] = 1
    phase[over_id >= 16] = 3
    return phase


def compute_pressure_scores_audited(bbb: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ball-by-ball pressure intensity scores without data leakage.
    Returns DataFrame with batting_pressure_score and bowling_pressure_score (0-100).
    """
    df = bbb.sort_values(["match_id", "innings", "innings_ball_num"]).copy()

    # 1. Exact Pre-Delivery Cumulative State (strictly within match_id + innings)
    # Using cumsum() - current_val guarantees ZERO boundary leakage
    grp = df.groupby(["match_id", "innings"])
    df["cum_runs"] = grp["total_delivery_runs"].cumsum() - df["total_delivery_runs"]
    df["cum_wickets"] = grp["is_wicket"].cumsum() - df["is_wicket"].astype(int)
    df["cum_legal_balls"] = grp["bowler_legal_ball"].cumsum() - df["bowler_legal_ball"].astype(int)

    # 2. Innings 1 Final Scores for Second Innings Target
    inn1_totals = df[df["innings"] == 1].groupby("match_id").agg(
        inn1_final_runs=("total_delivery_runs", "sum"),
        inn1_final_wickets=("is_wicket", "sum"),
        inn1_final_legal_balls=("bowler_legal_ball", "sum")
    ).reset_index()

    # Merge match metadata only if not already present
    match_cols = ["revised_target", "second_inn_revised_overs"]
    cols_to_merge = [c for c in match_cols if c in matches.columns and c not in df.columns]
    if cols_to_merge:
        df = df.merge(matches[["match_id"] + cols_to_merge].drop_duplicates("match_id"), on="match_id", how="left")
    
    # Merge inn1 totals if not present
    if "inn1_final_runs" not in df.columns:
        df = df.merge(inn1_totals, on="match_id", how="left")

    # Set chase target (inn1_runs + 1, or revised target for DLS)
    df["chase_target"] = np.where(
        (df["innings"] == 2) & df["revised_target"].notna(),
        df["revised_target"],
        df["inn1_final_runs"] + 1
    )

    # Max overs and legal balls
    df["max_overs"] = 20.0
    if "second_inn_revised_overs" in df.columns:
        mask_rev = (df["innings"] == 2) & df["second_inn_revised_overs"].notna()
        df.loc[mask_rev, "max_overs"] = df.loc[mask_rev, "second_inn_revised_overs"]
    max_legal_balls = df["max_overs"] * 6.0

    # Balls and Wickets remaining
    df["balls_remaining"] = (max_legal_balls - df["cum_legal_balls"]).clip(lower=0.0)
    df["wickets_remaining"] = (10 - df["cum_wickets"]).clip(lower=0, upper=10)
    df["wickets_lost_ratio"] = (df["cum_wickets"] / 10.0).clip(0, 1)
    df["innings_phase"] = classify_innings_phase(df["overid"])
    phase_factor = (df["innings_phase"] - 1.0) / 2.0  # 0.0, 0.5, 1.0

    # -------------------------------------------------------------
    # A. 2nd Innings Pressure (Chase Dynamics)
    # -------------------------------------------------------------
    overs_remaining = (df["balls_remaining"] / 6.0).clip(lower=0.1)
    df["runs_required"] = (df["chase_target"] - df["cum_runs"]).clip(lower=0)
    df["rrr"] = df["runs_required"] / overs_remaining

    overs_elapsed = (df["cum_legal_balls"] / 6.0).clip(lower=0.1)
    df["crr"] = df["cum_runs"] / overs_elapsed

    # Leverage Ratio: RRR / CRR
    leverage = (df["rrr"] / df["crr"].clip(lower=1.0)).clip(upper=4.0)
    rrr_pressure = ((df["rrr"] - 6.0) / 10.0).clip(0, 1)  # 6 RPO = 0, 16 RPO = 1.0
    balls_urgency = ((1.0 - df["balls_remaining"] / max_legal_balls).clip(0, 1)) ** 1.5

    p_bat_chase = (
        0.30 * rrr_pressure +
        0.25 * df["wickets_lost_ratio"] +
        0.20 * balls_urgency +
        0.15 * (leverage / 4.0) +
        0.10 * phase_factor
    ).clip(0, 1) * 100.0

    # -------------------------------------------------------------
    # B. 1st Innings Pressure (Setting Total & Collapse Defense)
    # -------------------------------------------------------------
    par_t20_crr = 8.2  # IPL empirical scoring par
    scoring_deficit = ((par_t20_crr - df["crr"]) / par_t20_crr).clip(0, 1)
    # Early/Middle collapse weighting (losing 3+ wickets in overs 1-12 is severe pressure)
    collapse_urgency = (df["wickets_lost_ratio"] * (1.0 - df["cum_legal_balls"] / max_legal_balls)).clip(0, 1)

    p_bat_first = (
        0.35 * collapse_urgency +
        0.25 * scoring_deficit +
        0.25 * balls_urgency +
        0.15 * phase_factor
    ).clip(0, 1) * 100.0

    # Assign Batting Pressure based on innings
    df["batting_pressure_score"] = np.where(df["innings"] == 2, p_bat_chase, p_bat_first)

    # -------------------------------------------------------------
    # C. Bowling Pressure
    # -------------------------------------------------------------
    # 2nd Innings Bowling Pressure: Opposition close to target with wickets in hand
    target_proximity = (df["cum_runs"] / df["chase_target"].clip(lower=1)).clip(0, 1)
    opp_wickets_in_hand = (df["wickets_remaining"] / 10.0)
    rrr_defense_pressure = ((12.0 - df["rrr"]) / 8.0).clip(0, 1)  # if RRR is under 12, bowler is under pressure

    p_bowl_chase = (
        0.30 * rrr_defense_pressure +
        0.25 * target_proximity +
        0.20 * opp_wickets_in_hand +
        0.15 * balls_urgency +
        0.10 * phase_factor
    ).clip(0, 1) * 100.0

    # 1st Innings Bowling Pressure: Opposition scoring above par with wickets in hand
    scoring_surplus = ((df["crr"] - par_t20_crr) / par_t20_crr).clip(0, 1)
    p_bowl_first = (
        0.35 * scoring_surplus +
        0.30 * opp_wickets_in_hand +
        0.20 * balls_urgency +
        0.15 * phase_factor
    ).clip(0, 1) * 100.0

    df["bowling_pressure_score"] = np.where(df["innings"] == 2, p_bowl_chase, p_bowl_first)

    # Super Overs: High baseline pressure
    df.loc[df["is_super_over"], "batting_pressure_score"] = 80.0
    df.loc[df["is_super_over"], "bowling_pressure_score"] = 80.0

    # Round scores
    df["batting_pressure_score"] = df["batting_pressure_score"].round(2)
    df["bowling_pressure_score"] = df["bowling_pressure_score"].round(2)

    # Pressure categories
    df["batting_pressure_cat"] = pd.cut(
        df["batting_pressure_score"],
        bins=[-0.1, 25.0, 50.0, 75.0, 100.1],
        labels=["Low", "Moderate", "High", "Extreme"]
    )
    df["bowling_pressure_cat"] = pd.cut(
        df["bowling_pressure_score"],
        bins=[-0.1, 25.0, 50.0, 75.0, 100.1],
        labels=["Low", "Moderate", "High", "Extreme"]
    )

    logger.info(f"Audited pressure engine complete. High/Extreme pressure deliveries: "
                f"{(df['batting_pressure_score'] >= 50.0).sum():,}")
    return df
