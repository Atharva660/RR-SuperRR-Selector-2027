"""
RR Pressure Reliability Index (RR-PRI) & Player Ranking Module
Core ranking framework for the Rajasthan Royals challenge.
"""
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# --- RR-PRI Weights ---
# Determined through domain logic + sensitivity testing
# All components are directionally consistent (higher = better performance)
BATTING_PRI_WEIGHTS = {
    "pressure_sr_component": 0.25,     # Bayesian pressure strike rate (normalised)
    "pressure_avg_component": 0.20,    # Bayesian pressure average
    "sr_improvement": 0.15,            # Pressure SR vs normal SR (improvement)
    "consistency_component": 0.20,     # Match-level consistency
    "success_rate_component": 0.10,    # Fraction of matches with good pressure knock
    "sample_weight": 0.10,             # Sample size reliability factor
}

BOWLING_PRI_WEIGHTS = {
    "pressure_eco_component": 0.25,    # Bayesian pressure economy (inverted - lower=better)
    "pressure_wickets_component": 0.20, # Wickets per over under pressure
    "eco_improvement": 0.15,           # Pressure eco vs normal eco (improvement)
    "consistency_component": 0.20,     # Match-level consistency
    "good_spell_rate_component": 0.10, # Rate of effective spells
    "sample_weight": 0.10,             # Sample size reliability
}


def compute_batting_pri(batting: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RR-PRI for batters.
    Scale: 0 to 100.
    All components scaled to [0,1] before weighting.
    """
    df = batting.copy()
    scaler = MinMaxScaler()

    # 1. Pressure SR component (Bayesian adjusted)
    df["sr_comp"] = scaler.fit_transform(df[["bayesian_pressure_sr"]])

    # 2. Pressure average component
    df["avg_comp"] = scaler.fit_transform(df[["bayesian_pressure_avg"]])

    # 3. SR improvement (can be negative; clip to range)
    sr_imp = df["sr_differential"].clip(-50, 50)
    df["sr_imp_comp"] = (sr_imp - sr_imp.min()) / (sr_imp.max() - sr_imp.min() + 1e-6)

    # 4. Consistency component
    df["cons_comp"] = scaler.fit_transform(df[["consistency_score"]])

    # 5. Success rate
    df["success_comp"] = scaler.fit_transform(df[["pressure_success_rate"]])

    # 6. Sample size weight: log(balls) normalised
    max_log = np.log(df["pressure_balls"].max() + 1)
    df["sample_comp"] = np.log(df["pressure_balls"] + 1) / max_log

    # Weighted RR-PRI
    w = BATTING_PRI_WEIGHTS
    df["rr_pri"] = (
        w["pressure_sr_component"] * df["sr_comp"] +
        w["pressure_avg_component"] * df["avg_comp"] +
        w["sr_improvement"] * df["sr_imp_comp"] +
        w["consistency_component"] * df["cons_comp"] +
        w["success_rate_component"] * df["success_comp"] +
        w["sample_weight"] * df["sample_comp"]
    ) * 100

    df["rr_pri"] = df["rr_pri"].round(2)

    return df


def compute_bowling_pri(bowling: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RR-PRI for bowlers.
    Economy: LOWER is better, so invert the economy component.
    """
    df = bowling.copy()
    scaler = MinMaxScaler()

    # 1. Pressure economy (inverted: lower economy = higher score)
    eco_inv = 1 / (df["bayesian_pressure_eco"].clip(lower=0.1))
    df["eco_comp"] = scaler.fit_transform(eco_inv.values.reshape(-1, 1))

    # 2. Wickets per over under pressure
    df["wkts_per_over"] = df["pressure_wickets"] / df["pressure_overs"].clip(lower=0.1)
    df["wkts_comp"] = scaler.fit_transform(df[["wkts_per_over"]])

    # 3. Economy improvement: pressure_eco vs all_eco (negative = improvement)
    eco_imp = -df["eco_differential"].clip(-4, 4)  # negative diff = better -> positive imp
    df["eco_imp_comp"] = (eco_imp - eco_imp.min()) / (eco_imp.max() - eco_imp.min() + 1e-6)

    # 4. Consistency
    df["cons_comp"] = scaler.fit_transform(df[["bowl_consistency_score"]])

    # 5. Good spell rate
    df["spell_comp"] = scaler.fit_transform(df[["good_spell_rate"]])

    # 6. Sample size
    max_log = np.log(df["pressure_legal_balls"].max() + 1)
    df["sample_comp"] = np.log(df["pressure_legal_balls"] + 1) / max_log

    w = BOWLING_PRI_WEIGHTS
    df["rr_pri"] = (
        w["pressure_eco_component"] * df["eco_comp"] +
        w["pressure_wickets_component"] * df["wkts_comp"] +
        w["eco_improvement"] * df["eco_imp_comp"] +
        w["consistency_component"] * df["cons_comp"] +
        w["good_spell_rate_component"] * df["spell_comp"] +
        w["sample_weight"] * df["sample_comp"]
    ) * 100

    df["rr_pri"] = df["rr_pri"].round(2)

    return df


def rank_batters(batting: pd.DataFrame) -> pd.DataFrame:
    """Rank batters by RR-PRI descending."""
    ranked = batting.sort_values("rr_pri", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked


def rank_bowlers(bowling: pd.DataFrame) -> pd.DataFrame:
    """Rank bowlers by RR-PRI descending."""
    ranked = bowling.sort_values("rr_pri", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked


def generate_batter_leaderboard(
    ranked_batters: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """Generate formatted Top-N batter leaderboard."""
    cols = [
        "rank", "player_name", "striker_id",
        "all_matches", "pressure_innings",
        "pressure_runs", "pressure_balls",
        "pressure_strike_rate", "pressure_avg",
        "all_strike_rate", "all_avg",
        "sr_differential", "avg_differential",
        "bayesian_pressure_sr", "bayesian_pressure_avg",
        "consistency_score", "pressure_success_rate",
        "rr_pri",
    ]
    top = ranked_batters.head(top_n)
    # Only include columns that exist
    existing_cols = [c for c in cols if c in top.columns]
    return top[existing_cols].copy()


def generate_bowler_leaderboard(
    ranked_bowlers: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """Generate formatted Top-N bowler leaderboard."""
    cols = [
        "rank", "player_name", "bowler_id",
        "all_matches", "pressure_bowl_matches",
        "pressure_overs", "pressure_wickets",
        "pressure_economy", "pressure_bowl_avg", "pressure_bowl_sr",
        "pressure_dot_pct",
        "all_economy", "all_bowl_avg",
        "eco_differential",
        "bayesian_pressure_eco",
        "bowl_consistency_score", "good_spell_rate",
        "rr_pri",
    ]
    top = ranked_bowlers.head(top_n)
    existing_cols = [c for c in cols if c in top.columns]
    return top[existing_cols].copy()

