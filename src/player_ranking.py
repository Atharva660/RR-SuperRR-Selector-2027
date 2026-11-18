"""
Audited Player Ranking & RR Pressure Reliability Index (RR-PRI) Module
RR SuperRR Selector Season 2 - Independent Model Audit & Refinement

Implements the official mathematical formulations for:
- RR-PRI (Batters)
- RR-PRI (Bowlers)
- Explicit Category A (Primary IPL Qualified), Category B (Limited IPL), Category C (Supplementary T20)
"""
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# --- BATTING RR-PRI ---
BATTING_PRI_WEIGHTS = {
    "bayes_sr": 0.25,        # Scoring rate under pressure
    "bayes_avg": 0.20,       # Wicket preservation / staying power
    "sr_differential": 0.15, # Positive leverage improvement
    "consistency": 0.20,     # Match-level consistency (1 / (1 + CV))
    "success_rate": 0.10,    # Frequency of high-impact contributions
    "sample_exposure": 0.10, # Information-theoretic sample reliability
}

# --- BOWLING RR-PRI ---
BOWLING_PRI_WEIGHTS = {
    "bayes_eco": 0.25,        # Run containment under pressure (inverted)
    "bayes_wpo": 0.20,        # Wicket threat per over under pressure
    "eco_differential": 0.15, # Positive economy improvement
    "consistency": 0.20,      # Spell-level consistency (1 / (1 + CV))
    "good_spell_rate": 0.10,  # Frequency of high-impact spells
    "sample_exposure": 0.10,  # Information-theoretic sample reliability
}


def compute_audited_batting_pri(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mathematical RR-PRI for eligible batters."""
    df = df.copy()
    scaler = MinMaxScaler()

    # 1. Bayes SR Component
    df["c_bayes_sr"] = scaler.fit_transform(df[["bayes_sr"]].clip(lower=100.0, upper=220.0))

    # 2. Bayes Avg Component (clamped to prevent outlier explosion)
    df["c_bayes_avg"] = scaler.fit_transform(df[["bayes_avg"]].clip(lower=10.0, upper=60.0))

    # 3. Strike Rate Differential (clamped to [-30, +50])
    sr_diff_clamped = df["sr_differential"].clip(lower=-30.0, upper=50.0)
    df["c_sr_diff"] = (sr_diff_clamped - sr_diff_clamped.min()) / (sr_diff_clamped.max() - sr_diff_clamped.min() + 1e-6)

    # 4. Consistency Component
    df["c_consistency"] = scaler.fit_transform(df[["consistency_score"]].fillna(0.5))

    # 5. Success Rate Component
    df["c_success"] = scaler.fit_transform(df[["pressure_success_rate"]].fillna(0.0))

    # 6. Sample Size Reliability
    df["c_sample"] = scaler.fit_transform(df[["sample_reliability"]].fillna(0.0))

    # Weighted Sum -> RR-PRI (0 to 100)
    w = BATTING_PRI_WEIGHTS
    df["rr_pri"] = (
        w["bayes_sr"] * df["c_bayes_sr"] +
        w["bayes_avg"] * df["c_bayes_avg"] +
        w["sr_differential"] * df["c_sr_diff"] +
        w["consistency"] * df["c_consistency"] +
        w["success_rate"] * df["c_success"] +
        w["sample_exposure"] * df["c_sample"]
    ) * 100.0

    df["rr_pri"] = df["rr_pri"].round(2)
    return df


def compute_audited_bowling_pri(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mathematical RR-PRI for eligible bowlers."""
    df = df.copy()
    scaler = MinMaxScaler()

    # 1. Economy Inversion (lower economy = higher score)
    # Clip between 6.0 and 12.0 RPO
    eco_clamped = df["bayes_eco"].clip(lower=6.0, upper=12.0)
    df["c_bayes_eco"] = (12.0 - eco_clamped) / (12.0 - 6.0)

    # 2. Wickets Per Over (clamped to [0.1, 0.8])
    wpo_clamped = df["bayes_wpo"].clip(lower=0.1, upper=0.8)
    df["c_bayes_wpo"] = (wpo_clamped - 0.1) / (0.8 - 0.1)

    # 3. Economy Differential (normal economy - pressure economy; positive = better under pressure)
    eco_diff_clamped = df["eco_differential"].clip(lower=-3.0, upper=3.0)
    df["c_eco_diff"] = (eco_diff_clamped - (-3.0)) / (3.0 - (-3.0))

    # 4. Consistency Component
    df["c_consistency"] = scaler.fit_transform(df[["consistency_score"]].fillna(0.5))

    # 5. Good Spell Rate Component
    df["c_spell"] = scaler.fit_transform(df[["good_spell_rate"]].fillna(0.0))

    # 6. Sample Size Reliability
    df["c_sample"] = scaler.fit_transform(df[["sample_reliability"]].fillna(0.0))

    # Weighted Sum -> RR-PRI (0 to 100)
    w = BOWLING_PRI_WEIGHTS
    df["rr_pri"] = (
        w["bayes_eco"] * df["c_bayes_eco"] +
        w["bayes_wpo"] * df["c_bayes_wpo"] +
        w["eco_differential"] * df["c_eco_diff"] +
        w["consistency"] * df["c_consistency"] +
        w["good_spell_rate"] * df["c_spell"] +
        w["sample_exposure"] * df["c_sample"]
    ) * 100.0

    df["rr_pri"] = df["rr_pri"].round(2)
    return df


def classify_eligibility_tier(row, is_batter: bool = True) -> str:
    """Classify player into Category A, B, or C."""
    if is_batter:
        tot_balls = row.get("pressure_balls", 0)
        ipl_balls = row.get("ipl_pressure_balls", 0)
        tot_matches = row.get("all_matches", 0)
        if tot_balls >= 50 and ipl_balls >= 25 and tot_matches >= 10:
            return "Category A: Primary IPL Qualified"
        elif ipl_balls > 0 and tot_balls >= 30:
            return "Category B: Limited IPL Exposure"
        else:
            return "Category C: Supplementary T20 Talent"
    else:
        tot_balls = row.get("pressure_legal_balls", 0)
        ipl_balls = row.get("ipl_pressure_legal_balls", 0)
        tot_matches = row.get("all_matches", 0)
        if tot_balls >= 60 and ipl_balls >= 30 and tot_matches >= 10:
            return "Category A: Primary IPL Qualified"
        elif ipl_balls > 0 and tot_balls >= 36:
            return "Category B: Limited IPL Exposure"
        else:
            return "Category C: Supplementary T20 Talent"
