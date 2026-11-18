"""
Audited Feature Engineering Pipeline
RR SuperRR Selector Season 2 - Independent Model Audit & Refinement

Computes:
1. Batting features under pressure vs normal.
2. Bowling features under pressure vs normal.
3. Strict adherence to MCC/IPL cricket rules (no run-outs to bowlers, no byes/leg-byes to bowlers).
4. Competition weighting: IPL (75%) + Other T20 / T20I (25%).
5. Empirical Bayes shrinkage with 90% credible intervals.
6. Information-theoretic sample reliability scaling.
"""
import pandas as pd
import numpy as np
import logging
from bayesian_model import (
    bayesian_strike_rate, bayesian_batting_average,
    bayesian_bowling_economy, bayesian_wickets_per_over,
    sample_reliability_factor
)

logger = logging.getLogger(__name__)

PRESSURE_THRESHOLD = 50.0

def build_audited_batting_features(bbb: pd.DataFrame, name_map: pd.DataFrame) -> pd.DataFrame:
    """Build audited batting performance metrics and Bayesian features."""
    df = bbb[~bbb["is_super_over"]].copy()

    # Striker deliveries where batter faced the ball (not wide)
    df_faced = df[df["batter_ball_faced"]].copy()
    df_faced["is_pressure"] = df_faced["batting_pressure_score"] >= PRESSURE_THRESHOLD

    def aggregate_bat(sub, prefix=""):
        p = f"{prefix}_" if prefix else ""
        agg = sub.groupby("striker_id").agg(
            runs=("batter_runs", "sum"),
            balls=("batter_ball_faced", "sum"),
            dismissals=("batter_dismissed", "sum"),
            fours=("is_four", "sum"),
            sixes=("is_six", "sum"),
            dots=("is_dot_bat", "sum"),
            matches=("match_id", "nunique"),
        ).reset_index()

        agg[f"{p}runs"] = agg["runs"]
        agg[f"{p}balls"] = agg["balls"]
        agg[f"{p}dismissals"] = agg["dismissals"]
        agg[f"{p}matches"] = agg["matches"]
        agg[f"{p}strike_rate"] = (agg["runs"] / agg["balls"].clip(lower=1) * 100.0).round(2)
        agg[f"{p}avg"] = np.where(agg["dismissals"] > 0, (agg["runs"] / agg["dismissals"]).round(2), agg["runs"].astype(float))
        agg[f"{p}boundary_pct"] = ((agg["fours"] + agg["sixes"]) / agg["balls"].clip(lower=1) * 100.0).round(2)
        agg[f"{p}dot_pct"] = (agg["dots"] / agg["balls"].clip(lower=1) * 100.0).round(2)

        keep_cols = ["striker_id"] + [c for c in agg.columns if c.startswith(p)]
        return agg[keep_cols]

    # Baseline (All Deliveries)
    all_bat = aggregate_bat(df_faced, "all")

    # Pressure (All Competitions)
    press_faced = df_faced[df_faced["is_pressure"]].copy()
    press_bat = aggregate_bat(press_faced, "pressure")

    # Pressure IPL-Only
    ipl_press_faced = press_faced[press_faced["is_ipl"]].copy()
    ipl_press_bat = aggregate_bat(ipl_press_faced, "ipl_pressure")

    # Match-level consistency on pressure deliveries
    mr = press_faced.groupby(["striker_id", "match_id"])["batter_runs"].sum().reset_index()
    cons = mr.groupby("striker_id").agg(
        match_std=("batter_runs", "std"),
        match_mean=("batter_runs", "mean"),
        press_matches=("match_id", "count"),
        above20=("batter_runs", lambda x: (x >= 20).sum())
    ).reset_index()
    cons["match_std"] = cons["match_std"].fillna(0.0)
    cons["cv"] = cons["match_std"] / cons["match_mean"].clip(lower=0.1)
    cons["consistency_score"] = (1.0 / (1.0 + cons["cv"])).round(4)
    cons["pressure_success_rate"] = (cons["above20"] / cons["press_matches"].clip(lower=1)).round(4)

    # Merge
    bat = all_bat.merge(press_bat, on="striker_id", how="inner")
    bat = bat.merge(ipl_press_bat, on="striker_id", how="left").fillna(0.0)
    bat = bat.merge(cons[["striker_id", "consistency_score", "pressure_success_rate", "press_matches"]], on="striker_id", how="left").fillna(0.0)
    bat = bat.merge(name_map, left_on="striker_id", right_on="player_id", how="left")

    # Competition-weighted effective stats (IPL 75%, other 25%)
    # Effective balls & runs
    ipl_b = bat["ipl_pressure_balls"]
    tot_b = bat["pressure_balls"]
    non_ipl_b = (tot_b - ipl_b).clip(lower=0.0)
    ipl_r = bat["ipl_pressure_runs"]
    tot_r = bat["pressure_runs"]
    non_ipl_r = (tot_r - ipl_r).clip(lower=0.0)
    ipl_d = bat["ipl_pressure_dismissals"]
    tot_d = bat["pressure_dismissals"]
    non_ipl_d = (tot_d - ipl_d).clip(lower=0.0)

    bat["eff_pressure_balls"] = (0.75 * ipl_b + 0.25 * non_ipl_b).round(1)
    bat["eff_pressure_runs"] = (0.75 * ipl_r + 0.25 * non_ipl_r).round(1)
    bat["eff_pressure_dismissals"] = (0.75 * ipl_d + 0.25 * non_ipl_d).round(1)

    # Bayesian Shrinkage on effective stats
    bayes_sr_df = bayesian_strike_rate(bat["eff_pressure_runs"], bat["eff_pressure_balls"], prior_sr=135.0, prior_weight=60.0)
    bat["bayes_sr"] = bayes_sr_df["bayes_sr"]
    bat["sr_ci_low"] = bayes_sr_df["sr_ci_low"]
    bat["sr_ci_high"] = bayes_sr_df["sr_ci_high"]

    bat["bayes_avg"] = bayesian_batting_average(bat["eff_pressure_runs"], bat["eff_pressure_dismissals"], prior_avg=25.0, prior_dismissals=4.0)

    # Strike rate differential (pressure SR vs normal SR)
    bat["sr_differential"] = (bat["pressure_strike_rate"] - bat["all_strike_rate"]).round(2)
    bat["avg_differential"] = (bat["pressure_avg"] - bat["all_avg"]).round(2)

    # Sample Reliability Factor
    bat["sample_reliability"] = sample_reliability_factor(bat["pressure_balls"], scale_balls=120.0)

    logger.info(f"Batting features built for {len(bat)} total batters.")
    return bat


def build_audited_bowling_features(bbb: pd.DataFrame, name_map: pd.DataFrame) -> pd.DataFrame:
    """Build audited bowling performance metrics and Bayesian features."""
    df = bbb[~bbb["is_super_over"]].copy()
    df["is_pressure"] = df["bowling_pressure_score"] >= PRESSURE_THRESHOLD

    def aggregate_bowl(sub, prefix=""):
        p = f"{prefix}_" if prefix else ""
        agg = sub.groupby("bowler_id").agg(
            runs_conceded=("bowler_runs_conceded", "sum"),
            legal_balls=("bowler_legal_ball", "sum"),
            wickets=("bowler_wicket", "sum"),
            dots=("is_dot_bowl", "sum"),
            matches=("match_id", "nunique"),
        ).reset_index()

        agg[f"{p}runs_conceded"] = agg["runs_conceded"]
        agg[f"{p}legal_balls"] = agg["legal_balls"]
        agg[f"{p}wickets"] = agg["wickets"]
        agg[f"{p}overs"] = (agg["legal_balls"] / 6.0).round(2)
        agg[f"{p}matches"] = agg["matches"]
        agg[f"{p}economy"] = (agg["runs_conceded"] / agg[f"{p}overs"].clip(lower=0.1)).round(2)
        agg[f"{p}bowl_avg"] = np.where(agg["wickets"] > 0, (agg["runs_conceded"] / agg["wickets"]).round(2), agg["runs_conceded"].astype(float))
        agg[f"{p}bowl_sr"] = np.where(agg["wickets"] > 0, (agg["legal_balls"] / agg["wickets"]).round(2), agg["legal_balls"].astype(float))
        agg[f"{p}dot_pct"] = (agg["dots"] / agg["legal_balls"].clip(lower=1) * 100.0).round(2)

        keep_cols = ["bowler_id"] + [c for c in agg.columns if c.startswith(p)]
        return agg[keep_cols]

    # Baseline (All Deliveries)
    all_bowl = aggregate_bowl(df, "all")

    # Pressure (All Competitions)
    press_df = df[df["is_pressure"]].copy()
    press_bowl = aggregate_bowl(press_df, "pressure")

    # Pressure IPL-Only
    ipl_press_df = press_df[press_df["is_ipl"]].copy()
    ipl_press_bowl = aggregate_bowl(ipl_press_df, "ipl_pressure")

    # Match-level consistency on pressure deliveries
    mr = press_df.groupby(["bowler_id", "match_id"]).agg(
        rc=("bowler_runs_conceded", "sum"),
        lb=("bowler_legal_ball", "sum"),
        wk=("bowler_wicket", "sum")
    ).reset_index()
    mr["eco"] = mr["rc"] / (mr["lb"] / 6.0).clip(lower=0.1)
    mr["good_spell"] = (mr["eco"] <= 8.5) | (mr["wk"] >= 2)

    cons = mr.groupby("bowler_id").agg(
        eco_std=("eco", "std"),
        eco_mean=("eco", "mean"),
        press_matches=("match_id", "count"),
        good_spells=("good_spell", "sum")
    ).reset_index()
    cons["eco_std"] = cons["eco_std"].fillna(0.0)
    cons["eco_cv"] = cons["eco_std"] / cons["eco_mean"].clip(lower=0.1)
    cons["consistency_score"] = (1.0 / (1.0 + cons["eco_cv"])).round(4)
    cons["good_spell_rate"] = (cons["good_spells"] / cons["press_matches"].clip(lower=1)).round(4)

    # Merge
    bowl = all_bowl.merge(press_bowl, on="bowler_id", how="inner")
    bowl = bowl.merge(ipl_press_bowl, on="bowler_id", how="left").fillna(0.0)
    bowl = bowl.merge(cons[["bowler_id", "consistency_score", "good_spell_rate", "press_matches"]], on="bowler_id", how="left").fillna(0.0)
    bowl = bowl.merge(name_map, left_on="bowler_id", right_on="player_id", how="left")

    # Competition-weighted effective stats (IPL 75%, other 25%)
    ipl_b = bowl["ipl_pressure_legal_balls"]
    tot_b = bowl["pressure_legal_balls"]
    non_ipl_b = (tot_b - ipl_b).clip(lower=0.0)
    ipl_r = bowl["ipl_pressure_runs_conceded"]
    tot_r = bowl["pressure_runs_conceded"]
    non_ipl_r = (tot_r - ipl_r).clip(lower=0.0)
    ipl_w = bowl["ipl_pressure_wickets"]
    tot_w = bowl["pressure_wickets"]
    non_ipl_w = (tot_w - ipl_w).clip(lower=0.0)

    bowl["eff_pressure_legal_balls"] = (0.75 * ipl_b + 0.25 * non_ipl_b).round(1)
    bowl["eff_pressure_runs_conceded"] = (0.75 * ipl_r + 0.25 * non_ipl_r).round(1)
    bowl["eff_pressure_wickets"] = (0.75 * ipl_w + 0.25 * non_ipl_w).round(1)

    # Bayesian Shrinkage
    bayes_eco_df = bayesian_bowling_economy(bowl["eff_pressure_runs_conceded"], bowl["eff_pressure_legal_balls"], prior_eco=8.80, prior_weight=72.0)
    bowl["bayes_eco"] = bayes_eco_df["bayes_eco"]
    bowl["eco_ci_low"] = bayes_eco_df["eco_ci_low"]
    bowl["eco_ci_high"] = bayes_eco_df["eco_ci_high"]

    bowl["bayes_wpo"] = bayesian_wickets_per_over(bowl["eff_pressure_wickets"], bowl["eff_pressure_legal_balls"], prior_balls_per_wkt=18.0, prior_weight=72.0)

    # Economy differential (normal eco - pressure eco; positive = improved under pressure)
    bowl["eco_differential"] = (bowl["all_economy"] - bowl["pressure_economy"]).round(2)

    # Sample Reliability Factor
    bowl["sample_reliability"] = sample_reliability_factor(bowl["pressure_legal_balls"], scale_balls=120.0)

    logger.info(f"Bowling features built for {len(bowl)} total bowlers.")
    return bowl
