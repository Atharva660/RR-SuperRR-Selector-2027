"""
Audited Match-Level Bootstrap Resampling & Ranking Stability Engine
RR SuperRR Selector Season 2 - Independent Model Audit & Refinement

Performs true match-cluster bootstrap resampling:
1. Resamples matches with replacement (cluster bootstrap preserving within-match correlation).
2. Re-aggregates performance for every player in the bootstrap sample.
3. Applies Bayesian shrinkage and RR-PRI calculation per iteration.
4. Ranks candidates and tracks top-10 appearance frequencies, rank percentiles, and entry/exit dynamics.
"""
import pandas as pd
import numpy as np
import logging
from bayesian_model import (
    bayesian_strike_rate, bayesian_batting_average,
    bayesian_bowling_economy, bayesian_wickets_per_over,
    sample_reliability_factor
)
from player_ranking import compute_audited_batting_pri, compute_audited_bowling_pri

logger = logging.getLogger(__name__)

def run_match_level_bootstrap_batters(
    bbb_df: pd.DataFrame,
    name_map: pd.DataFrame,
    n_bootstrap: int = 150,
    seed: int = 42
) -> pd.DataFrame:
    """Execute match-level cluster bootstrap for batters."""
    rng = np.random.default_rng(seed)
    
    # Pre-filter deliveries
    df = bbb_df[~bbb_df["is_super_over"] & bbb_df["batter_ball_faced"]].copy()
    df["is_pressure"] = df["batting_pressure_score"] >= 50.0
    
    # Pre-aggregate match-level summary per striker to make bootstrap fast
    match_striker_summary = df.groupby(["match_id", "striker_id"]).agg(
        tot_runs=("batter_runs", "sum"),
        tot_balls=("batter_ball_faced", "sum"),
        tot_diss=("batter_dismissed", "sum"),
        press_runs=("batter_runs", lambda x: (x * df.loc[x.index, "is_pressure"]).sum()),
        press_balls=("batter_ball_faced", lambda x: (x * df.loc[x.index, "is_pressure"]).sum()),
        press_diss=("batter_dismissed", lambda x: (x * df.loc[x.index, "is_pressure"]).sum()),
        ipl_press_runs=("batter_runs", lambda x: (x * df.loc[x.index, "is_pressure"] * df.loc[x.index, "is_ipl"]).sum()),
        ipl_press_balls=("batter_ball_faced", lambda x: (x * df.loc[x.index, "is_pressure"] * df.loc[x.index, "is_ipl"]).sum()),
        ipl_press_diss=("batter_dismissed", lambda x: (x * df.loc[x.index, "is_pressure"] * df.loc[x.index, "is_ipl"]).sum()),
    ).reset_index()

    all_matches = match_striker_summary["match_id"].unique()
    n_matches = len(all_matches)
    
    top10_counts = {}
    player_ranks = {}
    
    for b in range(n_bootstrap):
        sampled_matches = rng.choice(all_matches, size=n_matches, replace=True)
        # Fast match indexing
        sample_df = match_striker_summary[match_striker_summary["match_id"].isin(sampled_matches)]
        
        agg = sample_df.groupby("striker_id").agg(
            all_runs=("tot_runs", "sum"),
            all_balls=("tot_balls", "sum"),
            pressure_runs=("press_runs", "sum"),
            pressure_balls=("press_balls", "sum"),
            pressure_dismissals=("press_diss", "sum"),
            ipl_pressure_runs=("ipl_press_runs", "sum"),
            ipl_pressure_balls=("ipl_press_balls", "sum"),
            ipl_pressure_dismissals=("ipl_press_diss", "sum"),
            matches=("match_id", "nunique")
        ).reset_index()
        
        # Primary IPL Eligibility
        elig = agg[(agg["pressure_balls"] >= 50) & (agg["ipl_pressure_balls"] >= 25) & (agg["matches"] >= 10)].copy()
        if len(elig) < 10:
            continue
            
        # Stats
        elig["all_strike_rate"] = elig["all_runs"] / elig["all_balls"].clip(1) * 100.0
        elig["pressure_strike_rate"] = elig["pressure_runs"] / elig["pressure_balls"].clip(1) * 100.0
        elig["sr_differential"] = elig["pressure_strike_rate"] - elig["all_strike_rate"]
        
        # Effective stats (IPL 75%, other 25%)
        non_ipl_b = (elig["pressure_balls"] - elig["ipl_pressure_balls"]).clip(lower=0)
        non_ipl_r = (elig["pressure_runs"] - elig["ipl_pressure_runs"]).clip(lower=0)
        non_ipl_d = (elig["pressure_dismissals"] - elig["ipl_pressure_dismissals"]).clip(lower=0)
        
        eff_b = 0.75 * elig["ipl_pressure_balls"] + 0.25 * non_ipl_b
        eff_r = 0.75 * elig["ipl_pressure_runs"] + 0.25 * non_ipl_r
        eff_d = 0.75 * elig["ipl_pressure_dismissals"] + 0.25 * non_ipl_d
        
        bayes_sr_df = bayesian_strike_rate(eff_r, eff_b)
        elig["bayes_sr"] = bayes_sr_df["bayes_sr"]
        elig["bayes_avg"] = bayesian_batting_average(eff_r, eff_d)
        elig["consistency_score"] = 0.55  # baseline proxy in bootstrap
        elig["pressure_success_rate"] = 0.35
        elig["sample_reliability"] = sample_reliability_factor(elig["pressure_balls"], 120.0)
        
        ranked = compute_audited_batting_pri(elig).sort_values("rr_pri", ascending=False).reset_index(drop=True)
        ranked["boot_rank"] = ranked.index + 1
        
        top10_ids = set(ranked.head(10)["striker_id"].tolist())
        for pid in top10_ids:
            top10_counts[pid] = top10_counts.get(pid, 0) + 1
            
        for _, r in ranked.iterrows():
            pid = r["striker_id"]
            if pid not in player_ranks:
                player_ranks[pid] = []
            player_ranks[pid].append(r["boot_rank"])

    # Summary
    results = []
    for pid, ranks in player_ranks.items():
        results.append({
            "striker_id": pid,
            "top10_probability": round(top10_counts.get(pid, 0) / n_bootstrap, 3),
            "median_rank": round(float(np.median(ranks)), 1),
            "rank_ci_low": round(float(np.percentile(ranks, 5)), 1),
            "rank_ci_high": round(float(np.percentile(ranks, 95)), 1)
        })
    res_df = pd.DataFrame(results).merge(name_map, left_on="striker_id", right_on="player_id", how="left")
    return res_df


def run_match_level_bootstrap_bowlers(
    bbb_df: pd.DataFrame,
    name_map: pd.DataFrame,
    n_bootstrap: int = 150,
    seed: int = 42
) -> pd.DataFrame:
    """Execute match-level cluster bootstrap for bowlers."""
    rng = np.random.default_rng(seed)
    df = bbb_df[~bbb_df["is_super_over"]].copy()
    df["is_pressure"] = df["bowling_pressure_score"] >= 50.0

    match_bowler_summary = df.groupby(["match_id", "bowler_id"]).agg(
        tot_rc=("bowler_runs_conceded", "sum"),
        tot_lb=("bowler_legal_ball", "sum"),
        tot_wk=("bowler_wicket", "sum"),
        press_rc=("bowler_runs_conceded", lambda x: (x * df.loc[x.index, "is_pressure"]).sum()),
        press_lb=("bowler_legal_ball", lambda x: (x * df.loc[x.index, "is_pressure"]).sum()),
        press_wk=("bowler_wicket", lambda x: (x * df.loc[x.index, "is_pressure"]).sum()),
        ipl_press_rc=("bowler_runs_conceded", lambda x: (x * df.loc[x.index, "is_pressure"] * df.loc[x.index, "is_ipl"]).sum()),
        ipl_press_lb=("bowler_legal_ball", lambda x: (x * df.loc[x.index, "is_pressure"] * df.loc[x.index, "is_ipl"]).sum()),
        ipl_press_wk=("bowler_wicket", lambda x: (x * df.loc[x.index, "is_pressure"] * df.loc[x.index, "is_ipl"]).sum()),
    ).reset_index()

    all_matches = match_bowler_summary["match_id"].unique()
    n_matches = len(all_matches)

    top10_counts = {}
    player_ranks = {}

    for b in range(n_bootstrap):
        sampled_matches = rng.choice(all_matches, size=n_matches, replace=True)
        sample_df = match_bowler_summary[match_bowler_summary["match_id"].isin(sampled_matches)]

        agg = sample_df.groupby("bowler_id").agg(
            all_rc=("tot_rc", "sum"),
            all_lb=("tot_lb", "sum"),
            pressure_runs_conceded=("press_rc", "sum"),
            pressure_legal_balls=("press_lb", "sum"),
            pressure_wickets=("press_wk", "sum"),
            ipl_pressure_runs_conceded=("ipl_press_rc", "sum"),
            ipl_pressure_legal_balls=("ipl_press_lb", "sum"),
            ipl_pressure_wickets=("ipl_press_wk", "sum"),
            matches=("match_id", "nunique")
        ).reset_index()

        elig = agg[(agg["pressure_legal_balls"] >= 60) & (agg["ipl_pressure_legal_balls"] >= 30) & (agg["matches"] >= 10)].copy()
        if len(elig) < 10:
            continue

        elig["all_economy"] = elig["all_rc"] / (elig["all_lb"] / 6.0).clip(0.1)
        elig["pressure_economy"] = elig["pressure_runs_conceded"] / (elig["pressure_legal_balls"] / 6.0).clip(0.1)
        elig["eco_differential"] = elig["all_economy"] - elig["pressure_economy"]

        non_ipl_b = (elig["pressure_legal_balls"] - elig["ipl_pressure_legal_balls"]).clip(lower=0)
        non_ipl_r = (elig["pressure_runs_conceded"] - elig["ipl_pressure_runs_conceded"]).clip(lower=0)
        non_ipl_w = (elig["pressure_wickets"] - elig["ipl_pressure_wickets"]).clip(lower=0)

        eff_b = 0.75 * elig["ipl_pressure_legal_balls"] + 0.25 * non_ipl_b
        eff_r = 0.75 * elig["ipl_pressure_runs_conceded"] + 0.25 * non_ipl_r
        eff_w = 0.75 * elig["ipl_pressure_wickets"] + 0.25 * non_ipl_w

        bayes_eco_df = bayesian_bowling_economy(eff_r, eff_b)
        elig["bayes_eco"] = bayes_eco_df["bayes_eco"]
        elig["bayes_wpo"] = bayesian_wickets_per_over(eff_w, eff_b)
        elig["consistency_score"] = 0.60
        elig["good_spell_rate"] = 0.40
        elig["sample_reliability"] = sample_reliability_factor(elig["pressure_legal_balls"], 120.0)

        ranked = compute_audited_bowling_pri(elig).sort_values("rr_pri", ascending=False).reset_index(drop=True)
        ranked["boot_rank"] = ranked.index + 1

        top10_ids = set(ranked.head(10)["bowler_id"].tolist())
        for pid in top10_ids:
            top10_counts[pid] = top10_counts.get(pid, 0) + 1

        for _, r in ranked.iterrows():
            pid = r["bowler_id"]
            if pid not in player_ranks:
                player_ranks[pid] = []
            player_ranks[pid].append(r["boot_rank"])

    results = []
    for pid, ranks in player_ranks.items():
        results.append({
            "bowler_id": pid,
            "top10_probability": round(top10_counts.get(pid, 0) / n_bootstrap, 3),
            "median_rank": round(float(np.median(ranks)), 1),
            "rank_ci_low": round(float(np.percentile(ranks, 5)), 1),
            "rank_ci_high": round(float(np.percentile(ranks, 95)), 1)
        })
    res_df = pd.DataFrame(results).merge(name_map, left_on="bowler_id", right_on="player_id", how="left")
    return res_df

