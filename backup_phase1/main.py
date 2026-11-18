"""
Main Analysis Pipeline - RR SuperRR Selector Season 2
Executes all phases from data loading to final rankings.
"""
import sys
import os
import logging
import json
import warnings
warnings.filterwarnings("ignore")

# Setup path
sys.path.insert(0, r"c:\Atharva\RR-SuperRR-Selector-2027\src")

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(r"c:\Atharva\RR-SuperRR-Selector-2027\outputs\logs\pipeline.log", mode="w"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("main_pipeline")

BASE = r"c:\Atharva\RR-SuperRR-Selector-2027"
PROCESSED = os.path.join(BASE, "data", "processed")
OUTPUTS = os.path.join(BASE, "outputs")

# Import modules
from data_loader import load_all_datasets, build_master_dataset, load_player_names
from data_cleaning import (assess_data_quality, clean_ball_by_ball,
                            clean_matches, clean_player_mapping)
from pressure_engine import compute_pressure_scores
from feature_engineering import build_batting_features, build_bowling_features
from bayesian_model import (bayesian_shrink_strike_rate, bayesian_shrink_economy,
                             bayesian_shrink_avg, compute_top10_probability)
from player_ranking import (compute_batting_pri, compute_bowling_pri,
                             rank_batters, rank_bowlers,
                             generate_batter_leaderboard, generate_bowler_leaderboard)


def phase_a_load(audit_log):
    """PHASE A: Load and inspect all datasets."""
    logger.info("=" * 60)
    logger.info("PHASE A: Data Loading & Schema Inspection")
    logger.info("=" * 60)

    datasets = load_all_datasets()
    name_map = load_player_names(datasets)

    logger.info(f"ball_by_ball: {datasets['ball_by_ball'].shape}")
    logger.info(f"matches: {datasets['matches'].shape}")
    logger.info(f"competitions: {datasets['competitions'].shape}")
    logger.info(f"teams: {datasets['teams'].shape}")
    logger.info(f"playing_eleven: {datasets['playing_eleven'].shape}")
    logger.info(f"player_mapping: {datasets['player_mapping'].shape}")
    logger.info(f"Unique players in mapping: {name_map['player_id'].nunique()}")

    return datasets, name_map


def phase_b_clean(datasets, audit_log):
    """PHASE B: Data cleaning and validation."""
    logger.info("=" * 60)
    logger.info("PHASE B: Data Cleaning & Validation")
    logger.info("=" * 60)

    # Quality reports
    bbb_report = assess_data_quality(datasets["ball_by_ball"], "ball_by_ball")
    matches_report = assess_data_quality(datasets["matches"], "matches")

    logger.info(f"BBB missing values: {sum(v > 0 for v in bbb_report['missing_values'].values())} columns with nulls")
    logger.info(f"Matches missing values: {sum(v > 0 for v in matches_report['missing_values'].values())} columns with nulls")

    # Clean
    bbb_clean = clean_ball_by_ball(datasets["ball_by_ball"], audit_log)
    matches_clean = clean_matches(datasets["matches"], audit_log)
    pm_clean = clean_player_mapping(datasets["player_mapping"], audit_log)

    # Save cleaned data
    bbb_clean.to_parquet(os.path.join(PROCESSED, "ball_by_ball_clean.parquet"), index=False)
    matches_clean.to_parquet(os.path.join(PROCESSED, "matches_clean.parquet"), index=False)
    pm_clean.to_csv(os.path.join(PROCESSED, "player_mapping_clean.csv"), index=False)

    logger.info(f"Cleaned BBB rows: {len(bbb_clean):,}")

    # Save quality report
    report = {
        "ball_by_ball": {k: str(v) if not isinstance(v, (int, float, dict, list)) else v
                         for k, v in bbb_report.items()},
        "matches": {k: str(v) if not isinstance(v, (int, float, dict, list)) else v
                    for k, v in matches_report.items()},
        "audit_log": audit_log,
    }
    with open(os.path.join(OUTPUTS, "reports", "data_quality_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    return bbb_clean, matches_clean, pm_clean


def phase_c_eda(bbb, matches):
    """PHASE C: Basic EDA stats."""
    logger.info("=" * 60)
    logger.info("PHASE C: Exploratory Data Analysis")
    logger.info("=" * 60)

    ipl_matches = matches[matches["comp_id"].isin([1358, 1715, 2193, 2725, 3298])]
    logger.info(f"Total matches: {len(matches)}")
    logger.info(f"IPL matches: {len(ipl_matches)}")
    logger.info(f"IPL deliveries: {bbb[bbb['is_ipl']].shape[0]:,}")
    logger.info(f"T20I deliveries: {bbb[bbb['is_t20i']].shape[0]:,}")

    ipl_bbb = bbb[bbb["is_ipl"]]
    logger.info(f"Unique batters (IPL): {ipl_bbb['striker_id'].nunique()}")
    logger.info(f"Unique bowlers (IPL): {ipl_bbb['bowler_id'].nunique()}")

    # Basic run scoring
    avg_sr = ipl_bbb.groupby("striker_id")["batter_runs"].sum().sum() / \
             ipl_bbb[ipl_bbb["is_legal_ball"]]["is_legal_ball"].sum() * 100
    logger.info(f"IPL average strike rate (all): {avg_sr:.1f}")


def phase_d_pressure(bbb, matches):
    """PHASE D: Pressure engine."""
    logger.info("=" * 60)
    logger.info("PHASE D: Pressure Situation Engine")
    logger.info("=" * 60)

    bbb_with_pressure = compute_pressure_scores(bbb, matches)

    # Statistics
    high_press = (bbb_with_pressure["batting_pressure_score"] >= 50).sum()
    extreme_press = (bbb_with_pressure["batting_pressure_score"] >= 75).sum()
    logger.info(f"High/Extreme batting pressure deliveries: {high_press:,}")
    logger.info(f"Extreme batting pressure deliveries: {extreme_press:,}")

    # Save
    bbb_with_pressure.to_parquet(
        os.path.join(PROCESSED, "bbb_with_pressure.parquet"), index=False
    )

    return bbb_with_pressure


def phase_e_features(bbb, name_map):
    """PHASE E: Feature engineering."""
    logger.info("=" * 60)
    logger.info("PHASE E: Feature Engineering")
    logger.info("=" * 60)

    batting = build_batting_features(bbb, name_map, ipl_only=False)
    bowling = build_bowling_features(bbb, name_map, ipl_only=False)

    batting.to_parquet(os.path.join(PROCESSED, "batting_features.parquet"), index=False)
    bowling.to_parquet(os.path.join(PROCESSED, "bowling_features.parquet"), index=False)

    logger.info(f"Batting features built: {len(batting)} eligible batters")
    logger.info(f"Bowling features built: {len(bowling)} eligible bowlers")

    return batting, bowling


def phase_f_bayesian(batting, bowling, bbb, name_map):
    """PHASE F: Bayesian reliability adjustment."""
    logger.info("=" * 60)
    logger.info("PHASE F: Bayesian Reliability Framework")
    logger.info("=" * 60)

    # Batting Bayesian adjustments
    batting["bayesian_pressure_sr"] = bayesian_shrink_strike_rate(
        batting["pressure_runs"],
        batting["pressure_balls"],
        prior_sr=130.0,
        prior_weight=50.0
    )
    batting["bayesian_pressure_avg"] = bayesian_shrink_avg(
        batting["pressure_runs"],
        batting["pressure_dismissals"],
        prior_avg=25.0,
        prior_dismissals=5.0
    )

    # Bowling Bayesian adjustments
    bowling["bayesian_pressure_eco"] = bayesian_shrink_economy(
        bowling["pressure_runs_conceded"],
        bowling["pressure_legal_balls"],
        prior_eco=9.0,
        prior_weight=60.0
    )

    logger.info("Bayesian shrinkage applied to batting SR, AVG and bowling economy")
    return batting, bowling


def phase_h_pri(batting, bowling):
    """PHASE H: RR-PRI computation."""
    logger.info("=" * 60)
    logger.info("PHASE H: RR Pressure Reliability Index")
    logger.info("=" * 60)

    batting = compute_batting_pri(batting)
    bowling = compute_bowling_pri(bowling)

    return batting, bowling


def phase_i_rankings(batting, bowling, bbb, name_map):
    """PHASE I: Final rankings and leaderboards."""
    logger.info("=" * 60)
    logger.info("PHASE I: Final Player Rankings")
    logger.info("=" * 60)

    ranked_batters = rank_batters(batting)
    ranked_bowlers = rank_bowlers(bowling)

    top10_batters = generate_batter_leaderboard(ranked_batters, 10)
    top10_bowlers = generate_bowler_leaderboard(ranked_bowlers, 10)

    # --- Bootstrap Top-10 probability estimation (match-level, fast) ---
    logger.info("Computing bootstrap top-10 probabilities (match-level, 100 iterations)...")
    rng = np.random.default_rng(42)
    all_matches = bbb["match_id"].unique()
    n_m = len(all_matches)

    press_bbb = bbb[(bbb["is_ipl"] | bbb["is_t20i"]) & ~bbb["is_super_over"] &
                    bbb["is_legal_ball"]].copy()
    press_bbb_bowl = bbb[(bbb["is_ipl"] | bbb["is_t20i"]) & ~bbb["is_super_over"]].copy()

    bat_top10_counts = {}
    bowl_top10_counts = {}
    N_BOOT = 100

    for _ in range(N_BOOT):
        sampled = rng.choice(all_matches, size=n_m, replace=True)
        s = press_bbb[press_bbb["match_id"].isin(sampled)]
        try:
            press_s = s[s["batting_pressure_score"] >= 50]
            agg = press_s.groupby("striker_id").agg(
                pr=("batter_runs", "sum"), pb=("is_legal_ball", "sum")
            ).reset_index()
            agg = agg[agg["pb"] >= 20]
            agg["psr"] = agg["pr"] / agg["pb"] * 100
            top10_ids = set(agg.nlargest(10, "psr")["striker_id"].tolist())
            for pid in top10_ids:
                bat_top10_counts[pid] = bat_top10_counts.get(pid, 0) + 1
        except Exception:
            pass

        sb = press_bbb_bowl[press_bbb_bowl["match_id"].isin(sampled)]
        try:
            press_sb = sb[sb["bowling_pressure_score"] >= 50]
            agg_b = press_sb.groupby("bowler_id").agg(
                rc=("total_runs", "sum"), lb=("is_legal_ball", "sum"), wk=("is_wicket", "sum")
            ).reset_index()
            agg_b = agg_b[agg_b["lb"] >= 36]
            agg_b["peco"] = agg_b["rc"] / (agg_b["lb"] / 6).clip(0.1)
            agg_b["score"] = agg_b["wk"] / agg_b["peco"].clip(0.1)
            top10_bowl = set(agg_b.nlargest(10, "score")["bowler_id"].tolist())
            for bid in top10_bowl:
                bowl_top10_counts[bid] = bowl_top10_counts.get(bid, 0) + 1
        except Exception:
            pass

    # Build probability columns for top10 tables
    top10_batters["top10_probability"] = top10_batters["striker_id"].map(
        lambda x: round(bat_top10_counts.get(x, 0) / N_BOOT, 3)
    )
    top10_bowlers["top10_probability"] = top10_bowlers["bowler_id"].map(
        lambda x: round(bowl_top10_counts.get(x, 0) / N_BOOT, 3)
    )

    # Save
    ranked_batters.to_csv(os.path.join(OUTPUTS, "rankings", "all_batters_ranked.csv"), index=False)
    ranked_bowlers.to_csv(os.path.join(OUTPUTS, "rankings", "all_bowlers_ranked.csv"), index=False)
    top10_batters.to_csv(os.path.join(OUTPUTS, "rankings", "top10_batters.csv"), index=False)
    top10_bowlers.to_csv(os.path.join(OUTPUTS, "rankings", "top10_bowlers.csv"), index=False)

    # Excel output
    with pd.ExcelWriter(os.path.join(OUTPUTS, "rankings", "RR_SuperRR_Final_Rankings.xlsx")) as writer:
        top10_batters.to_excel(writer, sheet_name="Top10_Batters", index=False)
        top10_bowlers.to_excel(writer, sheet_name="Top10_Bowlers", index=False)
        ranked_batters.head(50).to_excel(writer, sheet_name="All_Batters_Top50", index=False)
        ranked_bowlers.head(50).to_excel(writer, sheet_name="All_Bowlers_Top50", index=False)

    logger.info("Rankings saved to outputs/rankings/")
    return ranked_batters, ranked_bowlers, top10_batters, top10_bowlers


def print_leaderboards(top10_batters, top10_bowlers):
    """Print formatted leaderboards to console."""
    print("\n" + "=" * 80)
    print("🏏 TOP 10 BATTERS UNDER PRESSURE — RR SuperRR Selector Season 2")
    print("=" * 80)
    for _, row in top10_batters.iterrows():
        name = row.get("player_name", row.get("striker_id", "?"))
        print(f"  #{int(row['rank']):2d}  {name:<30}  "
              f"P-SR: {row.get('pressure_strike_rate', 0):.1f}  "
              f"Bayes-SR: {row.get('bayesian_pressure_sr', 0):.1f}  "
              f"P-AVG: {row.get('pressure_avg', 0):.1f}  "
              f"RR-PRI: {row.get('rr_pri', 0):.2f}")

    print("\n" + "=" * 80)
    print("🎯 TOP 10 BOWLERS UNDER PRESSURE — RR SuperRR Selector Season 2")
    print("=" * 80)
    for _, row in top10_bowlers.iterrows():
        name = row.get("player_name", row.get("bowler_id", "?"))
        print(f"  #{int(row['rank']):2d}  {name:<30}  "
              f"P-ECO: {row.get('pressure_economy', 0):.2f}  "
              f"Bayes-ECO: {row.get('bayesian_pressure_eco', 0):.2f}  "
              f"P-WKT: {row.get('pressure_wickets', 0):.0f}  "
              f"RR-PRI: {row.get('rr_pri', 0):.2f}")
    print("=" * 80)


def main():
    audit_log = []

    # PHASE A
    datasets, name_map = phase_a_load(audit_log)

    # PHASE B
    bbb, matches, pm = phase_b_clean(datasets, audit_log)

    # Build master dataset
    logger.info("Building master dataset...")
    master = build_master_dataset({
        "ball_by_ball": bbb,
        "matches": matches,
        "competitions": datasets["competitions"],
        "teams": datasets["teams"],
        "playing_eleven": datasets["playing_eleven"],
        "player_mapping": pm,
    })
    logger.info(f"Master dataset: {master.shape}")

    # PHASE C
    phase_c_eda(master, matches)

    # PHASE D
    master = phase_d_pressure(master, matches)

    # PHASE E
    batting, bowling = phase_e_features(master, name_map)

    # PHASE F
    batting, bowling = phase_f_bayesian(batting, bowling, master, name_map)

    # PHASE H
    batting, bowling = phase_h_pri(batting, bowling)

    # PHASE I
    ranked_batters, ranked_bowlers, top10_batters, top10_bowlers = \
        phase_i_rankings(batting, bowling, master, name_map)

    # Save audit log
    with open(os.path.join(OUTPUTS, "logs", "audit_log.txt"), "w") as f:
        f.write("\n".join(audit_log))

    # Print leaderboards
    print_leaderboards(top10_batters, top10_bowlers)

    # Summary stats
    total_deliveries = len(master)
    ipl_deliveries = master["is_ipl"].sum()
    high_pressure = (master["batting_pressure_score"] >= 50).sum()

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total deliveries analysed: {total_deliveries:,}")
    logger.info(f"IPL deliveries: {ipl_deliveries:,}")
    logger.info(f"High/Extreme pressure deliveries: {high_pressure:,}")
    logger.info(f"Eligible batters: {len(batting)}")
    logger.info(f"Eligible bowlers: {len(bowling)}")
    logger.info("=" * 60)

    return {
        "total_deliveries": total_deliveries,
        "ipl_deliveries": int(ipl_deliveries),
        "high_pressure_deliveries": int(high_pressure),
        "eligible_batters": len(batting),
        "eligible_bowlers": len(bowling),
        "top10_batters": top10_batters.to_dict("records"),
        "top10_bowlers": top10_bowlers.to_dict("records"),
    }


if __name__ == "__main__":
    result = main()
