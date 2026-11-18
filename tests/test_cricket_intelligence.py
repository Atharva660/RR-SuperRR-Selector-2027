"""
Comprehensive Audited Automated Test Suite
Rajasthan Royals SupeRR Selector Season 2 - Independent Model Audit

Covers all 8 Mandatory Test Categories:
A. Data Integrity Tests
B. Cricket Rules & MCC Laws Tests (Synthetic edge cases)
C. Dynamic Pressure Engine Tests (Zero leakage)
D. Statistical Models & Bayesian Shrinkage Tests
E. Player Ranking & RR-PRI Integrity Tests
F. Machine Learning & Leakage-Free Validation Tests
G. End-to-End Output File Verification Tests
H. Regression & Top-10 Integrity Tests
"""
import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

BASE = r"c:\Atharva\RR-SuperRR-Selector-2027"
sys.path.insert(0, os.path.join(BASE, "src"))

from bayesian_model import (
    bayesian_strike_rate, bayesian_batting_average,
    bayesian_bowling_economy, bayesian_wickets_per_over,
    sample_reliability_factor
)
from player_ranking import compute_audited_batting_pri, compute_audited_bowling_pri, classify_eligibility_tier
from data_cleaning import clean_ball_by_ball_audited, generate_reconciliation_report
from pressure_engine import compute_pressure_scores_audited

# =============================================================
# CATEGORY A: DATA INTEGRITY TESTS
# =============================================================
class TestDataIntegrity:

    def test_reconciliation_completeness(self):
        """Verify 100% of raw 200,999 deliveries are accounted for with 0 discrepancy."""
        bbb = pd.read_csv(os.path.join(BASE, "Data Pack", "ball_by_ball.csv"))
        matches = pd.read_csv(os.path.join(BASE, "Data Pack", "matches.csv"))
        comps = pd.read_csv(os.path.join(BASE, "Data Pack", "competitions.csv"))
        
        rep = generate_reconciliation_report(bbb, matches, comps)
        assert rep["raw_deliveries_total"] == 200999
        assert rep["ipl_deliveries"] == 87687
        assert rep["other_t20_deliveries"] == 6985
        assert rep["t20i_deliveries"] == 106327
        assert rep["reconciled_sum"] == 200999
        assert rep["discrepancy"] == 0

    def test_matches_reconciliation(self):
        """Verify all 883 matches are accounted for (867 played + 16 abandoned)."""
        matches = pd.read_csv(os.path.join(BASE, "Data Pack", "matches.csv"))
        bbb = pd.read_csv(os.path.join(BASE, "Data Pack", "ball_by_ball.csv"))
        
        played = bbb["match_id"].nunique()
        assert len(matches) == 883
        assert played == 867
        assert (len(matches) - played) == 16


# =============================================================
# CATEGORY B: CRICKET LAWS & RULE VERIFICATION
# =============================================================
class TestCricketRules:

    def test_wides_not_faced_by_batter(self):
        """Wides must not count as legal balls faced by striker."""
        synthetic_bbb = pd.DataFrame([{
            "match_id": 1, "innings": 1, "overid": 1, "over_ball_num": 1, "innings_ball_num": 1,
            "striker_id": 101, "bowler_id": 201, "dismissed_player_id": np.nan, "runs": 0,
            "extras1_runs": 1, "extras_2_runs": 0, "extras_type": "wide", "is_wicket": False,
            "dismissal_type_id": np.nan
        }])
        cleaned = clean_ball_by_ball_audited(synthetic_bbb, [])
        assert cleaned["batter_ball_faced"].iloc[0] == False
        assert cleaned["bowler_legal_ball"].iloc[0] == False
        assert cleaned["bowler_runs_conceded"].iloc[0] == 1  # wide charged to bowler

    def test_byes_legbyes_not_charged_to_bowler(self):
        """Byes and leg-byes must not be charged to bowler's runs conceded."""
        synthetic_bbb = pd.DataFrame([
            {"match_id": 1, "innings": 1, "overid": 1, "over_ball_num": 1, "innings_ball_num": 1,
             "striker_id": 101, "bowler_id": 201, "dismissed_player_id": np.nan, "runs": 0,
             "extras1_runs": 4, "extras_2_runs": 0, "extras_type": "byes", "is_wicket": False, "dismissal_type_id": np.nan},
            {"match_id": 1, "innings": 1, "overid": 1, "over_ball_num": 2, "innings_ball_num": 2,
             "striker_id": 101, "bowler_id": 201, "dismissed_player_id": np.nan, "runs": 0,
             "extras1_runs": 2, "extras_2_runs": 0, "extras_type": "legbyes", "is_wicket": False, "dismissal_type_id": np.nan}
        ])
        cleaned = clean_ball_by_ball_audited(synthetic_bbb, [])
        assert cleaned["bowler_runs_conceded"].sum() == 0  # 0 runs charged
        assert cleaned["bowler_legal_ball"].sum() == 2     # Both were legally bowled
        assert cleaned["batter_ball_faced"].sum() == 2    # Batter faced both balls

    def test_run_outs_excluded_from_bowler_wickets(self):
        """Run outs must never be credited as bowler wickets."""
        synthetic_bbb = pd.DataFrame([{
            "match_id": 1, "innings": 1, "overid": 1, "over_ball_num": 1, "innings_ball_num": 1,
            "striker_id": 101, "bowler_id": 201, "dismissed_player_id": 101, "runs": 1,
            "extras1_runs": 0, "extras_2_runs": 0, "extras_type": np.nan, "is_wicket": True,
            "dismissal_type_id": 4.0  # RunOut
        }])
        cleaned = clean_ball_by_ball_audited(synthetic_bbb, [])
        assert cleaned["bowler_wicket"].iloc[0] == False
        assert cleaned["batter_dismissed"].iloc[0] == True


# =============================================================
# CATEGORY C: PRESSURE ENGINE & ZERO LEAKAGE
# =============================================================
class TestPressureEngine:

    def test_pressure_scores_bounded(self):
        """Pressure scores must strictly remain within [0.0, 100.0]."""
        p_path = os.path.join(BASE, "data", "processed", "bbb_with_pressure.parquet")
        df = pd.read_parquet(p_path)
        assert (df["batting_pressure_score"] >= 0.0).all()
        assert (df["batting_pressure_score"] <= 100.0).all()
        assert (df["bowling_pressure_score"] >= 0.0).all()
        assert (df["bowling_pressure_score"] <= 100.0).all()

    def test_pre_delivery_state_zero_leakage(self):
        """First ball of an innings must always start with 0 cumulative runs and 0 wickets."""
        p_path = os.path.join(BASE, "data", "processed", "bbb_with_pressure.parquet")
        df = pd.read_parquet(p_path)
        first_balls = df.groupby(["match_id", "innings"]).first()
        assert (first_balls["cum_runs"] == 0).all()
        assert (first_balls["cum_wickets"] == 0).all()


# =============================================================
# CATEGORY D: STATISTICAL & BAYESIAN ESTIMATORS
# =============================================================
class TestStatisticalModels:

    def test_bayesian_sr_shrinkage_convergence(self):
        """Zero balls faced must shrink to prior SR (135.0); large sample dominates."""
        res_zero = bayesian_strike_rate(pd.Series([0]), pd.Series([0]), prior_sr=135.0, prior_weight=60.0)
        assert res_zero["bayes_sr"].iloc[0] == 135.0
        
        # 1000 balls, 2000 runs (SR 200)
        res_large = bayesian_strike_rate(pd.Series([2000]), pd.Series([1000]), prior_sr=135.0, prior_weight=60.0)
        assert res_large["bayes_sr"].iloc[0] > 195.0

    def test_credible_intervals_uncertainty(self):
        """Small sample must have wider credible interval than large sample."""
        small = bayesian_strike_rate(pd.Series([40]), pd.Series([20]), prior_sr=135.0, prior_weight=60.0)
        large = bayesian_strike_rate(pd.Series([400]), pd.Series([200]), prior_sr=135.0, prior_weight=60.0)
        
        ci_width_small = small["sr_ci_high"].iloc[0] - small["sr_ci_low"].iloc[0]
        ci_width_large = large["sr_ci_high"].iloc[0] - large["sr_ci_low"].iloc[0]
        assert ci_width_small > ci_width_large

    def test_sample_reliability_factor(self):
        """Sample factor must be monotonically increasing and bounded in [0, 1]."""
        balls = pd.Series([0, 30, 60, 120, 240, 500])
        s = sample_reliability_factor(balls, scale_balls=120.0)
        assert s.iloc[0] == 0.0
        assert s.is_monotonic_increasing
        assert (s <= 1.0).all()


# =============================================================
# CATEGORY E: PLAYER RANKINGS & RR-PRI INTEGRITY
# =============================================================
class TestPlayerRankings:

    def test_top10_batters_count_and_uniqueness(self):
        """Exactly 10 unique players with descending RR-PRI in top 10 batters."""
        top10_bat = pd.read_csv(os.path.join(BASE, "outputs", "rankings", "top10_batters_final.csv"))
        assert len(top10_bat) == 10
        assert top10_bat["striker_id"].nunique() == 10
        assert top10_bat["rr_pri"].is_monotonic_decreasing

    def test_top10_bowlers_count_and_uniqueness(self):
        """Exactly 10 unique players with descending RR-PRI in top 10 bowlers."""
        top10_bowl = pd.read_csv(os.path.join(BASE, "outputs", "rankings", "top10_bowlers_final.csv"))
        assert len(top10_bowl) == 10
        assert top10_bowl["bowler_id"].nunique() == 10
        assert top10_bowl["rr_pri"].is_monotonic_decreasing

    def test_no_zero_ipl_players_in_primary_leaderboards(self):
        """Every player in Category A primary leaderboard must have substantial IPL exposure."""
        top10_bat = pd.read_csv(os.path.join(BASE, "outputs", "rankings", "top10_batters_final.csv"))
        top10_bowl = pd.read_csv(os.path.join(BASE, "outputs", "rankings", "top10_bowlers_final.csv"))
        
        assert (top10_bat["ipl_pressure_balls"] >= 25).all()
        assert (top10_bowl["ipl_pressure_legal_balls"] >= 30).all()


# =============================================================
# CATEGORY F: MACHINE LEARNING TESTS
# =============================================================
class TestMachineLearning:

    def test_ml_evaluation_metrics_validity(self):
        """ML models must achieve valid ROC-AUC (> 0.50) and calibrated Brier Score (< 0.25)."""
        rep_path = os.path.join(BASE, "outputs", "reports", "model_evaluation_report.json")
        with open(rep_path, "r") as f:
            rep = json.load(f)
            
        res = rep["benchmark_results"]
        assert res["LogisticRegression"]["ROC_AUC"] > 0.55
        assert res["XGBoost"]["ROC_AUC"] > 0.55
        assert res["LogisticRegression"]["Brier_Score"] < 0.20


# =============================================================
# CATEGORY G: END-TO-END OUTPUT VERIFICATION
# =============================================================
class TestEndToEndOutputs:

    def test_mandatory_deliverables_exist(self):
        """All required deliverables must exist on disk and be non-empty."""
        mandatory_files = [
            os.path.join(BASE, "outputs", "rankings", "top10_batters_final.csv"),
            os.path.join(BASE, "outputs", "rankings", "top10_bowlers_final.csv"),
            os.path.join(BASE, "outputs", "rankings", "RR_SuperRR_FINAL_Rankings.xlsx"),
            os.path.join(BASE, "outputs", "reports", "RR_SupeRR_Selector_Methodology_Report.pdf"),
            os.path.join(BASE, "outputs", "reports", "data_reconciliation_report.json"),
            os.path.join(BASE, "outputs", "reports", "model_evaluation_report.json"),
            os.path.join(BASE, "outputs", "reports", "ranking_stability_report.json"),
            os.path.join(BASE, "dashboard", "frontend", "index.html"),
            os.path.join(BASE, "dashboard", "backend", "main.py"),
            os.path.join(BASE, "README.md"),
            os.path.join(BASE, "requirements.txt"),
        ]
        for f in mandatory_files:
            assert os.path.exists(f), f"Missing file: {f}"
            assert os.path.getsize(f) > 0, f"Empty file: {f}"


# =============================================================
# CATEGORY H: REGRESSION TESTS
# =============================================================
class TestRegression:

    def test_rinku_and_narine_proven_champions(self):
        """Rinku Singh and Sunil Narine must demonstrate high probability in top 10."""
        bat = pd.read_csv(os.path.join(BASE, "outputs", "rankings", "top10_batters_final.csv"))
        bowl = pd.read_csv(os.path.join(BASE, "outputs", "rankings", "top10_bowlers_final.csv"))
        
        rinku = bat[bat["player_name"] == "Rinku Singh"]
        narine = bowl[bowl["player_name"] == "Sunil Narine"]
        
        assert len(rinku) == 1
        assert rinku["top10_probability"].iloc[0] >= 0.80
        assert len(narine) == 1
        assert narine["top10_probability"].iloc[0] >= 0.80
