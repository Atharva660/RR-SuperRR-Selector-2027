"""
Unit Test Suite for RR SupeRR Selector Analytics Pipeline
Tests pressure engine calculations, Bayesian shrinkage estimators, and ranking integrity.
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bayesian_model import bayesian_shrink_strike_rate, bayesian_shrink_economy, bayesian_shrink_avg
from player_ranking import compute_batting_pri, compute_bowling_pri

class TestCricketPipeline(unittest.TestCase):

    def test_bayesian_strike_rate_shrinkage(self):
        # Sample with 0 balls faced should equal prior (130.0)
        sr_zero = bayesian_shrink_strike_rate(pd.Series([0]), pd.Series([0]), prior_sr=130.0, prior_weight=50.0)
        self.assertAlmostEqual(sr_zero.iloc[0], 130.0, places=1)
        
        # High sample (1000 balls, 2000 runs -> SR 200) should dominate prior
        sr_high = bayesian_shrink_strike_rate(pd.Series([2000]), pd.Series([1000]), prior_sr=130.0, prior_weight=50.0)
        self.assertGreater(sr_high.iloc[0], 195.0)

    def test_bayesian_economy_shrinkage(self):
        # 0 balls should equal prior economy (9.0)
        eco_zero = bayesian_shrink_economy(pd.Series([0]), pd.Series([0]), prior_eco=9.0, prior_weight=60.0)
        self.assertAlmostEqual(eco_zero.iloc[0], 9.0, places=1)

    def test_rr_pri_bounds(self):
        # Create mock batting dataframe
        mock_bat = pd.DataFrame([{
            "bayesian_pressure_sr": 180.0,
            "bayesian_pressure_avg": 35.0,
            "sr_differential": 40.0,
            "consistency_score": 0.65,
            "pressure_success_rate": 0.40,
            "pressure_balls": 150
        }, {
            "bayesian_pressure_sr": 110.0,
            "bayesian_pressure_avg": 15.0,
            "sr_differential": -20.0,
            "consistency_score": 0.35,
            "pressure_success_rate": 0.10,
            "pressure_balls": 60
        }])
        
        res = compute_batting_pri(mock_bat)
        self.assertTrue((res["rr_pri"] >= 0).all())
        self.assertTrue((res["rr_pri"] <= 100).all())
        # The better performer must have a higher RR-PRI
        self.assertGreater(res["rr_pri"].iloc[0], res["rr_pri"].iloc[1])

if __name__ == "__main__":
    unittest.main()

