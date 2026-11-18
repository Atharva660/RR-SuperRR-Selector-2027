"""
Audited Bayesian Reliability Framework & Uncertainty Quantification
RR SuperRR Selector Season 2 - Independent Model Audit & Refinement

Implements conjugate Empirical Bayes estimators with explicit credible intervals:
1. Batting Strike Rate: Conjugate Gamma-Poisson / Beta-Binomial with prior SR = 135.0, prior weight N_0 = 60 balls.
2. Batting Average: Gamma-Poisson hazard shrinkage with prior Avg = 25.0, prior weight D_0 = 4 dismissals.
3. Bowling Economy: Gamma-Poisson rate shrinkage with prior Eco = 8.80, prior weight N_0 = 72 balls (12 overs).
4. Bowling Wickets: Gamma-Poisson rate shrinkage with prior Strike Rate = 18.0 balls/wkt, prior weight N_0 = 72 balls.
5. Analytical 90% Credible Intervals (CI) quantifying sample-size uncertainty.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- BATTING BAYESIAN ESTIMATORS ---
def bayesian_strike_rate(
    runs: pd.Series,
    balls: pd.Series,
    prior_sr: float = 135.0,
    prior_weight: float = 60.0
) -> pd.DataFrame:
    """
    Empirical Bayes strike rate estimator with 90% credible intervals.
    Returns DataFrame with [bayes_sr, sr_ci_low, sr_ci_high, sr_se].
    """
    prior_runs = (prior_sr / 100.0) * prior_weight
    post_balls = prior_weight + balls
    post_runs = prior_runs + runs
    bayes_sr = (post_runs / post_balls * 100.0).round(2)

    # Standard error and 90% CI (z = 1.645)
    se = (np.sqrt(runs + 1.0) / post_balls * 100.0).round(2)
    ci_low = (bayes_sr - 1.645 * se).clip(lower=50.0).round(2)
    ci_high = (bayes_sr + 1.645 * se).round(2)

    return pd.DataFrame({
        "bayes_sr": bayes_sr,
        "sr_ci_low": ci_low,
        "sr_ci_high": ci_high,
        "sr_se": se
    })


def bayesian_batting_average(
    runs: pd.Series,
    dismissals: pd.Series,
    prior_avg: float = 25.0,
    prior_dismissals: float = 4.0
) -> pd.Series:
    """
    Empirical Bayes batting average shrinkage to prevent small-sample explosion on 0 dismissals.
    """
    prior_runs = prior_avg * prior_dismissals
    bayes_avg = ((prior_runs + runs) / (prior_dismissals + dismissals)).round(2)
    return bayes_avg


# --- BOWLING BAYESIAN ESTIMATORS ---
def bayesian_bowling_economy(
    runs_conceded: pd.Series,
    legal_balls: pd.Series,
    prior_eco: float = 8.80,
    prior_weight: float = 72.0  # 12 overs
) -> pd.DataFrame:
    """
    Empirical Bayes bowling economy estimator with 90% credible intervals.
    Returns DataFrame with [bayes_eco, eco_ci_low, eco_ci_high, eco_se].
    """
    prior_runs_per_ball = prior_eco / 6.0
    prior_runs = prior_runs_per_ball * prior_weight
    post_balls = prior_weight + legal_balls
    post_runs = prior_runs + runs_conceded
    bayes_eco = ((post_runs / post_balls) * 6.0).round(2)

    se = ((np.sqrt(runs_conceded + 1.0) / post_balls) * 6.0).round(2)
    ci_low = (bayes_eco - 1.645 * se).clip(lower=3.0).round(2)
    ci_high = (bayes_eco + 1.645 * se).round(2)

    return pd.DataFrame({
        "bayes_eco": bayes_eco,
        "eco_ci_low": ci_low,
        "eco_ci_high": ci_high,
        "eco_se": se
    })


def bayesian_wickets_per_over(
    wickets: pd.Series,
    legal_balls: pd.Series,
    prior_balls_per_wkt: float = 18.0,
    prior_weight: float = 72.0
) -> pd.Series:
    """
    Empirical Bayes wicket-taking effectiveness per over.
    """
    prior_wkt_rate = 1.0 / prior_balls_per_wkt
    prior_wkts = prior_wkt_rate * prior_weight
    post_balls = prior_weight + legal_balls
    bayes_wpo = (((prior_wkts + wickets) / post_balls) * 6.0).round(3)
    return bayes_wpo


# --- SAMPLE SIZE RELIABILITY FUNCTION ---
def sample_reliability_factor(balls: pd.Series, scale_balls: float = 120.0) -> pd.Series:
    """
    Information-theoretic exposure scaling: S = 1 - exp(-balls / scale_balls).
    Smooth, non-linear penalty for low-sample players.
    - 30 balls: S = 0.22
    - 60 balls: S = 0.39
    - 120 balls: S = 0.63
    - 240 balls: S = 0.86
    - 360 balls: S = 0.95
    """
    return (1.0 - np.exp(-balls / scale_balls)).round(4)
