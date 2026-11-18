"""
Bayesian Shrinkage Module - RR SuperRR Selector Season 2
Adjusts player estimates for sample size using Bayesian shrinkage.
"""
import pandas as pd
import numpy as np
import logging
from scipy import stats

logger = logging.getLogger(__name__)


def bayesian_shrink_strike_rate(
    obs_runs: pd.Series,
    obs_balls: pd.Series,
    prior_sr: float = 130.0,
    prior_weight: float = 50.0
) -> pd.Series:
    """
    Bayesian shrinkage for batting strike rate.
    Prior: Beta-Binomial with prior_sr/100 success rate and prior_weight pseudo-observations.
    Posterior mean: (prior_runs + obs_runs) / (prior_balls + obs_balls)

    Prior choice: T20 average pressure strike rate ~130 (domain informed)
    Prior weight = 50 balls (moderate - 1 good T20 innings worth)
    """
    prior_runs = (prior_sr / 100) * prior_weight
    posterior_sr = (prior_runs + obs_runs) / (prior_weight + obs_balls) * 100
    return posterior_sr.round(2)


def bayesian_shrink_economy(
    obs_runs: pd.Series,
    obs_balls: pd.Series,
    prior_eco: float = 9.0,
    prior_weight: float = 60.0
) -> pd.Series:
    """
    Bayesian shrinkage for bowling economy rate.
    Prior: average T20 pressure economy ~9.0 runs per over
    Prior weight = 60 balls (10 overs)
    """
    prior_runs_per_ball = prior_eco / 6.0
    prior_runs_total = prior_runs_per_ball * prior_weight
    posterior_runs_per_ball = (prior_runs_total + obs_runs) / (prior_weight + obs_balls)
    return (posterior_runs_per_ball * 6).round(2)


def bayesian_shrink_avg(
    obs_runs: pd.Series,
    obs_dismissals: pd.Series,
    prior_avg: float = 25.0,
    prior_dismissals: float = 5.0
) -> pd.Series:
    """
    Bayesian shrinkage for batting average.
    Model: Gamma-Poisson conjugate approximation.
    Prior avg = 25 (T20 pressure average), prior dismissals = 5
    """
    prior_runs = prior_avg * prior_dismissals
    posterior_avg = (prior_runs + obs_runs) / (prior_dismissals + obs_dismissals).clip(lower=1)
    return posterior_avg.round(2)


def bootstrap_confidence_intervals(
    bbb_subset: pd.DataFrame,
    player_col: str,
    metric_fn,
    n_bootstrap: int = 500,
    ci_level: float = 0.90,
    seed: int = 42
) -> pd.DataFrame:
    """
    Bootstrap confidence intervals at match level for player rankings.
    Resamples matches with replacement to estimate ranking uncertainty.

    Returns DataFrame with player_id, ci_low, ci_high, mean_estimate
    """
    rng = np.random.default_rng(seed)
    players = bbb_subset[player_col].unique()
    results = []

    matches = bbb_subset["match_id"].unique()
    n_matches = len(matches)

    for _ in range(n_bootstrap):
        sampled_matches = rng.choice(matches, size=n_matches, replace=True)
        sample_df = bbb_subset[bbb_subset["match_id"].isin(sampled_matches)]
        player_metrics = metric_fn(sample_df)
        results.append(player_metrics)

    # Stack and compute CIs
    all_results = pd.concat(results)
    metric_col = [c for c in all_results.columns if c != player_col][0]

    ci_low = all_results.groupby(player_col)[metric_col].quantile((1 - ci_level) / 2)
    ci_high = all_results.groupby(player_col)[metric_col].quantile(1 - (1 - ci_level) / 2)
    mean_est = all_results.groupby(player_col)[metric_col].mean()

    ci_df = pd.DataFrame({
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_mean": mean_est,
    }).reset_index()

    return ci_df


def compute_top10_probability(
    bbb: pd.DataFrame,
    player_col: str,
    ranking_fn,
    n_bootstrap: int = 300,
    seed: int = 42
) -> pd.DataFrame:
    """
    Estimate probability of each player appearing in top 10
    via match-level bootstrap resampling.
    """
    rng = np.random.default_rng(seed)
    matches = bbb["match_id"].unique()
    n_matches = len(matches)
    top10_counts = {}

    for i in range(n_bootstrap):
        sampled = rng.choice(matches, size=n_matches, replace=True)
        sample_df = bbb[bbb["match_id"].isin(sampled)]
        try:
            ranked = ranking_fn(sample_df)
            top10 = set(ranked.head(10)[player_col].tolist())
            for pid in top10:
                top10_counts[pid] = top10_counts.get(pid, 0) + 1
        except Exception:
            continue

    prob_df = pd.DataFrame([
        {"player_id": k, "top10_probability": round(v / n_bootstrap, 3)}
        for k, v in top10_counts.items()
    ])
    return prob_df

