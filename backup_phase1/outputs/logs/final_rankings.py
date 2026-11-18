"""
IPL-aware final rankings with competition weighting.
Runs after main pipeline to produce competition-adjusted leaderboards.
"""
import sys
sys.path.insert(0, r"c:\Atharva\RR-SuperRR-Selector-2027\src")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler

BASE = r"c:\Atharva\RR-SuperRR-Selector-2027"

bbb = pd.read_parquet(f"{BASE}/data/processed/bbb_with_pressure.parquet")
pm = pd.read_csv(f"{BASE}/data/processed/player_mapping_clean.csv")
name_map = pm.groupby("player_id")["player_name"].agg(lambda x: x.mode().iloc[0]).reset_index()

PRESS_THRESH = 50
MIN_TOTAL_PRESS_BALLS_BAT = 50
MIN_IPL_PRESS_BALLS_BAT = 20   # Must have played some IPL under pressure
MIN_TOTAL_PRESS_BALLS_BOWL = 60
MIN_IPL_PRESS_BALLS_BOWL = 20

# --- BUILD BATTING FEATURES WITH COMPETITION WEIGHTING ---
df = bbb[(bbb["is_ipl"] | bbb["is_t20i"]) & ~bbb["is_super_over"] & bbb["is_legal_ball"]].copy()
df["is_dismissed"] = df.apply(lambda r: r["is_wicket"] and r["dismissed_player_id"] == r["striker_id"], axis=1)
df["is_boundary"] = df["is_four"] | df["is_six"]
df["is_dot"] = df["batter_runs"] == 0
df["is_pressure"] = df["batting_pressure_score"] >= PRESS_THRESH

press_df = df[df["is_pressure"]].copy()

def agg_batting(sub, suffix=""):
    agg = sub.groupby("striker_id").agg(
        runs=("batter_runs", "sum"),
        balls=("is_legal_ball", "sum"),
        dismissals=("is_dismissed", "sum"),
        boundaries=("is_boundary", "sum"),
        dots=("is_dot", "sum"),
        matches=("match_id", "nunique"),
    ).reset_index()
    agg["sr"] = (agg["runs"] / agg["balls"].clip(1) * 100).round(2)
    agg["avg"] = (agg["runs"] / agg["dismissals"].clip(1)).round(2)
    agg.loc[agg["dismissals"] == 0, "avg"] = agg.loc[agg["dismissals"] == 0, "runs"]
    agg["boundary_pct"] = (agg["boundaries"] / agg["balls"].clip(1) * 100).round(2)
    agg["dot_pct"] = (agg["dots"] / agg["balls"].clip(1) * 100).round(2)
    if suffix:
        agg = agg.rename(columns={c: f"{suffix}_{c}" for c in agg.columns if c != "striker_id"})
    return agg

# All + pressure (IPL+T20I combined)
all_stats = agg_batting(df, "all")
press_stats = agg_batting(press_df, "press")

# IPL-only pressure
ipl_press = press_df[press_df["is_ipl"]]
ipl_stats = agg_batting(ipl_press, "ipl_press")

# Match-level consistency
def match_consistency(sub):
    mr = sub.groupby(["striker_id","match_id"])["batter_runs"].sum().reset_index()
    mr.columns = ["striker_id","match_id","match_runs"]
    cons = mr.groupby("striker_id").agg(
        std=("match_runs","std"), mean=("match_runs","mean"),
        n_matches=("match_id","count"),
        above20=("match_runs", lambda x: (x>=20).sum())
    ).reset_index()
    cons["std"] = cons["std"].fillna(0)
    cons["cv"] = cons["std"] / cons["mean"].clip(0.1)
    cons["consistency_score"] = 1 / (1 + cons["cv"])
    cons["success_rate"] = cons["above20"] / cons["n_matches"]
    return cons[["striker_id","consistency_score","success_rate","n_matches"]]

cons = match_consistency(press_df)

batting = all_stats.merge(press_stats, on="striker_id", how="inner")
batting = batting.merge(ipl_stats, on="striker_id", how="left")
batting = batting.merge(cons, on="striker_id", how="left")
batting = batting.merge(name_map, left_on="striker_id", right_on="player_id", how="left")

# IPL pressure balls (fillna 0 for those with no IPL)
batting["ipl_press_balls"] = batting["ipl_press_balls"].fillna(0)
batting["ipl_press_runs"] = batting["ipl_press_runs"].fillna(0)
batting["ipl_press_matches"] = batting["ipl_press_matches"].fillna(0)

# ELIGIBILITY: total press balls >= threshold AND at least some IPL press balls
batting = batting[
    (batting["press_balls"] >= MIN_TOTAL_PRESS_BALLS_BAT) &
    (batting["all_matches"] >= 10)
].copy()

# IPL weight: fraction of pressure exposure from IPL (0-1)
batting["ipl_weight"] = (batting["ipl_press_balls"] / batting["press_balls"].clip(1)).clip(0, 1)
batting["has_ipl_exposure"] = batting["ipl_press_balls"] >= MIN_IPL_PRESS_BALLS_BAT

# --- BAYESIAN ADJUSTMENTS ---
prior_sr = 130.0; prior_w = 50.0
prior_avg = 25.0; prior_d = 5.0

batting["bayes_press_sr"] = (
    (prior_sr/100 * prior_w + batting["press_runs"]) / (prior_w + batting["press_balls"]) * 100
).round(2)
batting["bayes_press_avg"] = (
    (prior_avg * prior_d + batting["press_runs"]) / (prior_d + batting["press_dismissals"]).clip(1)
).round(2)

# SR improvement
batting["sr_improvement"] = batting["press_sr"] - batting["all_sr"]

# --- RR-PRI COMPUTATION ---
scaler = MinMaxScaler()

# Score components (all normalized to [0,1])
batting["c_bayes_sr"] = scaler.fit_transform(batting[["bayes_press_sr"]])
batting["c_bayes_avg"] = scaler.fit_transform(batting[["bayes_press_avg"]].clip(upper=100))
sr_imp = batting["sr_improvement"].clip(-50, 50)
batting["c_sr_imp"] = (sr_imp - sr_imp.min()) / (sr_imp.max() - sr_imp.min() + 1e-6)
cons_col = batting["consistency_score"].fillna(0.5)
batting["c_cons"] = scaler.fit_transform(cons_col.values.reshape(-1,1))
batting["c_success"] = scaler.fit_transform(batting["success_rate"].fillna(0).values.reshape(-1,1))
max_log = np.log(batting["press_balls"].max() + 1)
batting["c_sample"] = np.log(batting["press_balls"] + 1) / max_log

# IPL weight boosts IPL-primary players slightly
batting["ipl_boost"] = 0.03 * batting["ipl_weight"]

batting["rr_pri"] = (
    0.25 * batting["c_bayes_sr"] +
    0.20 * batting["c_bayes_avg"] +
    0.15 * batting["c_sr_imp"] +
    0.20 * batting["c_cons"] +
    0.10 * batting["c_success"] +
    0.10 * batting["c_sample"] +
    batting["ipl_boost"]
) * 100

batting["rr_pri"] = batting["rr_pri"].round(2)
batting_ranked = batting.sort_values("rr_pri", ascending=False).reset_index(drop=True)
batting_ranked["rank"] = batting_ranked.index + 1

# Save full ranked list
batting_ranked.to_csv(f"{BASE}/outputs/rankings/all_batters_ipl_weighted.csv", index=False)

# Top 10
top10_bat = batting_ranked.head(10)[[
    "rank","player_name","striker_id","all_matches",
    "press_runs","press_balls","press_sr","press_avg",
    "all_sr","sr_improvement","bayes_press_sr","bayes_press_avg",
    "consistency_score","success_rate","ipl_press_balls","ipl_weight","rr_pri"
]]
top10_bat.to_csv(f"{BASE}/outputs/rankings/top10_batters_final.csv", index=False)

print("=== TOP 10 BATTERS (IPL-weighted) ===")
for _, r in top10_bat.iterrows():
    print(f"  #{int(r['rank']):2d}  {r['player_name']:<28}  "
          f"P-SR: {r['press_sr']:.1f}  Bayes-SR: {r['bayes_press_sr']:.1f}  "
          f"P-AVG: {r['press_avg']:.1f}  IPL%: {r['ipl_weight']:.0%}  "
          f"RR-PRI: {r['rr_pri']:.2f}")

# =================================================================
# BOWLING
# =================================================================
df_bowl = bbb[(bbb["is_ipl"] | bbb["is_t20i"]) & ~bbb["is_super_over"]].copy()
df_bowl["is_boundary"] = df_bowl["is_four"] | df_bowl["is_six"]
df_bowl["is_dot"] = df_bowl["total_runs"] == 0
df_bowl["is_press_bowl"] = df_bowl["bowling_pressure_score"] >= PRESS_THRESH

press_bowl = df_bowl[df_bowl["is_press_bowl"]].copy()

def agg_bowling(sub, suffix=""):
    agg = sub.groupby("bowler_id").agg(
        runs=("total_runs","sum"),
        balls=("is_legal_ball","sum"),
        wickets=("is_wicket","sum"),
        boundaries=("is_boundary","sum"),
        dots=("is_dot","sum"),
        matches=("match_id","nunique"),
    ).reset_index()
    agg["overs"] = (agg["balls"] / 6).round(2)
    agg["economy"] = (agg["runs"] / agg["overs"].clip(0.01)).round(2)
    agg["bowl_avg"] = (agg["runs"] / agg["wickets"].clip(1)).round(2)
    agg["bowl_sr"] = (agg["balls"] / agg["wickets"].clip(1)).round(2)
    agg["boundary_rate"] = (agg["boundaries"] / agg["balls"].clip(1) * 100).round(2)
    agg["dot_pct"] = (agg["dots"] / agg["balls"].clip(1) * 100).round(2)
    if suffix:
        agg = agg.rename(columns={c: f"{suffix}_{c}" for c in agg.columns if c != "bowler_id"})
    return agg

all_bowl = agg_bowling(df_bowl, "all")
press_bowl_stats = agg_bowling(press_bowl, "press")

ipl_press_bowl = press_bowl[press_bowl["is_ipl"]]
ipl_bowl_stats = agg_bowling(ipl_press_bowl, "ipl_press")

def bowl_match_consistency(sub):
    mr = sub.groupby(["bowler_id","match_id"]).agg(
        rc=("total_runs","sum"), lb=("is_legal_ball","sum"), wk=("is_wicket","sum")
    ).reset_index()
    mr["eco"] = mr["rc"] / (mr["lb"] / 6).clip(0.01)
    mr["good"] = (mr["eco"] <= 9) | (mr["wk"] >= 2)
    cons = mr.groupby("bowler_id").agg(
        eco_std=("eco","std"), eco_mean=("eco","mean"),
        n=("match_id","count"), good_spells=("good","sum")
    ).reset_index()
    cons["eco_std"] = cons["eco_std"].fillna(0)
    cons["eco_cv"] = cons["eco_std"] / cons["eco_mean"].clip(0.1)
    cons["bowl_consistency"] = 1 / (1 + cons["eco_cv"])
    cons["good_spell_rate"] = cons["good_spells"] / cons["n"]
    return cons[["bowler_id","bowl_consistency","good_spell_rate"]]

bowl_cons = bowl_match_consistency(press_bowl)

bowling = all_bowl.merge(press_bowl_stats, on="bowler_id", how="inner")
bowling = bowling.merge(ipl_bowl_stats, on="bowler_id", how="left")
bowling = bowling.merge(bowl_cons, on="bowler_id", how="left")
bowling = bowling.merge(name_map, left_on="bowler_id", right_on="player_id", how="left")

bowling["ipl_press_balls"] = bowling["ipl_press_balls"].fillna(0)
bowling["ipl_weight"] = (bowling["ipl_press_balls"] / bowling["press_balls"].clip(1)).clip(0,1)

# Eligibility
bowling = bowling[
    (bowling["press_balls"] >= MIN_TOTAL_PRESS_BALLS_BOWL) &
    (bowling["all_matches"] >= 10)
].copy()

# Bayesian economy
prior_eco = 9.0; prior_eco_w = 60.0
prior_eco_runs = prior_eco / 6 * prior_eco_w
bowling["bayes_press_eco"] = (
    (prior_eco_runs + bowling["press_runs"]) / (prior_eco_w + bowling["press_balls"]) * 6
).round(2)

bowling["eco_improvement"] = -(bowling["press_economy"] - bowling["all_economy"])  # positive = better

# --- BOWLING RR-PRI ---
# Invert economy (lower = better)
eco_inv = 1 / bowling["bayes_press_eco"].clip(0.1)
bowling["c_eco"] = scaler.fit_transform(eco_inv.values.reshape(-1,1))

wpo = bowling["press_wickets"] / bowling["press_overs"].clip(0.1)
bowling["c_wkts"] = scaler.fit_transform(wpo.values.reshape(-1,1))

eco_imp = bowling["eco_improvement"].clip(-4,4)
bowling["c_eco_imp"] = (eco_imp - eco_imp.min()) / (eco_imp.max() - eco_imp.min() + 1e-6)

bowling["c_cons"] = scaler.fit_transform(bowling["bowl_consistency"].fillna(0.5).values.reshape(-1,1))
bowling["c_spell"] = scaler.fit_transform(bowling["good_spell_rate"].fillna(0).values.reshape(-1,1))
max_log_b = np.log(bowling["press_balls"].max() + 1)
bowling["c_sample"] = np.log(bowling["press_balls"] + 1) / max_log_b

bowling["ipl_boost"] = 0.03 * bowling["ipl_weight"]

bowling["rr_pri"] = (
    0.25 * bowling["c_eco"] +
    0.20 * bowling["c_wkts"] +
    0.15 * bowling["c_eco_imp"] +
    0.20 * bowling["c_cons"] +
    0.10 * bowling["c_spell"] +
    0.10 * bowling["c_sample"] +
    bowling["ipl_boost"]
) * 100
bowling["rr_pri"] = bowling["rr_pri"].round(2)

bowling_ranked = bowling.sort_values("rr_pri", ascending=False).reset_index(drop=True)
bowling_ranked["rank"] = bowling_ranked.index + 1
bowling_ranked.to_csv(f"{BASE}/outputs/rankings/all_bowlers_ipl_weighted.csv", index=False)

top10_bowl = bowling_ranked.head(10)[[
    "rank","player_name","bowler_id","all_matches",
    "press_overs","press_wickets","press_economy","press_bowl_avg",
    "press_dot_pct","all_economy","eco_improvement",
    "bayes_press_eco","bowl_consistency","good_spell_rate",
    "ipl_press_balls","ipl_weight","rr_pri"
]]
top10_bowl.to_csv(f"{BASE}/outputs/rankings/top10_bowlers_final.csv", index=False)

print()
print("=== TOP 10 BOWLERS (IPL-weighted) ===")
for _, r in top10_bowl.iterrows():
    print(f"  #{int(r['rank']):2d}  {r['player_name']:<28}  "
          f"P-ECO: {r['press_economy']:.2f}  Bayes-ECO: {r['bayes_press_eco']:.2f}  "
          f"P-WKT: {r['press_wickets']:.0f}  IPL%: {r['ipl_weight']:.0%}  "
          f"RR-PRI: {r['rr_pri']:.2f}")

# Save Excel
with pd.ExcelWriter(f"{BASE}/outputs/rankings/RR_SuperRR_FINAL_Rankings.xlsx") as writer:
    top10_bat.to_excel(writer, sheet_name="Top10_Batters", index=False)
    top10_bowl.to_excel(writer, sheet_name="Top10_Bowlers", index=False)
    batting_ranked.head(50).to_excel(writer, sheet_name="All_Batters_Top50", index=False)
    bowling_ranked.head(50).to_excel(writer, sheet_name="All_Bowlers_Top50", index=False)

print("\nFinal rankings saved.")

