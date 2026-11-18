"""
Definitive Final Rankings with proper competition handling.
Priority: IPL players first; T20I performance = supplementary evidence only.
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

# =============================================================
#  STEP 1: COMPUTE ALL PRESSURE STATS BY COMPETITION TYPE
# =============================================================
def agg_batter(sub, suffix):
    if len(sub) == 0:
        return pd.DataFrame(columns=["striker_id"])
    sub = sub.copy()
    sub["is_dismissed"] = sub.apply(
        lambda r: r["is_wicket"] and r["dismissed_player_id"] == r["striker_id"], axis=1)
    sub["is_boundary"] = sub["is_four"] | sub["is_six"]
    sub["is_dot"] = sub["batter_runs"] == 0
    agg = sub.groupby("striker_id").agg(
        runs=("batter_runs","sum"), balls=("is_legal_ball","sum"),
        dismissals=("is_dismissed","sum"), boundaries=("is_boundary","sum"),
        dots=("is_dot","sum"), matches=("match_id","nunique"),
    ).reset_index()
    agg["sr"] = (agg["runs"] / agg["balls"].clip(1) * 100).round(2)
    agg["avg"] = np.where(agg["dismissals"] > 0, agg["runs"] / agg["dismissals"], agg["runs"]).round(2)
    agg = agg.rename(columns={c: f"{suffix}_{c}" for c in agg.columns if c != "striker_id"})
    return agg


def agg_bowler(sub, suffix):
    if len(sub) == 0:
        return pd.DataFrame(columns=["bowler_id"])
    sub = sub.copy()
    sub["is_boundary"] = sub["is_four"] | sub["is_six"]
    sub["is_dot"] = sub["total_runs"] == 0
    agg = sub.groupby("bowler_id").agg(
        runs=("total_runs","sum"), balls=("is_legal_ball","sum"),
        wickets=("is_wicket","sum"), boundaries=("is_boundary","sum"),
        dots=("is_dot","sum"), matches=("match_id","nunique"),
    ).reset_index()
    agg["overs"] = (agg["balls"] / 6).round(2)
    agg["economy"] = (agg["runs"] / agg["overs"].clip(0.01)).round(2)
    agg["bowl_avg"] = (agg["runs"] / agg["wickets"].clip(1)).round(2)
    agg["bowl_sr"] = (agg["balls"] / agg["wickets"].clip(1)).round(2)
    agg["dot_pct"] = (agg["dots"] / agg["balls"].clip(1) * 100).round(2)
    agg = agg.rename(columns={c: f"{suffix}_{c}" for c in agg.columns if c != "bowler_id"})
    return agg


# --- BATTING ---
bat_base = bbb[(bbb["is_ipl"] | bbb["is_t20i"]) & ~bbb["is_super_over"] & bbb["is_legal_ball"]]
bat_press_all = bat_base[bat_base["batting_pressure_score"] >= PRESS_THRESH]

# All deliveries (IPL+T20I) for baseline
all_bat = agg_batter(bat_base, "all")

# IPL pressure only
ipl_press_bat = agg_batter(bat_press_all[bat_press_all["is_ipl"]], "ipl_press")

# T20I pressure only
t20i_press_bat = agg_batter(bat_press_all[bat_press_all["is_t20i"]], "t20i_press")

# Combined pressure
comb_press_bat = agg_batter(bat_press_all, "comb_press")

# Match consistency (all pressure)
def bat_consistency(sub):
    mr = sub.groupby(["striker_id","match_id"])["batter_runs"].sum().reset_index()
    cons = mr.groupby("striker_id").agg(
        std=("batter_runs","std"), mean=("batter_runs","mean"),
        n=("match_id","count"), above20=("batter_runs", lambda x: (x>=20).sum())
    ).reset_index()
    cons["std"] = cons["std"].fillna(0)
    cons["cv"] = cons["std"] / cons["mean"].clip(0.1)
    cons["cons"] = 1 / (1 + cons["cv"])
    cons["success_rate"] = cons["above20"] / cons["n"]
    return cons[["striker_id","cons","success_rate","n"]]

bat_cons = bat_consistency(bat_press_all)

batting = all_bat.merge(ipl_press_bat, on="striker_id", how="left")
batting = batting.merge(t20i_press_bat, on="striker_id", how="left")
batting = batting.merge(comb_press_bat, on="striker_id", how="left")
batting = batting.merge(bat_cons, on="striker_id", how="left")
batting = batting.merge(name_map, left_on="striker_id", right_on="player_id", how="left")

batting = batting.fillna(0)

# ELIGIBILITY: At least 50 combined pressure balls AND at least 10 total matches
# AND at least 15 IPL pressure balls (meaningful IPL presence required)
batting["ELIGIBLE"] = (
    (batting["comb_press_balls"] >= 50) &
    (batting["all_matches"] >= 10) &
    (batting["ipl_press_balls"] >= 15)   # Must have material IPL pressure exposure
)
bat_elig = batting[batting["ELIGIBLE"]].copy()
print(f"Eligible batters: {len(bat_elig)}")

# --- BAYESIAN for IPL-primary performance ---
# If player has IPL press balls: use IPL stats, else use combined
# Bayesian prior: T20 pressure SR = 130
prior_sr = 130.0; prior_w = 50.0
prior_avg = 25.0; prior_d = 5.0

def mix_bat_stats(row):
    """Compute competition-mixed pressure stats. IPL weighted 70%, T20I 30% if both exist."""
    ipl_b = row["ipl_press_balls"]
    t20i_b = row["t20i_press_balls"]
    total_b = ipl_b + t20i_b
    if total_b < 1:
        return 0, 0, 0, 0
    # Weighted average runs and dismissals
    ipl_w = ipl_b / total_b
    t20i_w = t20i_b / total_b
    # Apply IPL weight: 0.7 * ipl_frac + 0.3 * t20i_frac
    effective_ipl_w = min(ipl_w * 1.4, 1.0)   # boost IPL weight
    effective_t20i_w = 1 - effective_ipl_w
    runs = effective_ipl_w * row["ipl_press_runs"] + effective_t20i_w * row["t20i_press_runs"]
    balls = effective_ipl_w * ipl_b + effective_t20i_w * t20i_b
    dismissals = effective_ipl_w * row["ipl_press_dismissals"] + effective_t20i_w * row["t20i_press_dismissals"]
    return runs, balls, dismissals, effective_ipl_w

res = bat_elig.apply(mix_bat_stats, axis=1)
bat_elig["eff_runs"], bat_elig["eff_balls"], bat_elig["eff_dismissals"], bat_elig["ipl_weight"] = \
    zip(*res)

bat_elig["bayes_sr"] = (
    (prior_sr/100 * prior_w + bat_elig["eff_runs"]) / (prior_w + bat_elig["eff_balls"]) * 100
).round(2)
bat_elig["bayes_avg"] = (
    (prior_avg * prior_d + bat_elig["eff_runs"]) / (prior_d + bat_elig["eff_dismissals"]).clip(1)
).round(2)

bat_elig["sr_improvement"] = bat_elig["comb_press_sr"] - bat_elig["all_sr"]

# --- RR-PRI ---
scaler = MinMaxScaler()
bat_elig["c_bayes_sr"] = scaler.fit_transform(bat_elig[["bayes_sr"]])
bat_elig["c_bayes_avg"] = scaler.fit_transform(bat_elig[["bayes_avg"]].clip(upper=120))
sr_imp = bat_elig["sr_improvement"].clip(-50, 50)
bat_elig["c_sr_imp"] = (sr_imp - sr_imp.min()) / (sr_imp.max() - sr_imp.min() + 1e-6)
bat_elig["c_cons"] = scaler.fit_transform(bat_elig["cons"].fillna(0.5).values.reshape(-1,1))
bat_elig["c_success"] = scaler.fit_transform(bat_elig["success_rate"].values.reshape(-1,1))
max_log = np.log(bat_elig["comb_press_balls"].max() + 1)
bat_elig["c_sample"] = np.log(bat_elig["comb_press_balls"] + 1) / max_log

bat_elig["rr_pri"] = (
    0.25 * bat_elig["c_bayes_sr"] +
    0.20 * bat_elig["c_bayes_avg"] +
    0.15 * bat_elig["c_sr_imp"] +
    0.20 * bat_elig["c_cons"] +
    0.10 * bat_elig["c_success"] +
    0.10 * bat_elig["c_sample"]
) * 100
bat_elig["rr_pri"] = bat_elig["rr_pri"].round(2)
bat_ranked = bat_elig.sort_values("rr_pri", ascending=False).reset_index(drop=True)
bat_ranked["rank"] = bat_ranked.index + 1

bat_ranked.to_csv(f"{BASE}/outputs/rankings/batters_final_ranked.csv", index=False)

print("\n=== FINAL TOP 10 BATTERS UNDER PRESSURE ===")
print(f"{'Rank':<5} {'Player':<28} {'P-SR':>7} {'BSR':>7} {'P-AVG':>7} {'IPL%':>6} {'Consist':>8} {'RR-PRI':>8}")
for i, r in bat_ranked.head(10).iterrows():
    print(f"  #{r['rank']:<3} {r['player_name']:<28} {r['comb_press_sr']:>7.1f} {r['bayes_sr']:>7.1f} "
          f"{r['comb_press_avg']:>7.1f} {r['ipl_weight']:>6.0%} {r['cons']:>8.3f} {r['rr_pri']:>8.2f}")


# =============================================================
#  BOWLING FINAL
# =============================================================
bowl_base = bbb[(bbb["is_ipl"] | bbb["is_t20i"]) & ~bbb["is_super_over"]]
bowl_press_all = bowl_base[bowl_base["bowling_pressure_score"] >= PRESS_THRESH]

all_bowl = agg_bowler(bowl_base, "all")
ipl_press_bowl = agg_bowler(bowl_press_all[bowl_press_all["is_ipl"]], "ipl_press")
t20i_press_bowl = agg_bowler(bowl_press_all[bowl_press_all["is_t20i"]], "t20i_press")
comb_press_bowl = agg_bowler(bowl_press_all, "comb_press")

def bowl_consistency(sub):
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
    cons["bowl_cons"] = 1 / (1 + cons["eco_cv"])
    cons["good_spell_rate"] = cons["good_spells"] / cons["n"]
    return cons[["bowler_id","bowl_cons","good_spell_rate"]]

bowl_cons = bowl_consistency(bowl_press_all)

bowling = all_bowl.merge(ipl_press_bowl, on="bowler_id", how="left")
bowling = bowling.merge(t20i_press_bowl, on="bowler_id", how="left")
bowling = bowling.merge(comb_press_bowl, on="bowler_id", how="left")
bowling = bowling.merge(bowl_cons, on="bowler_id", how="left")
bowling = bowling.merge(name_map, left_on="bowler_id", right_on="player_id", how="left")
bowling = bowling.fillna(0)

# ELIGIBILITY: total press balls >= 60, total matches >= 10, IPL press balls >= 20
bowling["ELIGIBLE"] = (
    (bowling["comb_press_balls"] >= 60) &
    (bowling["all_matches"] >= 10) &
    (bowling["ipl_press_balls"] >= 20)
)
bowl_elig = bowling[bowling["ELIGIBLE"]].copy()
print(f"\nEligible bowlers: {len(bowl_elig)}")

# Competition-mixed stats
prior_eco_rpb = 9.0 / 6; prior_eco_w = 60.0

def mix_bowl_stats(row):
    ipl_b = row["ipl_press_balls"]
    t20i_b = row["t20i_press_balls"]
    total_b = ipl_b + t20i_b
    if total_b < 1:
        return 0, 0, 0, 0
    ipl_frac = ipl_b / total_b
    eff_ipl_w = min(ipl_frac * 1.4, 1.0)
    eff_t20i_w = 1 - eff_ipl_w
    runs = eff_ipl_w * row["ipl_press_runs"] + eff_t20i_w * row["t20i_press_runs"]
    balls = eff_ipl_w * ipl_b + eff_t20i_w * t20i_b
    wickets = eff_ipl_w * row["ipl_press_wickets"] + eff_t20i_w * row["t20i_press_wickets"]
    return runs, balls, wickets, eff_ipl_w

res_b = bowl_elig.apply(mix_bowl_stats, axis=1)
bowl_elig["eff_runs"], bowl_elig["eff_balls"], bowl_elig["eff_wickets"], bowl_elig["ipl_weight"] = \
    zip(*res_b)

prior_runs_total = prior_eco_rpb * prior_eco_w
bowl_elig["bayes_eco"] = (
    (prior_runs_total + bowl_elig["eff_runs"]) / (prior_eco_w + bowl_elig["eff_balls"]) * 6
).round(2)

bowl_elig["eco_improvement"] = bowl_elig["all_economy"] - bowl_elig["comb_press_economy"]

# --- BOWLING RR-PRI ---
eco_inv = 1 / bowl_elig["bayes_eco"].clip(0.1)
bowl_elig["c_eco"] = scaler.fit_transform(eco_inv.values.reshape(-1,1))

wpo = bowl_elig["comb_press_wickets"] / bowl_elig["comb_press_overs"].clip(0.01)
bowl_elig["c_wkts"] = scaler.fit_transform(wpo.values.reshape(-1,1))

eco_imp = bowl_elig["eco_improvement"].clip(-4, 4)
bowl_elig["c_eco_imp"] = (eco_imp - eco_imp.min()) / (eco_imp.max() - eco_imp.min() + 1e-6)

bowl_elig["c_cons"] = scaler.fit_transform(bowl_elig["bowl_cons"].fillna(0.5).values.reshape(-1,1))
bowl_elig["c_spell"] = scaler.fit_transform(bowl_elig["good_spell_rate"].values.reshape(-1,1))
max_log_b = np.log(bowl_elig["comb_press_balls"].max() + 1)
bowl_elig["c_sample"] = np.log(bowl_elig["comb_press_balls"] + 1) / max_log_b

bowl_elig["rr_pri"] = (
    0.25 * bowl_elig["c_eco"] +
    0.20 * bowl_elig["c_wkts"] +
    0.15 * bowl_elig["c_eco_imp"] +
    0.20 * bowl_elig["c_cons"] +
    0.10 * bowl_elig["c_spell"] +
    0.10 * bowl_elig["c_sample"]
) * 100
bowl_elig["rr_pri"] = bowl_elig["rr_pri"].round(2)

bowl_ranked = bowl_elig.sort_values("rr_pri", ascending=False).reset_index(drop=True)
bowl_ranked["rank"] = bowl_ranked.index + 1
bowl_ranked.to_csv(f"{BASE}/outputs/rankings/bowlers_final_ranked.csv", index=False)

print("\n=== FINAL TOP 10 BOWLERS UNDER PRESSURE ===")
print(f"{'Rank':<5} {'Player':<28} {'P-ECO':>7} {'BECO':>7} {'P-WKT':>6} {'IPL%':>6} {'Consist':>8} {'RR-PRI':>8}")
for i, r in bowl_ranked.head(10).iterrows():
    print(f"  #{r['rank']:<3} {r['player_name']:<28} {r['comb_press_economy']:>7.2f} {r['bayes_eco']:>7.2f} "
          f"{r['comb_press_wickets']:>6.0f} {r['ipl_weight']:>6.0%} {r['bowl_cons']:>8.3f} {r['rr_pri']:>8.2f}")

# Save final Excel
top10_bat_final = bat_ranked.head(10)
top10_bowl_final = bowl_ranked.head(10)

with pd.ExcelWriter(f"{BASE}/outputs/rankings/RR_SuperRR_FINAL_Rankings.xlsx") as writer:
    top10_bat_final.to_excel(writer, sheet_name="Top10_Batters", index=False)
    top10_bowl_final.to_excel(writer, sheet_name="Top10_Bowlers", index=False)
    bat_ranked.head(50).to_excel(writer, sheet_name="All_Batters_Top50", index=False)
    bowl_ranked.head(50).to_excel(writer, sheet_name="All_Bowlers_Top50", index=False)

print("\nAll final ranking files saved.")
print(f"Batters file: {BASE}/outputs/rankings/batters_final_ranked.csv")
print(f"Bowlers file: {BASE}/outputs/rankings/bowlers_final_ranked.csv")
print(f"Excel: {BASE}/outputs/rankings/RR_SuperRR_FINAL_Rankings.xlsx")

