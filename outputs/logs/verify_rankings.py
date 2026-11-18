"""
Verify final rankings - check data integrity for all 20 final players.
"""
import pandas as pd
import numpy as np

BASE = r"c:\Atharva\RR-SuperRR-Selector-2027"
bbb = pd.read_parquet(f"{BASE}/data/processed/bbb_with_pressure.parquet")
pm = pd.read_csv(f"{BASE}/data/processed/player_mapping_clean.csv")

bat = pd.read_csv(f"{BASE}/outputs/rankings/batters_final_ranked.csv")
bowl = pd.read_csv(f"{BASE}/outputs/rankings/bowlers_final_ranked.csv")

print("=" * 80)
print("BATTING TOP 10 - DETAILED VERIFICATION")
print("=" * 80)
press_bat = bbb[(bbb["batting_pressure_score"] >= 50) & bbb["is_legal_ball"]]

for _, r in bat.head(10).iterrows():
    pid = r["striker_id"]
    player_del = press_bat[press_bat["striker_id"] == pid]
    ipl_del = player_del[player_del["is_ipl"]]
    t20i_del = player_del[player_del["is_t20i"]]
    
    # Verify player exists in mapping
    in_map = pm[pm["player_id"] == pid]["player_name"].values
    
    print(f"\nRank {int(r['rank'])}: {r['player_name']} (ID: {pid})")
    print(f"  In player mapping: {list(in_map)}")
    print(f"  IPL matches: {r['ipl_press_matches']:.0f}, pressure balls (IPL): {r['ipl_press_balls']:.0f}")
    print(f"  T20I pressure balls: {r.get('t20i_press_balls', 0):.0f}")
    print(f"  Combined pressure: runs={r['comb_press_runs']:.0f}, balls={r['comb_press_balls']:.0f}, "
          f"dismissals={r['comb_press_dismissals']:.0f}")
    print(f"  Bayes SR: {r['bayes_sr']:.1f} | Consistency: {r['cons']:.3f} | RR-PRI: {r['rr_pri']:.2f}")
    
    # Double-check raw data
    raw_runs = ipl_del["batter_runs"].sum()
    raw_balls = ipl_del["is_legal_ball"].sum()
    print(f"  Raw check (IPL press): runs={raw_runs}, balls={raw_balls}, SR={raw_runs/max(raw_balls,1)*100:.1f}")

print()
print("=" * 80)
print("BOWLING TOP 10 - DETAILED VERIFICATION")
print("=" * 80)
press_bowl = bbb[(bbb["bowling_pressure_score"] >= 50)]

for _, r in bowl.head(10).iterrows():
    pid = r["bowler_id"]
    player_del = press_bowl[press_bowl["bowler_id"] == pid]
    ipl_del = player_del[player_del["is_ipl"]]
    
    in_map = pm[pm["player_id"] == pid]["player_name"].values
    
    print(f"\nRank {int(r['rank'])}: {r['player_name']} (ID: {pid})")
    print(f"  In player mapping: {list(in_map)}")
    print(f"  IPL pressure balls: {r['ipl_press_balls']:.0f}, matches: {r['ipl_press_matches']:.0f}")
    print(f"  Combined pressure: runs={r['comb_press_runs']:.0f}, overs={r['comb_press_overs']:.1f}, "
          f"wkts={r['comb_press_wickets']:.0f}, eco={r['comb_press_economy']:.2f}")
    print(f"  Bayes ECO: {r['bayes_eco']:.2f} | Consistency: {r['bowl_cons']:.3f} | RR-PRI: {r['rr_pri']:.2f}")
    
    # Raw check
    raw_runs = ipl_del["total_runs"].sum()
    raw_balls = ipl_del["is_legal_ball"].sum()
    print(f"  Raw check (IPL press): runs={raw_runs}, balls={raw_balls}, eco={raw_runs/max(raw_balls/6,0.01):.2f}")

# Check Corbin Bosch - appears top of bowlers, verify
print()
print("CORBIN BOSCH deep dive:")
cb = bbb[(bbb["bowling_pressure_score"] >= 50) & (bbb["bowler_name"] == "Corbin Bosch")]
print(cb.groupby("is_ipl")["match_id"].nunique())
print(cb.groupby(["match_id","is_ipl"]).agg(rc=("total_runs","sum"), lb=("is_legal_ball","sum")).reset_index())

