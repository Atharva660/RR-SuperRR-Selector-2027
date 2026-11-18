"""Generate comprehensive visualisations for the report."""
import sys
sys.path.insert(0, r"c:\Atharva\RR-SuperRR-Selector-2027\src")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

BASE = r"c:\Atharva\RR-SuperRR-Selector-2027"
CHARTS = f"{BASE}/outputs/charts"

bbb = pd.read_parquet(f"{BASE}/data/processed/bbb_with_pressure.parquet")
bat_ranked = pd.read_csv(f"{BASE}/outputs/rankings/batters_final_ranked.csv")
bowl_ranked = pd.read_csv(f"{BASE}/outputs/rankings/bowlers_final_ranked.csv")

# Color scheme - Rajasthan Royals
RR_PINK = "#E0115F"
RR_BLUE = "#002366"
RR_GOLD = "#FFD700"
DARK_BG = "#0A0E1A"
ACCENT = "#00D4FF"

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor': '#0F1629',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': '#2A3A6A',
    'grid.color': '#1A2550',
    'grid.alpha': 0.5,
    'font.family': 'sans-serif',
})

# ============================================================
# CHART 1: Pressure Score Distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('PRESSURE INTENSITY DISTRIBUTION', 
             fontsize=16, fontweight='bold', color=RR_GOLD, y=1.01)

ipl_bbb = bbb[bbb["is_ipl"] & ~bbb["is_super_over"]]

ax1 = axes[0]
bat_scores = ipl_bbb["batting_pressure_score"].dropna()
ax1.hist(bat_scores, bins=50, color=RR_PINK, alpha=0.85, edgecolor='none')
ax1.axvline(50, color=RR_GOLD, linestyle='--', linewidth=2, label='High Pressure (50+)')
ax1.axvline(75, color='white', linestyle='--', linewidth=1.5, alpha=0.7, label='Extreme Pressure (75+)')
ax1.set_title('Batting Pressure Score Distribution (IPL)', color=ACCENT, fontsize=13, fontweight='bold')
ax1.set_xlabel('Batting Pressure Score', fontsize=11)
ax1.set_ylabel('Delivery Count', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
bowl_scores = ipl_bbb["bowling_pressure_score"].dropna()
ax2.hist(bowl_scores, bins=50, color=RR_BLUE, alpha=0.85, edgecolor='none')
ax2.axvline(50, color=RR_GOLD, linestyle='--', linewidth=2, label='High Pressure (50+)')
ax2.axvline(75, color='white', linestyle='--', linewidth=1.5, alpha=0.7, label='Extreme Pressure (75+)')
ax2.set_title('Bowling Pressure Score Distribution (IPL)', color=ACCENT, fontsize=13, fontweight='bold')
ax2.set_xlabel('Bowling Pressure Score', fontsize=11)
ax2.set_ylabel('Delivery Count', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CHARTS}/01_pressure_distribution.png", dpi=150, bbox_inches='tight',
            facecolor=DARK_BG)
plt.close()
print("Chart 1 saved")

# ============================================================
# CHART 2: Top 10 Batters - RR-PRI vs Pressure SR
# ============================================================
top10_bat = bat_ranked.head(10)

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(DARK_BG)

colors = [RR_PINK if i == 0 else RR_BLUE for i in range(len(top10_bat))]
bars = ax.barh(range(len(top10_bat)), top10_bat["rr_pri"], color=colors, edgecolor='none', height=0.6)

for i, (_, row) in enumerate(top10_bat.iterrows()):
    ax.text(row["rr_pri"] + 0.5, i, f'  {row["rr_pri"]:.1f}', va='center', 
            color='white', fontsize=11, fontweight='bold')
    ax.text(0.3, i, f'#{int(row["rank"])}', va='center', ha='left',
            color=RR_GOLD, fontsize=10, fontweight='bold')

ax.set_yticks(range(len(top10_bat)))
ax.set_yticklabels([r["player_name"] for _, r in top10_bat.iterrows()], fontsize=12)
ax.set_xlabel('RR Pressure Reliability Index (RR-PRI)', fontsize=12)
ax.set_title('TOP 10 BATTERS UNDER PRESSURE\nRR Pressure Reliability Index', 
             fontsize=15, fontweight='bold', color=RR_GOLD, pad=15)
ax.invert_yaxis()
ax.grid(True, axis='x', alpha=0.3)
ax.set_xlim(0, top10_bat["rr_pri"].max() + 15)

# Add Bayes SR annotation
for i, (_, row) in enumerate(top10_bat.iterrows()):
    ax.text(top10_bat["rr_pri"].max() + 8, i, f'B-SR: {row["bayes_sr"]:.0f}', 
            va='center', color=ACCENT, fontsize=9)

ax.text(top10_bat["rr_pri"].max() + 8, -0.7, 'Bayes SR', 
        va='center', color=ACCENT, fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{CHARTS}/02_top10_batters_pri.png", dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Chart 2 saved")

# ============================================================
# CHART 3: Top 10 Bowlers - RR-PRI
# ============================================================
top10_bowl = bowl_ranked.head(10)

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(DARK_BG)

colors = [RR_PINK if i == 0 else '#1A3A7A' for i in range(len(top10_bowl))]
bars = ax.barh(range(len(top10_bowl)), top10_bowl["rr_pri"], color=colors, edgecolor='none', height=0.6)

for i, (_, row) in enumerate(top10_bowl.iterrows()):
    ax.text(row["rr_pri"] + 0.5, i, f'  {row["rr_pri"]:.1f}', va='center',
            color='white', fontsize=11, fontweight='bold')
    ax.text(0.3, i, f'#{int(row["rank"])}', va='center', ha='left',
            color=RR_GOLD, fontsize=10, fontweight='bold')

ax.set_yticks(range(len(top10_bowl)))
ax.set_yticklabels([r["player_name"] for _, r in top10_bowl.iterrows()], fontsize=12)
ax.set_xlabel('RR Pressure Reliability Index (RR-PRI)', fontsize=12)
ax.set_title('TOP 10 BOWLERS UNDER PRESSURE\nRR Pressure Reliability Index',
             fontsize=15, fontweight='bold', color=RR_GOLD, pad=15)
ax.invert_yaxis()
ax.grid(True, axis='x', alpha=0.3)
ax.set_xlim(0, top10_bowl["rr_pri"].max() + 15)

for i, (_, row) in enumerate(top10_bowl.iterrows()):
    ax.text(top10_bowl["rr_pri"].max() + 7, i, f'B-ECO: {row["bayes_eco"]:.2f}',
            va='center', color=ACCENT, fontsize=9)

ax.text(top10_bowl["rr_pri"].max() + 7, -0.7, 'Bayes ECO', 
        va='center', color=ACCENT, fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{CHARTS}/03_top10_bowlers_pri.png", dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Chart 3 saved")

# ============================================================
# CHART 4: Pressure vs Normal Performance Comparison (Batters)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(DARK_BG)

x = np.arange(len(top10_bat))
width = 0.35

normal_sr = top10_bat["all_sr"].values
pressure_sr = top10_bat["comb_press_sr"].values

bars1 = ax.bar(x - width/2, normal_sr, width, label='Normal SR', color=RR_BLUE, alpha=0.85, edgecolor='none')
bars2 = ax.bar(x + width/2, pressure_sr, width, label='Pressure SR', color=RR_PINK, alpha=0.85, edgecolor='none')

ax.set_title('NORMAL vs PRESSURE STRIKE RATE — Top 10 Batters', 
             fontsize=14, fontweight='bold', color=RR_GOLD)
ax.set_xlabel('Player', fontsize=11)
ax.set_ylabel('Strike Rate', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([r["player_name"].split()[-1] for _, r in top10_bat.iterrows()], 
                    rotation=25, ha='right', fontsize=10)
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
ax.axhline(130, color=RR_GOLD, linestyle='--', linewidth=1, alpha=0.5, label='T20 Average ~130')

plt.tight_layout()
plt.savefig(f"{CHARTS}/04_batting_pressure_vs_normal.png", dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Chart 4 saved")

# ============================================================
# CHART 5: Bowling Pressure vs Normal Economy
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(DARK_BG)

x = np.arange(len(top10_bowl))
normal_eco = top10_bowl["all_economy"].values
pressure_eco = top10_bowl["comb_press_economy"].values

bars1 = ax.bar(x - width/2, normal_eco, width, label='Normal Economy', color=RR_BLUE, alpha=0.85)
bars2 = ax.bar(x + width/2, pressure_eco, width, label='Pressure Economy', color=RR_PINK, alpha=0.85)

ax.set_title('NORMAL vs PRESSURE ECONOMY — Top 10 Bowlers',
             fontsize=14, fontweight='bold', color=RR_GOLD)
ax.set_xlabel('Player', fontsize=11)
ax.set_ylabel('Economy Rate', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels([r["player_name"].split()[-1] for _, r in top10_bowl.iterrows()],
                    rotation=25, ha='right', fontsize=10)
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
ax.axhline(9.0, color=RR_GOLD, linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig(f"{CHARTS}/05_bowling_pressure_vs_normal.png", dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Chart 5 saved")

# ============================================================
# CHART 6: Pressure Score by Phase (IPL matches)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle('PRESSURE DYNAMICS BY INNINGS PHASE (IPL)', fontsize=15, fontweight='bold', color=RR_GOLD)

ipl = bbb[bbb["is_ipl"] & ~bbb["is_super_over"]].copy()
ipl["phase"] = pd.cut(ipl["overid"], bins=[0,6,15,20], labels=["Powerplay (1-6)", "Middle (7-15)", "Death (16-20)"])

# Batting pressure by phase
bat_phase = ipl.groupby("phase")["batting_pressure_score"].mean()
axes[0].bar(bat_phase.index, bat_phase.values, color=[RR_BLUE, '#1A5A8A', RR_PINK], edgecolor='none')
axes[0].set_title('Average Batting Pressure by Phase', color=ACCENT, fontsize=13, fontweight='bold')
axes[0].set_ylabel('Average Pressure Score', fontsize=11)
axes[0].set_xlabel('Match Phase', fontsize=11)
axes[0].grid(True, axis='y', alpha=0.3)
for i, v in enumerate(bat_phase.values):
    axes[0].text(i, v + 0.5, f'{v:.1f}', ha='center', color='white', fontsize=12, fontweight='bold')

# Bowling pressure by phase
bowl_phase = ipl.groupby("phase")["bowling_pressure_score"].mean()
axes[1].bar(bowl_phase.index, bowl_phase.values, color=[RR_BLUE, '#1A5A8A', RR_PINK], edgecolor='none')
axes[1].set_title('Average Bowling Pressure by Phase', color=ACCENT, fontsize=13, fontweight='bold')
axes[1].set_ylabel('Average Pressure Score', fontsize=11)
axes[1].set_xlabel('Match Phase', fontsize=11)
axes[1].grid(True, axis='y', alpha=0.3)
for i, v in enumerate(bowl_phase.values):
    axes[1].text(i, v + 0.5, f'{v:.1f}', ha='center', color='white', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{CHARTS}/06_pressure_by_phase.png", dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Chart 6 saved")

# ============================================================
# CHART 7: Consistency Scatter - Batters
# ============================================================
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor(DARK_BG)

elig_bat = bat_ranked[bat_ranked["comb_press_balls"] >= 50].copy()
sc = ax.scatter(elig_bat["comb_press_sr"], elig_bat["cons"], 
                c=elig_bat["rr_pri"], cmap='plasma', s=elig_bat["comb_press_balls"] / 2,
                alpha=0.7, edgecolors='none')

# Annotate top 10
for _, row in bat_ranked.head(10).iterrows():
    ax.annotate(row["player_name"].split()[-1], 
                (row["comb_press_sr"], row["cons"]),
                fontsize=8, color='white', fontweight='bold',
                xytext=(5, 5), textcoords='offset points')

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('RR-PRI Score', color='white', fontsize=10)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

ax.set_xlabel('Pressure Strike Rate', fontsize=12)
ax.set_ylabel('Consistency Score', fontsize=12)
ax.set_title('BATTING PRESSURE PERFORMANCE vs CONSISTENCY\n(bubble size = balls faced under pressure)',
             fontsize=13, fontweight='bold', color=RR_GOLD)
ax.grid(True, alpha=0.3)
ax.axvline(130, color=RR_GOLD, linestyle='--', alpha=0.4, linewidth=1)
ax.axhline(0.5, color=RR_GOLD, linestyle='--', alpha=0.4, linewidth=1)

plt.tight_layout()
plt.savefig(f"{CHARTS}/07_batting_consistency_scatter.png", dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Chart 7 saved")

# ============================================================
# CHART 8: Season-wise IPL batting pressure SR
# ============================================================
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(DARK_BG)

ipl_bbb_press = bbb[bbb["is_ipl"] & (bbb["batting_pressure_score"] >= 50) & bbb["is_legal_ball"]].copy()
season_sr = ipl_bbb_press.groupby("season").agg(
    total_runs=("batter_runs","sum"),
    total_balls=("is_legal_ball","sum")
).reset_index()
season_sr["avg_pressure_sr"] = season_sr["total_runs"] / season_sr["total_balls"] * 100

ax.plot(season_sr["season"], season_sr["avg_pressure_sr"], 
        'o-', color=RR_PINK, linewidth=3, markersize=10, markerfacecolor=RR_GOLD)
ax.fill_between(season_sr["season"], season_sr["avg_pressure_sr"], 
                alpha=0.2, color=RR_PINK)

for _, r in season_sr.iterrows():
    ax.text(r["season"], r["avg_pressure_sr"] + 0.5, f'{r["avg_pressure_sr"]:.0f}',
            ha='center', color='white', fontsize=11, fontweight='bold')

ax.set_title('IPL AVERAGE STRIKE RATE UNDER PRESSURE BY SEASON', 
             fontsize=14, fontweight='bold', color=RR_GOLD)
ax.set_xlabel('IPL Season', fontsize=12)
ax.set_ylabel('Average Pressure Strike Rate', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xticks(season_sr["season"])

plt.tight_layout()
plt.savefig(f"{CHARTS}/08_season_pressure_sr.png", dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Chart 8 saved")

print("\nAll 8 charts generated successfully!")

