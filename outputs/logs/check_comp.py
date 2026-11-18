import pandas as pd

# Towhid Hridoy is ONLY in T20I (Bangladesh), not IPL - check if we should restrict
bbb = pd.read_parquet(r'c:\Atharva\RR-SuperRR-Selector-2027\data\processed\bbb_with_pressure.parquet')
press = bbb[(bbb['batting_pressure_score'] >= 50) & (bbb['is_legal_ball'])]

hridoy = press[press['striker_id'] == 6651]
print('Hridoy is_ipl:', hridoy['is_ipl'].any(), 'is_t20i:', hridoy['is_t20i'].any())

# Check competition split for all top 10 batters
all_bat = pd.read_csv(r'c:\Atharva\RR-SuperRR-Selector-2027\outputs\rankings\all_batters_ranked.csv')
top10 = all_bat.head(10)

for _, row in top10.iterrows():
    pid = row['striker_id']
    player_press = press[press['striker_id'] == pid]
    ipl_balls = player_press[player_press['is_ipl']]['is_legal_ball'].sum()
    t20i_balls = player_press[player_press['is_t20i']]['is_legal_ball'].sum()
    print(f"{row['player_name']:25s}: IPL pressure balls={ipl_balls:4d}, T20I pressure balls={t20i_balls:4d}")

print()
# Same for bowlers
all_bowl = pd.read_csv(r'c:\Atharva\RR-SuperRR-Selector-2027\outputs\rankings\all_bowlers_ranked.csv')
press_bowl = bbb[(bbb['bowling_pressure_score'] >= 50)]
top10b = all_bowl.head(10)

for _, row in top10b.iterrows():
    pid = row['bowler_id']
    player_press = press_bowl[press_bowl['bowler_id'] == pid]
    ipl_balls = player_press[player_press['is_ipl']]['is_legal_ball'].sum()
    t20i_balls = player_press[player_press['is_t20i']]['is_legal_ball'].sum()
    print(f"{row['player_name']:25s}: IPL pressure balls={ipl_balls:4d}, T20I pressure balls={t20i_balls:4d}")

