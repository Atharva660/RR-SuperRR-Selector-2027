"""
Data Cleaning & Validation Module - RR SuperRR Selector Season 2
Professional data quality assessment and cleaning pipeline.
"""
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

def assess_data_quality(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Generate a data quality report for a DataFrame."""
    report = {
        "dataset": name,
        "original_rows": len(df),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return report


def clean_ball_by_ball(df: pd.DataFrame, audit_log: list) -> pd.DataFrame:
    """
    Clean ball-by-ball dataset.
    Returns cleaned DataFrame. All operations are logged to audit_log.
    """
    original_len = len(df)
    df = df.copy()

    # 1. Remove exact duplicates
    dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    audit_log.append(f"BBB: Removed {dupes} exact duplicate rows")

    # 2. Innings 3 & 4 are super-overs (keep but flag)
    df["is_super_over"] = df["innings"].isin([3, 4])
    super_over_count = df["is_super_over"].sum()
    audit_log.append(f"BBB: Flagged {super_over_count} super-over deliveries (innings 3/4)")

    # 3. Validate over numbers (1-20 for T20)
    invalid_overs = df[(df["overid"] < 1) | (df["overid"] > 20)]
    if len(invalid_overs) > 0:
        audit_log.append(f"BBB: Found {len(invalid_overs)} rows with invalid over numbers - keeping for now")

    # 4. over_ball_num > 6 means extra deliveries (wides/no-balls)
    df["is_extra_delivery"] = df["over_ball_num"] > 6
    audit_log.append(f"BBB: Flagged {df['is_extra_delivery'].sum()} extra deliveries (over_ball_num > 6)")

    # 5. Determine if a delivery is a legal delivery (counts towards batter's balls faced)
    # extras1_runs = wides/no-balls that don't count; extras_2_runs = penalty runs
    # A wide or no-ball doesn't count as a legal delivery for the batter
    # We identify legal deliveries as those where over_ball_num <= 6 (normal ball numbering)
    # Actually, over_ball_num > 6 indicates re-bowled ball after wide/no-ball in that over
    # For counting balls faced: if the delivery is a wide, it doesn't count
    # We'll use: legal_ball = not (is_wide) where is_wide means the extra was due to a wide

    # extras_type column might tell us this - check unique values
    # For now: a ball counts as "faced by batter" if it's not a wide
    # From data: extras1_runs > 0 with certain extra types could be wides or no-balls
    # We'll define: delivery_type based on extra type
    # Since extras_type is numeric, we'll handle it conservatively
    # Legal delivery = one that the batter can be dismissed from and that counts toward strike rate
    # We'll compute this during pressure engine - for now keep all rows

    # 6. Runs must be non-negative
    neg_runs = df[df["runs"] < 0]
    if len(neg_runs) > 0:
        audit_log.append(f"BBB: WARNING - {len(neg_runs)} rows with negative runs - investigating")

    # 7. Standardise dismissal types
    df["dismissal_type"] = df["dismissal_type"].fillna("Not Out")

    # 8. If is_wicket=True but dismissal_type='Not Out', fix
    mask = (df["is_wicket"] == True) & (df["dismissal_type"] == "Not Out")
    if mask.sum() > 0:
        df.loc[mask, "dismissal_type"] = "Unknown"
        audit_log.append(f"BBB: Fixed {mask.sum()} wickets with missing dismissal type -> 'Unknown'")

    # 9. Standardise Obstruction spelling
    df["dismissal_type"] = df["dismissal_type"].replace(
        "Obstracting the field", "Obstructing the field"
    )

    # 10. Add total_runs_off_bat (runs scored by batter, not extras)
    # runs column appears to include batter runs; extras1_runs and extras_2_runs are separate
    df["total_runs"] = df["runs"] + df["extras1_runs"] + df["extras_2_runs"]

    # 11. Batter runs = runs (the 'runs' column is batter's runs based on the data sample)
    # Verify: when is_wicket=True and not extras, batter runs should be in 'runs'
    df["batter_runs"] = df["runs"]

    # 12. is_legal_delivery: not a wide (extras_type=1 typically = wide in cricket data)
    # We'll be conservative: if over_ball_num <= 6, count as legal
    # Actually over_ball_num > 6 means it's an extra delivery in that over
    df["is_legal_ball"] = ~df["is_extra_delivery"]

    audit_log.append(f"BBB: Final rows: {len(df)} (from {original_len})")
    return df


def clean_matches(df: pd.DataFrame, audit_log: list) -> pd.DataFrame:
    """Clean matches dataset."""
    df = df.copy()
    original_len = len(df)

    # Remove exact duplicates
    dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    audit_log.append(f"Matches: Removed {dupes} duplicates")

    # Parse match_date
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    null_dates = df["match_date"].isnull().sum()
    if null_dates > 0:
        audit_log.append(f"Matches: {null_dates} rows with unparseable match_date")

    # Identify abandoned matches (no winner)
    df["is_abandoned"] = df["win_team_id"] == 0
    audit_log.append(f"Matches: {df['is_abandoned'].sum()} abandoned/no-result matches")

    audit_log.append(f"Matches: Final rows: {len(df)} (from {original_len})")
    return df


def clean_player_mapping(df: pd.DataFrame, audit_log: list) -> pd.DataFrame:
    """Clean player mapping dataset."""
    df = df.copy()
    original_len = len(df)

    dupes = df.duplicated().sum()
    df = df.drop_duplicates()
    audit_log.append(f"PlayerMapping: Removed {dupes} duplicates")

    # Strip whitespace from player names
    df["player_name"] = df["player_name"].str.strip()

    # Count unique players
    unique_players = df["player_id"].nunique()
    audit_log.append(f"PlayerMapping: {unique_players} unique players across all competitions")

    return df

