"""
01_dst_consistency_check.py
========================================
Diagnostic script to verify timezone consistency and daylight saving time
boundary handling in your PM2.5 dataset.

Europe switches between UTC and CEST on:
  - Last Sunday of March (spring forward, 02:00→03:00)
  - Last Sunday of October (fall back, 03:00→02:00)

In 2020: March 29, October 25
In 2021: March 28, October 31
etc.

This script checks that your timestamps are properly aligned and that
no data integrity issues arise from DST transitions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = Path("features")
STATIONS = [
    "landshuter_allee", "stachus", "lothstrasse", "johanneskirchen",
    "augsburg_bourges", "augsburg_lfu", "andechs",
    "burghausen", "oberaudorf",
]

# DST transition dates (Europe, last Sunday of March & October)
DST_TRANSITIONS = {
    2020: ("2020-03-29", "2020-10-25"),
    2021: ("2021-03-28", "2021-10-31"),
    2022: ("2022-03-27", "2022-10-30"),
    2023: ("2023-03-26", "2023-10-29"),
    2024: ("2024-03-31", "2024-10-27"),
}


def check_dst_boundaries(df, station_name):
    """
    Verify no missing hours or duplicates around DST transitions.
    """
    issues = []

    # Check each DST transition
    for year, (spring, fall) in DST_TRANSITIONS.items():
        for transition_date in [spring, fall]:
            transition = pd.Timestamp(transition_date)

            # Look at ±3 hours around transition
            window_start = transition - pd.Timedelta(hours=3)
            window_end = transition + pd.Timedelta(hours=3)

            subset = df.loc[window_start:window_end]
            if len(subset) == 0:
                continue

            # Check for time gaps (expected hourly, so diff should be 1 hour)
            time_diffs = subset.index.to_series().diff().dt.total_seconds() / 3600

            # Identify anomalies (should all be 1 hour except maybe at DST boundary)
            anomalies = time_diffs[(time_diffs != 1.0) & (time_diffs.notna())]

            if len(anomalies) > 0:
                for t, diff in anomalies.items():
                    issues.append({
                        "station": station_name,
                        "transition": transition_date,
                        "timestamp": t,
                        "time_diff_hours": round(diff, 2),
                        "severity": "HIGH" if diff != 1.0 else "OK"
                    })

    return issues


def check_timezone_info(df, station_name):
    """
    Check if the index has timezone information.
    """
    has_tz = df.index.tz is not None
    tz_name = str(df.index.tz) if has_tz else "None (naive)"
    return {
        "station": station_name,
        "has_timezone_info": has_tz,
        "timezone": tz_name,
        "index_type": str(type(df.index))
    }


def check_hourly_regularity(df, station_name):
    """
    Overall check: are timestamps regularly spaced at 1-hour intervals?
    (outside of intentional missing data flags)
    """
    if len(df) < 2:
        return {"station": station_name, "n_records": len(df), "status": "TOO_SHORT"}

    time_diffs = df.index.to_series().diff().dt.total_seconds() / 3600

    # Count anomalies
    n_anomalies = (time_diffs != 1.0).sum()
    pct_anomalies = 100 * n_anomalies / len(time_diffs)

    # Get the distribution of intervals
    diff_counts = time_diffs.value_counts().sort_index().head(10)

    return {
        "station": station_name,
        "n_records": len(df),
        "n_hourly_anomalies": n_anomalies,
        "pct_anomalies": round(pct_anomalies, 2),
        "most_common_interval_hours": float(diff_counts.index[0]) if len(diff_counts) > 0 else None,
        "interval_distribution": diff_counts.to_dict(),
    }


def run_all_checks():
    """
    Load each station and run all DST/timezone checks.
    """
    print("\n" + "=" * 80)
    print(" DAYLIGHT SAVING TIME CONSISTENCY CHECK")
    print("=" * 80)

    all_dst_issues = []
    all_tz_info = []
    all_regularity = []

    for station in STATIONS:
        filepath = DATA_DIR / f"{station}_features.csv"

        if not filepath.exists():
            print(f"\n⚠ {station}: File not found at {filepath}")
            print("  (Run pipeline/03_feature_engineering.py first)")
            continue

        print(f"\n{'─' * 80}")
        print(f"Station: {station}")
        print(f"{'─' * 80}")

        try:
            # Load with timezone-aware parsing
            df = pd.read_csv(filepath, index_col="datetime", parse_dates=True)

            # Check 1: Timezone info
            tz_check = check_timezone_info(df, station)
            all_tz_info.append(tz_check)
            print(f"  Timezone info: {tz_check['timezone']}")
            print(f"  Index type: {tz_check['index_type']}")

            # Check 2: Hourly regularity
            regularity = check_hourly_regularity(df, station)
            all_regularity.append(regularity)
            print(f"  Records: {regularity['n_records']}")
            print(f"  Hourly anomalies: {regularity['n_hourly_anomalies']} "
                  f"({regularity['pct_anomalies']}%)")
            if regularity['most_common_interval_hours']:
                print(f"  Most common interval: {regularity['most_common_interval_hours']:.1f} hours")

            # Check 3: DST boundaries
            dst_issues = check_dst_boundaries(df, station)
            all_dst_issues.extend(dst_issues)

            if dst_issues:
                print(f"  ⚠ Found {len(dst_issues)} potential DST boundary issues:")
                for issue in dst_issues:
                    print(f"      {issue['transition']} → "
                          f"diff={issue['time_diff_hours']}h [{issue['severity']}]")
            else:
                print(f"  ✓ No DST boundary anomalies detected")

        except Exception as e:
            print(f"  ✗ ERROR reading data: {e}")
            continue

    # Summary report
    print("\n" + "=" * 80)
    print(" SUMMARY")
    print("=" * 80)

    print("\n1. TIMEZONE CONSISTENCY")
    print("   ─" * 40)
    tz_df = pd.DataFrame(all_tz_info)
    if not tz_df.empty:
        for _, row in tz_df.iterrows():
            status = "✓" if row['has_timezone_info'] else "⚠"
            print(f"   {status} {row['station']:<30} TZ={row['timezone']}")

    print("\n2. HOURLY REGULARITY")
    print("   ─" * 40)
    reg_df = pd.DataFrame(all_regularity)
    if not reg_df.empty:
        for _, row in reg_df.iterrows():
            status = "✓" if row['pct_anomalies'] < 1 else "⚠"
            print(f"   {status} {row['station']:<30} "
                  f"Anomalies: {row['pct_anomalies']:.2f}%")

    print("\n3. DST BOUNDARY ISSUES")
    print("   ─" * 40)
    if all_dst_issues:
        issue_df = pd.DataFrame(all_dst_issues)
        print(f"   Found {len(issue_df)} total anomalies across DST transitions:")
        for _, row in issue_df.iterrows():
            print(f"   ⚠ {row['station']:<25} {row['transition']:<12} "
                  f"@ {row['timestamp']} (diff={row['time_diff_hours']}h)")
    else:
        print("   ✓ No DST boundary anomalies detected")
        issue_df = pd.DataFrame(columns=["station", "transition", "timestamp", "time_diff_hours", "severity"])

    Path("results").mkdir(exist_ok=True)
    tz_df.to_csv(Path("results") / "dst_check_timezone.csv", index=False)
    reg_df.to_csv(Path("results") / "dst_check_hourly_regularity.csv", index=False)
    issue_df.to_csv(Path("results") / "dst_check_boundary_issues.csv", index=False)
    print("\n✓ Saved results/dst_check_timezone.csv, dst_check_hourly_regularity.csv, "
          "dst_check_boundary_issues.csv")

    print("\n" + "=" * 80)
    print(" INTERPRETATION")
    print("=" * 80)
    print("""
If you see ⚠ warnings here, it means:

1. **No timezone info (TZ=None):**
   Your index is timezone-naive (no UTC offset). This is OK if:
   - Both UBA and DWD data are consistently in the same timezone
   - Timestamps never jump by non-hourly intervals
   Consider adding a note to your thesis: "Timestamps are in local time
   without explicit UTC offset, but verified for consistency..."

2. **Hourly anomalies (>1%):**
   Time diffs are not always 1 hour. Check if this is intentional
   (missing data) or a DST bug. Missing data should be represented as
   NaN rows, not by skipping timestamps.

3. **DST boundary jumps:**
   At a true DST transition, expect ONE 2-hour gap (spring forward)
   or ONE duplicate hour (fall back). If you see random 2-hour gaps
   elsewhere, that's suspicious.

ACTION:
If anomalies > 1%, investigate further by examining the raw data
around March 29 and October 25, 2020 in particular.
    """)


if __name__ == "__main__":
    run_all_checks()