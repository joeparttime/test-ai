"""Baseline v3: v2 plus a seasonal correction taken from last year's shape.

The 4-week baseline lags the seasonal ramp (it is centered about 2.5 weeks back).
v3 corrects that lag: last year's smoothed daily level on this date divided by
its level 18 days earlier. Holiday multipliers are measured against the same
weekday one week before and after, so they no longer absorb seasonality.
The first year has no "last year", so it gets no seasonal correction.
Compares v1, v2 and v3 over the whole dataset, then again from WARM_UP onward, when
a full year of history exists. Parameters are ILLUSTRATIVE.
"""
import warnings

import numpy as np
import pandas as pd

SUSTAIN = 2
WEEKS_BACK = 4
LAG_DAYS = 18            # roughly the center of a 4-week window
THRESHOLDS = [0.10, 0.15, 0.25]
WARM_UP = pd.Timestamp("2024-01-15")   # first date with a full year of history

HOLIDAYS = {
    "new_years_day": ["2023-01-01", "2024-01-01"],
    "thanksgiving": ["2023-11-23", "2024-11-28"],
    "black_friday": ["2023-11-24", "2024-11-29"],
    "christmas_eve": ["2023-12-24", "2024-12-24"],
    "christmas_day": ["2023-12-25", "2024-12-25"],
    "day_after_christmas": ["2023-12-26", "2024-12-26"],
}
holiday_of = {pd.Timestamp(d): n for n, ds in HOLIDAYS.items() for d in ds}

gpv = pd.read_csv("gpv.csv", parse_dates=["timestamp"], index_col="timestamp")["gpv"]
incidents = pd.read_csv("incidents.csv", parse_dates=["start", "end"])
per_week = 7 * 288
day = gpv.index.normalize()
is_holiday = day.isin(list(holiday_of))


def rolling_median(series):
    past = np.vstack([series.shift(per_week * k).values for k in range(1, WEEKS_BACK + 1)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return pd.Series(np.nanmedian(past, axis=0), index=series.index)


expected_v1 = rolling_median(gpv)
normal = rolling_median(gpv.mask(is_holiday))

# daily totals with holidays removed, smoothed over 7 days (weekday-neutral level)
daily = gpv.resample("D").sum()
daily_clean = daily.mask(daily.index.isin(list(holiday_of)))
level = daily_clean.rolling(7, center=True, min_periods=5).mean().interpolate(limit=5, limit_area="inside")


def seasonal_factor(d):
    last_year = d - pd.Timedelta(days=364)
    ref = last_year - pd.Timedelta(days=LAG_DAYS)
    a, b = level.get(last_year, np.nan), level.get(ref, np.nan)
    return a / b if pd.notna(a) and pd.notna(b) else 1.0


season_by_day = pd.Series({d: seasonal_factor(d) for d in daily.index})
season = pd.Series(season_by_day.reindex(day).values, index=gpv.index)

# v2 multipliers: ratio to the 4-week baseline (absorbs the seasonal lag)
mult_v2, mult_v3 = pd.Series(1.0, index=gpv.index), pd.Series(1.0, index=gpv.index)
learned_v2, learned_v3 = {}, {}
for d in sorted(holiday_of):
    name, m = holiday_of[d], day == d
    if name in learned_v2:
        mult_v2[m] = np.mean(learned_v2[name])
    if normal[m].notna().all():
        learned_v2.setdefault(name, []).append(gpv[m].sum() / normal[m].sum())
    # v3: ratio to the same weekday one week before/after (holidays excluded)
    if name in learned_v3:
        mult_v3[m] = np.mean(learned_v3[name])
    neighbors = [daily_clean.get(d + pd.Timedelta(days=k), np.nan) for k in (-7, 7)]
    if not np.all(np.isnan(neighbors)):
        learned_v3.setdefault(name, []).append(daily[d] / np.nanmean(neighbors))

expected_v2 = normal * mult_v2
expected_v3 = normal * season * mult_v3


def evaluate(expected, threshold, since=None):
    exp = expected.dropna()
    below = (gpv.loc[exp.index] / exp - 1 <= -threshold).astype(int)
    sustained = below.rolling(SUSTAIN).sum() == SUSTAIN
    alerts, in_run = [], False
    for ts, hit in sustained.items():
        if hit and not in_run:
            fire, in_run = ts, True
        elif not hit and in_run:
            alerts.append((fire, prev))
            in_run = False
        prev = ts
    if in_run:
        alerts.append((fire, prev))
    if since is not None:
        alerts = [(a, b) for a, b in alerts if a >= since]
    matched, missed = set(), []
    scored = incidents if since is None else incidents[incidents.start >= since]
    for _, inc in scored.iterrows():
        hits = [i for i, (a, b) in enumerate(alerts) if a <= inc.end and b >= inc.start]
        matched.update(hits)
        if not hits:
            missed.append(inc.start)
    false_alarms = [a for i, a in enumerate(alerts) if i not in matched]
    return len(scored) - len(missed), false_alarms, missed


print("Holiday multipliers learned (v2 vs v3):")
for name in learned_v2:
    print(f"  {name}: v2 {np.mean(learned_v2[name]):.2f}   v3 {np.mean(learned_v3[name]):.2f}")

print(f"\n{'threshold':>9} | {'v1 caught/false':>16} | {'v2 caught/false':>16} | {'v3 caught/false':>16}")
for t in THRESHOLDS:
    cells = []
    for exp in (expected_v1, expected_v2, expected_v3):
        c, f, _ = evaluate(exp, t)
        cells.append(f"{c}/{len(incidents)}  {len(f):>5}")
    print(f"{t:>9.0%} | {cells[0]:>16} | {cells[1]:>16} | {cells[2]:>16}")

n_late = int((incidents.start >= WARM_UP).sum())
print(f"\nFrom {WARM_UP:%Y-%m-%d} onward ({n_late} incidents):")
print(f"{'threshold':>9} | {'v1 caught/false':>16} | {'v2 caught/false':>16} | {'v3 caught/false':>16}")
for t in THRESHOLDS:
    cells = []
    for exp in (expected_v1, expected_v2, expected_v3):
        c, f, _ = evaluate(exp, t, since=WARM_UP)
        cells.append(f"{c}/{n_late}  {len(f):>5}")
    print(f"{t:>9.0%} | {cells[0]:>16} | {cells[1]:>16} | {cells[2]:>16}")"""Build an expected-volume baseline and test a simple detection rule against known incidents.

Baseline v1: for each 5-minute slot, the median of the same weekday/time slot
over the previous 4 weeks. The parameters below are ILLUSTRATIVE. Tune them.
"""
import warnings

import numpy as np
import pandas as pd

DROP_THRESHOLD = 0.25   # alert when volume is this far below expected
SUSTAIN = 2             # ...for this many consecutive 5-minute intervals
WEEKS_BACK = 4

gpv = pd.read_csv("gpv.csv", parse_dates=["timestamp"], index_col="timestamp")["gpv"]
incidents = pd.read_csv("incidents.csv", parse_dates=["start", "end"])

per_week = 7 * 288
past = np.vstack([gpv.shift(per_week * k).values for k in range(1, WEEKS_BACK + 1)])
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    expected = pd.Series(np.nanmedian(past, axis=0), index=gpv.index).dropna()
actual = gpv.loc[expected.index]
deviation = actual / expected - 1

below = (deviation <= -DROP_THRESHOLD).astype(int)
sustained = below.rolling(SUSTAIN).sum() == SUSTAIN

# group consecutive True values into alerts
alerts, in_run = [], False
for ts, hit in sustained.items():
    if hit and not in_run:
        fire, in_run = ts, True
    elif not hit and in_run:
        alerts.append((fire, prev))
        in_run = False
    prev = ts
if in_run:
    alerts.append((fire, prev))

print(f"Rule: volume {DROP_THRESHOLD:.0%}+ below expected for {SUSTAIN * 5} min "
      f"(baseline: median of last {WEEKS_BACK} same-weekday slots)\n")

caught, matched = 0, set()
for _, inc in incidents.iterrows():
    hits = [i for i, (a, b) in enumerate(alerts) if a <= inc.end and b >= inc.start]
    if hits:
        caught += 1
        matched.update(hits)
        delay = (alerts[hits[0]][0] - inc.start).total_seconds() / 60
        print(f"caught   {inc.start:%Y-%m-%d %H:%M}  drop {inc['drop']:.0%}  fired {delay:.0f} min after start")
    else:
        print(f"MISSED   {inc.start:%Y-%m-%d %H:%M}  drop {inc['drop']:.0%}")

false_alarms = [a for i, a in enumerate(alerts) if i not in matched]
print(f"\ncaught {caught}/{len(incidents)} incidents, {len(false_alarms)} false alarms")
by_day = pd.Series([a.strftime("%Y-%m-%d") for a, _ in false_alarms]).value_counts().sort_index()
print("\nFalse alarms by date:")
for day, n in by_day.items():
    print(f"  {day}: {n}")
