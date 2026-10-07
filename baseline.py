"""Build an expected-volume baseline and test a simple detection rule against known incidents.

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
