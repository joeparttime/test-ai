"""Generate two years of SYNTHETIC payment volume (5-minute intervals) with injected incidents.

Nothing here comes from a real company. The incident list is the ground truth
you use to score a detection rule.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
idx = pd.date_range("2023-01-01", "2024-12-31 23:55", freq="5min")

hour = idx.hour + idx.minute / 60
daily = 0.35 + 0.45 * np.exp(-((hour - 13) / 4.5) ** 2) + 0.35 * np.exp(-((hour - 20) / 2.5) ** 2)
weekly = np.where(idx.dayofweek >= 5, 1.15, 1.0)
doy = idx.dayofyear
yearly = 1 + 0.25 * np.exp(-((doy - 330) / 18) ** 2) + 0.10 * np.exp(-((doy - 355) / 10) ** 2)
trend = 1 + 0.18 * ((idx - idx[0]).days / 365)
noise = rng.normal(1, 0.03, len(idx))

holiday = pd.Series(1.0, index=idx)
for day, mult in {
    "2023-11-23": 0.6, "2023-11-24": 1.5, "2023-12-24": 0.8, "2023-12-25": 0.5,
    "2023-12-26": 0.8, "2024-01-01": 0.7, "2024-11-28": 0.6, "2024-11-29": 1.5,
    "2024-12-24": 0.8, "2024-12-25": 0.5, "2024-12-26": 0.8,
}.items():
    holiday[holiday.index.strftime("%Y-%m-%d") == day] = mult

gpv = pd.Series(250_000 * daily * weekly * yearly * trend * noise * holiday.values, index=idx)

# (start, duration in minutes, fraction of volume lost)
incidents = [
    ("2023-03-14 14:05", 25, 0.55),
    ("2023-07-02 03:30", 40, 0.35),
    ("2023-11-24 11:10", 30, 0.60),
    ("2024-02-09 19:20", 20, 0.40),
    ("2024-06-18 09:00", 45, 0.30),
    ("2024-04-22 16:40", 30, 0.15),
    ("2024-10-05 22:15", 35, 0.50),
]
rows = []
for start, minutes, drop in incidents:
    s = pd.Timestamp(start)
    e = s + pd.Timedelta(minutes=minutes)
    gpv[(gpv.index >= s) & (gpv.index < e)] *= 1 - drop
    rows.append({"start": s, "end": e, "drop": drop})

gpv.rename("gpv").to_csv("gpv.csv", index_label="timestamp")
pd.DataFrame(rows).to_csv("incidents.csv", index=False)
print(f"wrote gpv.csv ({len(gpv)} rows) and incidents.csv ({len(rows)} incidents)")
