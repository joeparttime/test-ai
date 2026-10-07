# test-ai
# GPV baseline lab

A small, fully synthetic project for practicing how to tell a real payment-volume incident from normal variation. No data from any employer is used.

## Run it

```
pip install pandas numpy
python generate_data.py   # two years of 5-minute payment volume plus 7 known incidents
python baseline.py        # builds an expected-volume baseline and scores a detection rule
```

## What it models

Daily and weekly cycles, a late-year seasonal peak, year-over-year growth, holidays, and noise. The incidents in `incidents.csv` are the ground truth.

## What to try

1. Change `DROP_THRESHOLD` and `SUSTAIN` in `baseline.py`. Watch missed incidents trade off against false alarms.
2. Look at the false alarms. Which dates are they on, and why does a 4-week baseline fail there?
3. Improve the baseline (for example, add same-period-last-year with growth) and rerun.
4. Write up what you changed and what you would never automate.

## Notes

Parameters are illustrative, not recommendations.
