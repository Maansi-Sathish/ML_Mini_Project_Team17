"""Person A: feature engineering. Public API: build_features(df, **flags).

Same function is used for train and test (fixed category lists -> identical columns).
Trees (Person B) can call build_features(df, onehot_buckets=False) to keep compact
numeric columns instead of dummies.
"""
import numpy as np
import pandas as pd

SEASONS = [1, 2, 3, 4]
WEATHERS = [1, 2, 3, 4]
HOURS = list(range(24))
MONTHS = list(range(1, 13))
DOWS = list(range(7))
TEMP_BINS = [-np.inf, 10, 15, 20, 25, 30, np.inf]
TEMP_LABELS = ["lt10", "10_15", "15_20", "20_25", "25_30", "gt30"]

WEEKDAY_PEAK = {7, 8, 9, 17, 18, 19}      # commute hours
WEEKEND_PEAK = set(range(10, 19))          # 10am - 6pm

# The 6 switchable ideas (used by src/ablation.py). Update defaults after the ablation.
ABLATION_FLAGS = ["onehot_buckets", "use_month", "use_dow", "drop_temp", "drop_holiday", "peak_hours"]


def _dummies(values, categories, prefix, index):
    s = pd.Series(pd.Categorical(values, categories=categories), index=index)
    return pd.get_dummies(s, prefix=prefix, dtype=float)


def build_features(df, onehot_buckets=True, use_month=True, use_dow=True,
                   drop_temp=False, drop_holiday=True, peak_hours=True):
    """Return a numeric feature DataFrame (no target / leakage columns).

    onehot_buckets : one-hot categoricals (hour, weather, ...) + temperature buckets
    use_month      : month replaces season
    use_dow        : add day of week
    drop_temp      : drop temp, keep atemp (they are ~collinear)
    drop_holiday   : drop the holiday flag
    peak_hours     : weekday / weekend peak-hour indicators
    """
    dt = pd.to_datetime(df["datetime"])
    idx = df.index
    hour, month, dow = dt.dt.hour, dt.dt.month, dt.dt.dayofweek
    workingday = df["workingday"].astype(int)

    out = pd.DataFrame(index=idx)
    out["year"] = dt.dt.year - 2011          # demand grew from 2011 -> 2012
    out["workingday"] = workingday
    out["humidity"] = df["humidity"]
    out["windspeed"] = df["windspeed"]
    out["atemp"] = df["atemp"]
    if not drop_temp:
        out["temp"] = df["temp"]
    if not drop_holiday:
        out["holiday"] = df["holiday"]
    if peak_hours:
        out["peak_weekday"] = ((workingday == 1) & hour.isin(WEEKDAY_PEAK)).astype(float)
        out["peak_weekend"] = ((workingday == 0) & hour.isin(WEEKEND_PEAK)).astype(float)

    cats = [("hour", hour, HOURS), ("weather", df["weather"], WEATHERS)]
    cats.append(("month", month, MONTHS) if use_month else ("season", df["season"], SEASONS))
    if use_dow:
        cats.append(("dow", dow, DOWS))

    parts = [out]
    for name, values, categories in cats:
        if onehot_buckets:
            parts.append(_dummies(values, categories, name, idx))
        else:
            parts.append(values.rename(name).astype(float).to_frame())

    if onehot_buckets:   # temperature buckets (roughly linear within each bucket)
        for col in (["atemp"] if drop_temp else ["temp", "atemp"]):
            b = pd.cut(df[col], TEMP_BINS, labels=TEMP_LABELS)
            parts.append(_dummies(b.astype(object), TEMP_LABELS, f"{col}_bin", idx))

    return pd.concat(parts, axis=1).astype(float)
