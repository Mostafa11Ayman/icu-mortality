"""Cleaning and per-stay feature building for the PhysioNet 2012 long table.

Input: the long table from data_io.read_set (RecordID, set, minutes,
Parameter, Value). Output: one row per RecordID.

Nothing here learns from the data (no imputation, no scaling), so it can be
run on all 12,000 stays at once without leaking information between splits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_MINUTES = 48 * 60

DESCRIPTORS = ["RecordID", "Age", "Gender", "Height", "ICUType", "Weight"]
ICU_TYPES = {1: "coronary", 2: "cardiac_surgery", 3: "medical", 4: "surgical"}

# Physiologically possible ranges. A value is kept when LOW < value <= HIGH.
# Only physically impossible values are removed (recording errors, -1 codes,
# zeros for things that cannot be zero). Extreme but possible values are kept,
# because they carry the mortality signal.
# Urine is not listed: 0 mL is real (anuria). GCS and FiO2 had no bad values.
VALID_RANGES: dict[str, tuple[float, float]] = {
    # vitals
    "HR": (0, 300),          # 0 bpm in a recorded vital stream is a lead-off artefact
    "Temp": (25, 45),        # below 25 C are entries like -17.8 or 0
    "RespRate": (0, 100),    # 0 breaths/min recorded for a monitored patient
    "SysABP": (0, 300), "DiasABP": (0, 300), "MAP": (0, 300),
    "NISysABP": (0, 300), "NIDiasABP": (0, 300), "NIMAP": (0, 300),
    "Weight": (20, 300),     # -1 and 0 are "unknown"; 472 kg is a typo
    # blood gases
    "pH": (6.5, 8.0),        # after rescaling 735 -> 7.35, see fix_ph_units
    "PaO2": (0, 800), "PaCO2": (0, 200), "SaO2": (0, 100),
}

DESCRIPTOR_RANGES: dict[str, tuple[float, float]] = {
    "Height": (100, 250),    # -1 is unknown; 1.8 or 462 cm are not human heights
    "Weight": (20, 300),
}


def split_descriptors(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate the static descriptor rows from the time series.

    Descriptors sit at minute 0, but real minute-0 measurements can be mixed in
    between them, so position in the file is not reliable. Weight is both a
    descriptor and a later time series: the first Weight row at minute 0 is the
    descriptor, any other Weight row is a measurement.
    Returns (descriptors wide table indexed by RecordID, time-series long table).
    """
    first_of_param = long.groupby(["RecordID", "Parameter"]).cumcount() == 0
    is_desc = (
        (long["minutes"] == 0)
        & long["Parameter"].isin(DESCRIPTORS)
        & first_of_param
    )
    desc = (long[is_desc]
            .pivot_table(index="RecordID", columns="Parameter", values="Value", aggfunc="first")
            .drop(columns="RecordID", errors="ignore")
            .reindex(columns=DESCRIPTORS[1:]))   # a record missing a descriptor gets NaN
    desc = desc.replace(-1, np.nan)
    for col, (lo, hi) in DESCRIPTOR_RANGES.items():
        desc[col] = desc[col].where((desc[col] > lo) & (desc[col] <= hi))
    ts = long[~is_desc & (long["Parameter"] != "RecordID")].copy()
    return desc, ts


def fix_ph_units(ts: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """pH entered without the decimal point (735 for 7.35, 73.5 for 7.35).

    Rescales values that land in 6.5-8.0 after dividing by 10 or 100.
    Returns the fixed table and how many values were rescaled.
    """
    ts = ts.copy()
    is_ph = ts["Parameter"] == "pH"
    v = ts.loc[is_ph, "Value"]
    fixed = v.copy()
    for factor in (10, 100):
        scaled = v / factor
        ok = (v > 8.0) & (scaled > 6.5) & (scaled <= 8.0)
        fixed[ok] = scaled[ok]
    n = int((fixed != v).sum())
    ts.loc[is_ph, "Value"] = fixed
    return ts, n


def clean_time_series(ts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the first 48 h and drop impossible values.

    Returns (cleaned long table, report with one row per rule).
    """
    report = []
    n_before = len(ts)
    ts = ts[ts["Parameter"].notna()]
    report.append({"rule": "blank parameter name (cannot tell what was measured)",
                   "parameter": "none", "removed": n_before - len(ts)})
    n_before = len(ts)
    ts = ts[ts["minutes"] <= WINDOW_MINUTES]
    report.append({"rule": "minutes > 2880 (after 48 h)", "parameter": "all",
                   "removed": n_before - len(ts)})

    ts, n_ph = fix_ph_units(ts)
    report.append({"rule": "pH rescaled (/10 or /100), not removed", "parameter": "pH",
                   "removed": 0, "rescaled": n_ph})

    keep = pd.Series(True, index=ts.index)
    for param, (lo, hi) in VALID_RANGES.items():
        rows = ts["Parameter"] == param
        v = ts.loc[rows, "Value"]
        bad = ~((v > lo) & (v <= hi))
        keep[bad[bad].index] = False
        report.append({"rule": f"outside ({lo}, {hi}]", "parameter": param,
                       "removed": int(bad.sum())})
    ts = ts[keep]
    report = pd.DataFrame(report).fillna({"rescaled": 0}).astype({"rescaled": int})
    return ts, report


def build_features(ts: pd.DataFrame, desc: pd.DataFrame) -> pd.DataFrame:
    """One row per RecordID.

    Per time-series variable: min, max, mean, last, count and has_<var>.
    MechVent is only ever recorded as 1, so it becomes one flag (ever ventilated).
    Static: Age, Gender, Height, Weight (admission), ICU type one-hot.
    Missing stays NaN on purpose: imputation is fitted later inside training folds.
    """
    ids = desc.index
    series = ts[ts["Parameter"] != "MechVent"].sort_values(["RecordID", "minutes"])
    g = series.groupby(["RecordID", "Parameter"])["Value"]
    stats = g.agg(["min", "max", "mean", "last", "count"]).unstack("Parameter")
    stats.columns = [f"{var}_{stat}" for stat, var in stats.columns]
    stats = stats.reindex(ids)

    variables = sorted(series["Parameter"].unique())
    for var in variables:
        c = f"{var}_count"
        stats[c] = stats[c].fillna(0).astype(int)
        stats[f"has_{var}"] = (stats[c] > 0).astype(int)

    ordered = [f"{v}_{s}" for v in variables for s in ("min", "max", "mean", "last", "count")]
    ordered += [f"has_{v}" for v in variables]
    stats = stats[ordered]

    vent = ts.loc[ts["Parameter"] == "MechVent", "RecordID"].unique()
    static = pd.DataFrame(index=ids)
    static["Age"] = desc["Age"]
    static["Gender"] = desc["Gender"]          # 0 female, 1 male, NaN unknown
    static["Height"] = desc["Height"]
    static["Weight_admit"] = desc["Weight"]
    static["MechVent_ever"] = ids.isin(vent).astype(int)
    for code, name in ICU_TYPES.items():
        static[f"ICUType_{name}"] = (desc["ICUType"] == code).astype(int)

    return static.join(stats)
