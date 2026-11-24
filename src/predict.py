"""Score one PhysioNet-format record with the frozen final model.

Used by the Streamlit demo (app/streamlit_app.py) and notebook 06. It reuses the
exact cleaning and feature code from src/features.py, so a record is treated the
same way as in training.

Because the final model is a logistic regression and its scaler centres the dev
data at 0, each feature's contribution to the log-odds is exactly
coefficient x transformed value. Summing a variable's features gives an exact
per-variable explanation (identical to linear SHAP with the dev mean as baseline).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import features as F

STATS = ("min", "max", "mean", "last", "count")


class RecordError(ValueError):
    """The uploaded file cannot be scored (bad format or no measurements)."""


def load_artifacts(root: Path):
    model = joblib.load(root / "models" / "final_model.joblib")
    meta = json.loads((root / "models" / "final_model.json").read_text())
    return model, meta


def parse_record(text: str, record_id: int = 0) -> pd.DataFrame:
    """PhysioNet record text (Time,Parameter,Value) -> long table like data_io.read_set."""
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:  # noqa: BLE001
        raise RecordError(f"Could not read the file as CSV: {e}") from e
    if list(df.columns[:3]) != ["Time", "Parameter", "Value"]:
        raise RecordError("Expected the header 'Time,Parameter,Value' (PhysioNet 2012 format).")
    df = df.dropna(subset=["Time"])
    try:
        hh_mm = df["Time"].astype(str).str.split(":", expand=True).astype(int)
    except Exception as e:  # noqa: BLE001
        raise RecordError("Time must be HH:MM (hours:minutes since ICU admission).") from e
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    if df["Value"].isna().all():
        raise RecordError("No numeric values found in the Value column.")
    return pd.DataFrame({
        "RecordID": record_id, "set": "upload",
        "minutes": hh_mm[0] * 60 + hh_mm[1],
        "Parameter": df["Parameter"].values, "Value": df["Value"].values,
    })


def record_features(long: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same pipeline as notebook 02: descriptors, cleaning, features. Returns (row, cleaning report)."""
    desc, ts = F.split_descriptors(long)
    if desc.empty or "Age" not in desc or desc["Age"].isna().all():
        raise RecordError("The record has no Age descriptor at time 00:00.")
    for col in F.DESCRIPTORS[1:]:
        if col not in desc:
            desc[col] = np.nan
    ts_clean, report = F.clean_time_series(ts)
    if ts_clean.empty:
        raise RecordError("The record has no usable measurements in the first 48 h. "
                          "Stays without measurements were excluded from this project, so it cannot be scored.")
    feats = F.build_features(ts_clean, desc)
    row = feats.reindex(columns=feature_names)
    for c in row.columns:   # variables never seen in this record: count 0, not measured
        if c.endswith("_count") or c.startswith("has_") or c.startswith("ICUType_") or c == "MechVent_ever":
            row[c] = row[c].fillna(0).astype(int)
    return row, report


def predict_risk(model, row: pd.DataFrame) -> float:
    return float(model.predict_proba(row)[:, 1][0])


def variable_of(feature: str) -> str:
    if feature.startswith("has_"):
        return feature[4:]
    if feature.startswith("ICUType_"):
        return "ICU type"
    head, _, tail = feature.rpartition("_")
    return head if tail in STATS else feature


def contributions(model, row: pd.DataFrame) -> tuple[float, pd.Series]:
    """Exact log-odds contributions per variable.

    Returns (baseline log-odds = intercept, Series variable -> contribution),
    with baseline + sum(contributions) = the patient's log-odds.
    """
    pre, lr = model[:-1], model[-1]
    xt = pre.transform(row)[0]
    contrib = pd.Series(lr.coef_[0] * xt, index=pre.get_feature_names_out())
    by_var = contrib.groupby(variable_of).sum()
    return float(lr.intercept_[0]), by_var.reindex(by_var.abs().sort_values(ascending=False).index)


def score_text(model, meta: dict, text: str) -> dict:
    """Full path used by the app: text -> features -> risk, flag, explanation."""
    row, report = record_features(parse_record(text), meta["features"])
    risk = predict_risk(model, row)
    base, contrib = contributions(model, row)
    thr = meta["threshold"]["value"]
    return {"risk": risk, "flag": risk >= thr, "threshold": thr, "baseline_logodds": base,
            "contributions": contrib, "features": row, "cleaning": report}
