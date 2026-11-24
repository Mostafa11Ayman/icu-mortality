"""Generate SYNTHETIC example records for the Streamlit demo.

These are invented patients with plausible values, written in the PhysioNet 2012
record format (Time,Parameter,Value). They are NOT real PhysioNet records, so they
can be committed to git. Run from the repo root:

    python app/make_examples.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "examples"
RNG = np.random.default_rng(2012)


def hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def series(lines, param, start, end, every, centre, spread, lo=None, hi=None, trend=0.0, decimals=0):
    """Add a noisy series from `start` to `end` hours, one value every `every` minutes."""
    t = start * 60
    n_total = max((end - start) * 60 // every, 1)
    i = 0
    while t <= end * 60:
        v = centre + trend * i / n_total + RNG.normal(0, spread)
        if lo is not None:
            v = max(v, lo)
        if hi is not None:
            v = min(v, hi)
        lines.append((t + int(RNG.integers(0, 10)), param, round(float(v), decimals)))
        t += every
        i += 1


def write(name, descriptors, lines):
    OUT.mkdir(exist_ok=True)
    rows = ["Time,Parameter,Value"]
    rows += [f"00:00,{k},{v}" for k, v in descriptors.items()]
    rows += [f"{hhmm(t)},{p},{v}" for t, p, v in sorted(lines, key=lambda r: (r[0], r[1]))]
    (OUT / name).write_text("\n".join(rows) + "\n")
    print(f"wrote {OUT / name} ({len(rows) - 1} rows)")


# 1. Low risk: 62-year-old after planned cardiac surgery, extubated after a few hours.
lines = []
series(lines, "HR", 0, 48, 60, 82, 5, trend=-6)
series(lines, "SysABP", 0, 20, 60, 118, 8)
series(lines, "DiasABP", 0, 20, 60, 60, 5)
series(lines, "MAP", 0, 20, 60, 79, 5)
series(lines, "NISysABP", 20, 48, 120, 124, 8)
series(lines, "NIDiasABP", 20, 48, 120, 64, 5)
series(lines, "NIMAP", 20, 48, 120, 84, 5)
series(lines, "Temp", 0, 48, 120, 36.6, 0.3, trend=0.5, decimals=1)
series(lines, "GCS", 0, 6, 120, 10, 1, lo=3, hi=15)
series(lines, "GCS", 8, 48, 240, 15, 0, lo=3, hi=15)
series(lines, "Urine", 0, 48, 60, 110, 30, lo=10)
series(lines, "RespRate", 8, 48, 120, 17, 2)
series(lines, "FiO2", 0, 6, 120, 0.5, 0.05, lo=0.21, hi=1, decimals=2)
series(lines, "pH", 0, 6, 120, 7.40, 0.03, decimals=2)
series(lines, "PaO2", 0, 6, 120, 180, 30)
series(lines, "PaCO2", 0, 6, 120, 40, 3)
series(lines, "HCT", 0, 48, 720, 30, 1.5, decimals=1)
series(lines, "Lactate", 0, 8, 240, 1.6, 0.3, lo=0.5, decimals=1)
for p, c, s in [("BUN", 16, 2), ("Creatinine", 0.9, 0.1), ("HCO3", 24, 1), ("Na", 138, 1.5), ("K", 4.2, 0.2),
                ("Glucose", 130, 15), ("Mg", 2.2, 0.2), ("Platelets", 160, 15), ("WBC", 11, 1.5)]:
    series(lines, p, 1, 48, 720, c, s, decimals=1 if c < 10 else 0)
lines += [(5, "MechVent", 1), (65, "MechVent", 1), (185, "MechVent", 1)]
write("example_low_risk_cardiac_surgery.txt",
      {"RecordID": 900001, "Age": 62, "Gender": 1, "Height": 175.3, "ICUType": 2, "Weight": 82.0}, lines)

# 2. High risk: 79-year-old medical ICU patient in septic shock, ventilated, failing kidneys.
lines = []
series(lines, "HR", 0, 48, 30, 118, 8, trend=6)
series(lines, "SysABP", 0, 48, 30, 92, 9, trend=-8)
series(lines, "DiasABP", 0, 48, 30, 48, 5)
series(lines, "MAP", 0, 48, 30, 62, 5, trend=-4)
series(lines, "Temp", 0, 48, 60, 38.6, 0.4, decimals=1)
series(lines, "GCS", 0, 48, 120, 7, 1, lo=3, hi=15, trend=-3)
series(lines, "Urine", 0, 48, 60, 25, 10, lo=0)
series(lines, "FiO2", 0, 48, 120, 0.7, 0.08, lo=0.21, hi=1, decimals=2)
series(lines, "pH", 0, 48, 240, 7.26, 0.03, trend=-0.04, decimals=2)
series(lines, "PaO2", 0, 48, 240, 75, 10)
series(lines, "PaCO2", 0, 48, 240, 33, 3)
series(lines, "SaO2", 0, 48, 240, 92, 2, hi=100)
series(lines, "Lactate", 0, 48, 360, 4.8, 0.8, lo=0.5, trend=1.5, decimals=1)
series(lines, "HCT", 0, 48, 480, 27, 1.5, decimals=1)
for p, c, s, tr in [("BUN", 62, 4, 10), ("Creatinine", 2.8, 0.2, 0.6), ("HCO3", 16, 1, -2), ("Na", 133, 2, 0),
                    ("K", 5.1, 0.3, 0), ("Glucose", 190, 25, 0), ("Mg", 1.8, 0.2, 0), ("Platelets", 85, 10, -20),
                    ("WBC", 21, 3, 3)]:
    series(lines, p, 1, 48, 480, c, s, trend=tr, decimals=1 if c < 10 else 0)
for p, c in [("Albumin", 2.2), ("Bilirubin", 2.6), ("ALT", 95), ("AST", 140), ("ALP", 160)]:
    series(lines, p, 2, 26, 1440, c, c * 0.05, decimals=1 if c < 10 else 0)
lines += [(t, "MechVent", 1) for t in range(10, 2880, 240)]
write("example_high_risk_medical_sepsis.txt",
      {"RecordID": 900002, "Age": 79, "Gender": 0, "Height": -1, "ICUType": 3, "Weight": 68.0}, lines)

# 3. Intermediate: 70-year-old after abdominal surgery, not ventilated, mild kidney stress.
lines = []
series(lines, "HR", 0, 48, 60, 96, 7)
series(lines, "NISysABP", 0, 48, 60, 112, 10)
series(lines, "NIDiasABP", 0, 48, 60, 58, 6)
series(lines, "NIMAP", 0, 48, 60, 76, 6)
series(lines, "Temp", 0, 48, 120, 37.6, 0.4, decimals=1)
series(lines, "GCS", 0, 48, 240, 14, 0.6, lo=3, hi=15)
series(lines, "Urine", 0, 48, 60, 55, 15, lo=0)
series(lines, "RespRate", 0, 48, 60, 22, 3)
series(lines, "Lactate", 0, 24, 480, 2.4, 0.4, lo=0.5, decimals=1)
series(lines, "HCT", 0, 48, 720, 31, 1.5, decimals=1)
for p, c, s in [("BUN", 32, 3), ("Creatinine", 1.5, 0.15), ("HCO3", 21, 1), ("Na", 136, 2), ("K", 4.0, 0.3),
                ("Glucose", 150, 20), ("Mg", 1.9, 0.2), ("Platelets", 210, 20), ("WBC", 14, 2)]:
    series(lines, p, 1, 48, 720, c, s, decimals=1 if c < 10 else 0)
write("example_intermediate_surgical.txt",
      {"RecordID": 900003, "Age": 70, "Gender": 1, "Height": 170.2, "ICUType": 4, "Weight": 77.0}, lines)
