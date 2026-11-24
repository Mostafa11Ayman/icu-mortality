# Project notes

Running log of decisions and key numbers. One entry per session.

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-24 | Dataset: PhysioNet 2012 (set-a/b/c) | Open access, no credentialing, has SAPS-I for a ready baseline |
| 2026-11-24 | Label: `In-hospital_death` from the outcomes files | This is the challenge's official target |
| 2026-11-24 | Metrics: AUROC, AUPRC, calibration | AUROC for ranking, AUPRC because deaths are the minority class, calibration because clinicians read probabilities |
| 2026-11-24 | Split: train/tune on a+b, test on c | Whole sets never mix, so no stay leaks between train and test; c is touched once at the end |

## Key numbers

| Quantity | Value | Source |
|---|---|---|
| Stays per set | a 4,000 / b 4,000 / c 4,000 (12,000) | 01_setup_and_framing |
| Mortality rate per set | a 13.9% / b 14.2% / c 14.6% (14.2% overall) | 01_setup_and_framing |
| SAPS-I AUROC / AUPRC on set-c | 0.658 [0.635, 0.681] / 0.251 [0.223, 0.284] (n=3,834) | 01_setup_and_framing |
| SOFA AUROC / AUPRC on set-c | 0.636 [0.611, 0.661] / 0.254 [0.224, 0.288] (n=3,878) | 01_setup_and_framing |
| Missing scores (-1) | SAPS-I 518 stays, SOFA 403 stays | 01_setup_and_framing |
| Mortality by SAPS-I decile | 4.0% (lowest) to 33.6% (highest) | 01_setup_and_framing |
| Dev folds / test | 5 x 1,600 stays at 14.0-14.1% / 4,000 at 14.6% | 01_setup_and_framing |

## Session log

### 2026-11-24
- Created repo scaffold and `01_setup_and_framing.ipynb`.
- Set up `.venv` on Python 3.14.3 (the only version installed; all packages installed fine) and registered the `icu-mortality` kernel.
- PhysioNet offers set-c only as `set-c.tar.gz`, so `src/data_io.py` now reads .zip, .tar.gz or an extracted folder (`find_source`).
- Ran notebook 01. Integrity is clean: 4,000 files and 4,000 outcome rows per set, 0 mismatches, 0 cross-set overlap.
- Baseline to beat on set-c: SAPS-I AUROC 0.658, AUPRC 0.251. AUPRC floor (random) is about 0.146.
- Next: `02_wrangling.ipynb`, long format to one row per stay (min/max/mean/last/count + missingness indicators).
