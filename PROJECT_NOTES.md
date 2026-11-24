# Project notes

Running log of decisions and key numbers. One entry per session.

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-24 | Dataset: PhysioNet 2012 (set-a/b/c) | Open access, no credentialing, has SAPS-I for a ready baseline |
| 2026-11-24 | Label: `In-hospital_death` from the outcomes files | This is the challenge's official target |
| 2026-11-24 | Metrics: AUROC, AUPRC, calibration | AUROC for ranking, AUPRC because deaths are the minority class, calibration because clinicians read probabilities |
| 2026-11-24 | Split: train/tune on a+b, test on c | Whole sets never mix, so no stay leaks between train and test; c is touched once at the end |
| 2026-11-24 | Cleaning: remove only physically impossible values (range table `VALID_RANGES` in `src/features.py`) | Extreme but possible values carry the mortality signal; dropping them would hide the sickest patients |
| 2026-11-24 | pH entered as 735 rescaled to 7.35, not dropped | Obvious missing decimal point, the value is recoverable |
| 2026-11-24 | Drop 2,403 set-c rows with a blank parameter name | No way to know which variable they belong to |
| 2026-11-24 | Features per variable: min, max, mean, last, count, has_ (36 variables, 225 features total) | Worst moment, overall level, state at 48 h, monitoring intensity, and whether it was measured at all |
| 2026-11-24 | No imputation or scaling in wrangling | They learn from the data, so they must be fitted inside the training folds to avoid leakage |
| 2026-11-24 | SAPS-I/SOFA stored next to features but never used as features; Length_of_stay/Survival dropped | Baseline comparison only / known only after the outcome |

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
| Raw measurements | 5,285,818 rows, all within 48 h | 02_wrangling |
| Values removed by cleaning | 8,335 (0.16%), mostly invasive BP of 0 | 02_wrangling |
| Feature table | 12,000 x 225 (36 vars x 6 + 9 static) | 02_wrangling |
| Unknown height / admission weight | 47.9% / 8.4% of stays | 02_wrangling |
| Stays with no time series at all | 13 (all survivors) | 02_wrangling |
| Mortality if lactate measured vs not (dev) | 18.1% vs 9.0% | 02_wrangling |

## Session log

### 2026-11-24
- Created repo scaffold and `01_setup_and_framing.ipynb`.
- Set up `.venv` on Python 3.14.3 (the only version installed; all packages installed fine) and registered the `icu-mortality` kernel.
- PhysioNet offers set-c only as `set-c.tar.gz`, so `src/data_io.py` now reads .zip, .tar.gz or an extracted folder (`find_source`).
- Ran notebook 01. Integrity is clean: 4,000 files and 4,000 outcome rows per set, 0 mismatches, 0 cross-set overlap.
- Baseline to beat on set-c: SAPS-I AUROC 0.658, AUPRC 0.251. AUPRC floor (random) is about 0.146.
- Pushed the repo to GitHub.
- Wrote `02_wrangling.ipynb` and `src/features.py`. Added `read_set()` to `src/data_io.py` (reads a whole archive in one pass; 21 s for all 12,000 stays).
- Found two data quirks: real measurements are mixed in between the minute-0 descriptor rows (so descriptors are picked by name, not file position), and set-c has 2,403 rows with a blank parameter name.
- Missingness is informative: lactate, pH, albumin, liver tests and troponins are measured more often in patients who die.
- Next: `03_eda.ipynb` on the dev set (distributions, missingness patterns, mortality by age and ICU type).
