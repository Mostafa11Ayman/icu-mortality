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
| 2026-11-24 | EDA on the dev set (a+b) only | Looking at the test set would bias choices made later |
| 2026-11-24 | Plan to exclude the 13 stays with no measurements (10 dev, 3 test) and report them separately | All survivors with no SAPS-I either; they look like unfilled records, and a model could learn "no data = survives" |
| 2026-11-24 | Logistic regression will need imputation + has_ flags + log of skewed labs + scaling + L2, all inside CV | Missingness is informative, labs are heavily skewed, 97 feature pairs have \|r\| > 0.9 |
| 2026-11-24 | Imbalance: class weights / rank metrics, no oversampling | 6.1 survivors per death is moderate; oversampling distorts calibration |
| 2026-11-24 | Exclude the 13 stays with no measurements (10 dev, 3 test) from training and testing; report separately | Mostafa's decision after EDA: likely unfilled records, would teach "no data = survives" |
| 2026-11-24 | Nested CV: hyperparameters tuned with an inner 3-fold grid inside each outer training fold; outer folds = splits.csv | Picking settings on the same folds we score would flatter the CV result |
| 2026-11-24 | Selection rule (set before results): best pooled OOF AUPRC; simplest model not significantly worse (paired bootstrap) wins | Avoids picking a complex model for a gain that is just noise |
| 2026-11-24 | Final model: unweighted L2 logistic regression (C = 0.03) | Best AUPRC, tied with LightGBM, simpler and well calibrated |
| 2026-11-24 | No class weighting in the final model | Weighting gave the same ranking but inflated probabilities (mean 34% vs 14% observed) |
| 2026-11-24 | Commit models/final_model.joblib (0.02 MB) | Small enough for git; the Streamlit demo will load it |
| 2026-11-24 | Evaluation order: leakage gate on dev, then threshold on dev, then freeze (hash + JSON), then set-c once | Nothing learned from set-c can flow back into the model |
| 2026-11-24 | Decision threshold 0.139: highest threshold with sensitivity >= 0.80 on dev OOF | Missing a death costs more than reviewing a false alarm; Youden (0.128) gives almost the same point |
| 2026-11-24 | Subgroup flag rule: AUROC > 0.05 below overall or O/E outside 0.8-1.25 | Stated before scoring so it cannot be bent to the results |
| 2026-11-24 | SHAP read per variable (6 features summed) | Single linear-model SHAP values just mirror coefficients; correlated features split their weight |
| 2026-11-24 | Demo scores an uploaded PhysioNet-format record through the same `src/features.py` pipeline (`src/predict.py`) | One code path for notebooks and app; checked on 25 real records: identical risk |
| 2026-11-24 | Demo examples are synthetic (`app/make_examples.py`) | Raw PhysioNet records stay out of git |
| 2026-11-24 | Explanations in the app = coefficient x transformed value, summed per variable | Exact for a linear model (equals linear SHAP with the dev mean as baseline), no SHAP dependency at run time |
| 2026-11-24 | Pinned requirements to installed versions; dropped seaborn and xgboost (never imported) | Reproducibility; a clean re-run of notebooks 01-05 reproduces the model file byte-for-byte |
| 2026-11-24 | `split_descriptors` reindexes to all 5 descriptors | A record missing a descriptor gets NaN instead of crashing; no change for the training data (verified by identical hashes) |

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
| Dev class balance | 6,878 survived / 1,122 died (6.1 : 1) | 03_eda |
| Mortality by age band (dev) | <40 7.7%, 40-59 9.9%, 60-69 12.4%, 70-79 14.5%, 80+ 22.6% | 03_eda |
| Mortality by ICU type (dev) | cardiac surgery 4.6%, coronary 13.1%, surgical 14.6%, medical 19.2% | 03_eda |
| Age cap / minors | 308 stays at exactly 90; 14 stays aged 15-17 | 03_eda |
| Strongest single features (univariate AUROC, dev) | GCS_last 0.747, GCS_mean 0.694, BUN_last 0.691, Urine_mean 0.690 | 03_eda |
| SAPS-I AUROC on dev | 0.661 | 03_eda |
| Feature pairs with \|r\| > 0.9 | 97 (74 within the same variable) | 03_eda |
| Modeling set (dev) | 7,990 stays, 1,122 deaths, 225 features | 04_modeling |
| LogReg unweighted, OOF dev | AUROC 0.867 [0.855, 0.877], AUPRC 0.550 [0.518, 0.581], Brier 0.0868 | 04_modeling |
| LightGBM, OOF dev | AUROC 0.866, AUPRC 0.547, Brier 0.0871 | 04_modeling |
| Random forest, OOF dev | AUROC 0.850, AUPRC 0.494 (significantly worse) | 04_modeling |
| SAPS-I on the same dev stays | AUROC 0.661, AUPRC 0.241 | 04_modeling |
| Final model minus SAPS-I (paired bootstrap, dev) | AUROC +0.207 [+0.190, +0.224], AUPRC +0.310 [+0.282, +0.338] | 04_modeling |
| Leakage checks | max 2,880 min; no outcome columns; shuffled-label AUROC 0.50; last measurement 47.5 h (survivors) vs 47.6 h (deaths) | 05_evaluation |
| **TEST (set-c) model** | **AUROC 0.879 [0.864, 0.892], AUPRC 0.564 [0.523, 0.606]**, 3,997 stays, 585 deaths | 05_evaluation |
| TEST SAPS-I / SOFA | AUROC 0.658 / 0.636, AUPRC 0.251 / 0.254 | 05_evaluation |
| TEST model minus SAPS-I | AUROC +0.221 [+0.197, +0.244], AUPRC +0.310 [+0.270, +0.350] | 05_evaluation |
| TEST calibration | mean predicted 14.4% vs observed 14.6%, intercept +0.03, slope 0.95, Brier 0.089 | 05_evaluation |
| TEST at threshold 0.139 | sens 0.812, spec 0.796, PPV 0.406, NPV 0.961, 29% flagged, NNE 2.5 (SAPS-I: PPV 0.178, 68% flagged) | 05_evaluation |
| TEST weakest subgroups | age 80+ AUROC 0.829, coronary ICU 0.830 (none flagged) | 05_evaluation |
| Top variables (SHAP, test) | GCS, BUN, RespRate pattern, Age, Creatinine, Urine, MechVent | 05_evaluation |

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
- Wrote and ran `03_eda.ipynb` (dev set only). Figures: `reports/figures/03_*.png`.
- GCS at 48 h alone (AUROC 0.747) beats SAPS-I on dev (0.661). ICU type and age are the strongest static factors; gender barely matters.
- Missingness comes in blocks (lab panels, blood gas, arterial line vs cuff) and differs by ICU type. RespRate is mostly charted for non-ventilated patients, so missing RespRate means higher risk.
- Wrote `src/models.py` (pipelines, LightGBM with early stopping inside the training fold, nested CV, bootstrap metrics) and ran `04_modeling.ipynb` on dev.
- Logistic regression, LightGBM and balanced logistic regression are statistically tied (AUROC about 0.867, AUPRC about 0.55); random forest is behind. The final model is unweighted logistic regression, saved to `models/`.
- The model beats SAPS-I by about 0.21 AUROC on dev. Class weighting changed nothing in ranking but ruined calibration.
- Caveat: LightGBM chose the smallest trees in the grid in every fold (edge of the grid).
- Ran `05_evaluation.ipynb`. The leakage gate passed on dev. The threshold was frozen on dev (0.139). Model hash and threshold are in `models/final_model.json`. Set-c was scored once.
- Final answer: test AUROC 0.879 vs SAPS-I 0.658, well calibrated, no subgroup below the flag line. Test is in line with dev CV (0.867), so no overfitting.
- After the first run I fixed two reporting issues only (no change to model, threshold or predictions): the single-feature SHAP direction check was meaningless for a linear model, so it was replaced by a variable-level check, and the SHAP background was raised from 100 to 1,000 stays.
- Findings to explain: ventilation looks protective because 92% of cardiac-surgery patients are ventilated routinely; higher creatinine lowers risk once BUN is in the model (BUN/creatinine ratio).
- Phase 6: built `src/predict.py`, the Streamlit demo `app/streamlit_app.py` (Predict, Examples, Results, About; not-for-clinical-use banner), and 3 synthetic example records. Example risks: high-risk medical sepsis 96.9% (flagged), intermediate surgical 11.2%, low-risk cardiac surgery 0.7%. Bad files give clear messages, and records with no measurements are refused.
- Tested the app headless in a browser (no server errors), then stopped it.
- Rewrote README.md for recruiters (results, figures, approach, findings, limitations, reproduce, citation). Dataset licence checked on PhysioNet: ODC Attribution License v1.0.
- Re-ran notebooks 01-05 from a clean cache: model, features, splits and test metrics are byte-identical.
- Added MIT licence (code) and `app/requirements.txt` (7 runtime packages). Fresh-clone test without data or LightGBM passed.
- Deployed to Streamlit Community Cloud (Python 3.12): https://icu-mortality-mostafa.streamlit.app
- Next (optional): external validation on MIMIC-IV (needs CITI training and credentialing), then project 2.
