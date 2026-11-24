# ICU In-Hospital Mortality Prediction

Can we predict in-hospital mortality from the first 48 hours of an ICU stay, and does a machine learning model beat the standard severity score SAPS-I?

**Status:** Phase 1 (setup and framing). Results will be added as the project progresses.

## Data

[PhysioNet/CinC Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/), open access. About 12,000 ICU stays split into set-a, set-b and set-c. Each stay has up to 48 hours of vitals and labs, plus an outcomes file with the in-hospital mortality label and the SAPS-I and SOFA scores.

The data is not stored in this repository. To reproduce, download these files into `data/raw/`:

```
set-a.zip  set-b.zip  set-c.zip
Outcomes-a.txt  Outcomes-b.txt  Outcomes-c.txt
```

## Approach

| Phase | What |
|---|---|
| 1. Setup and framing | Problem definition, metrics, SAPS-I baseline, split plan |
| 2. Wrangling | Long time series to one row per stay (min/max/mean/last, counts, missingness flags) |
| 3. EDA | Missingness patterns, class imbalance, mortality by age and ICU type |
| 4. Modeling | Logistic regression, random forest, gradient boosting vs SAPS-I |
| 5. Evaluation | Calibration, thresholds, subgroups, SHAP, leakage checks |
| 6. Packaging | Streamlit demo, final write-up |

**Metrics:** AUROC, AUPRC (classes are imbalanced), and calibration.
**Split:** train and tune on set-a and set-b, hold out set-c as an untouched test set.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Then open `notebooks/` in VS Code and select the `.venv` kernel.

## Repository layout

```
data/raw/          raw PhysioNet files (not in git)
data/processed/    derived tables (not in git)
notebooks/         numbered analysis notebooks
src/               reusable functions
app/               Streamlit demo
reports/figures/   saved plots
PROJECT_NOTES.md   running log of decisions and key numbers
```

## Results

_To be filled in._

## Limitations

_To be filled in._
