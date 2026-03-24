# 🌪️ FEMA Disaster Intelligence System
### A Two-Phase Machine Learning Approach to Disaster Response & Budget Planning

> **MSc Business Analytics · Artificial Intelligence II · ESADE Business School**

---

## 📌 Overview

When the U.S. federal government declares a natural disaster, FEMA faces two urgent questions simultaneously:

1. **Who needs help first?** — Across all affected counties, which ones require immediate escalation and resource deployment?
2. **How much will this cost?** — Before a single project application is filed, which funding tier will this disaster ultimately reach?

This project builds a two-phase ML system to support both decisions using only information available **at the moment of federal declaration** — no post-event data, no leakage.

| Phase | Problem Type | Algorithm | Unit of Analysis |
|-------|-------------|-----------|-----------------|
| **Phase 1 — County Triage** | Unsupervised | K-Means Clustering | County × Disaster |
| **Phase 2 — Funding Prediction** | Supervised | Multi-class Classification | Disaster |

---

## 🗂️ Repository Structure

```
├── phase1_clustering/
│   ├── 01_2_data_loading.ipynb
│   ├── 02__2_clustering.ipynb
│   └── 03__2_evaluation.ipynb
│
├── phase2_supervised/
│   ├── 01_data_loading_merging (2).ipynb
│   ├── 02_cleaning_feature_engineering (2).ipynb
│   ├── 03_eda (2).ipynb
│   ├── 04_modeling_disaster_level (2).ipynb
│   └── 05_evaluation_interpretation.ipynb
│
└── utils.py
```

---

## 📦 Data Sources

| Dataset | Source | Key Variables |
|---------|--------|---------------|
| FEMA PA Funded Projects | FEMA OpenData | 810,656 project rows, federal obligations, county FIPS |
| FEMA Disaster Declarations | FEMA OpenData | Incident dates, declaration type |
| CDC Social Vulnerability Index (2018) | CDC ATSDR | SVI score (0–1), population, land area |
| ACS Census — Median Income (S1903) | U.S. Census Bureau | Median household income by county |
| ACS Census — Poverty Rate (S1701) | U.S. Census Bureau | % population below poverty line by county |
| FEMA National Risk Index (NRI) | FEMA | Population, risk score, risk rating by county |

All datasets are publicly available. County-level sources are linked via **5-digit FIPS codes**.

---

## 🔵 Phase 1 — County Triage via K-Means Clustering

> **"Which counties within a declared disaster need immediate escalation?"**

Because FEMA does not publish any official county priority ranking, there is **no target variable to train on**. Supervised learning is not applicable. K-Means is used to group counties by structural similarity, and clusters are mapped to operational tiers through data-driven centroid ranking.

### Notebooks

---

#### `01_2_data_loading.ipynb` — Data Loading & Matrix Construction

Builds the base dataset for clustering by combining two sources:

- **CDC SVI** — downloads the 2018 county-level file (~3,142 counties). Cleans the sentinel value `-999` → `NaN`, standardises FIPS to 5 digits, and retains `svi_score`, `population`, `area_sqmi`, `pop_density`.
- **FEMA PA Projects** — re-uses the raw file from Phase 2. Reduces 810,656 project rows to **28,667 unique county × disaster combinations**.
- **`prior_disasters_5yr`** — for each county-disaster row, counts distinct disaster declarations affecting the same county in the preceding 5 years. This is computed via a sorted time-window loop, not a static join, to respect temporal ordering.
- Merges SVI features into the county-disaster matrix on FIPS (95.3% match rate). Drops statewide FIPS aggregates (ending in `000`).

**Output:** `county_disaster_matrix.csv` — 28,667 rows × 14 columns.

---

#### `02__2_clustering.ipynb` — K-Means Clustering & Priority Mapping

Core modelling notebook for Phase 1.

- **Feature selection:** `svi_score`, `pop_density` (capped at 99th percentile to reduce outlier distortion), `prior_disasters_5yr`. All three are observable before the disaster occurs.
- **Scaling:** `StandardScaler` applied so that population density (0–72,000 ppl/sqmi) does not dominate the Euclidean distance metric.
- **k validation:** Elbow method (inertia) and silhouette score tested for k = 2–8. k = 3 is selected based on joint criteria: operational fit (FEMA works with 3 response tiers) and interpretable centroid separation, even though k = 4 achieves a marginally higher silhouette (0.411 vs. 0.379).
- **Data-driven priority mapping:** Cluster IDs from K-Means are arbitrary. A composite vulnerability score (SVI weight 0.5, density weight 0.3, prior disasters weight 0.2) ranks centroids from highest to lowest → Priority 1 / 2 / 3.

**Cluster results:**

| Priority | Label | Count | Mean SVI | Mean Density | Mean Prior Disasters |
|----------|-------|-------|----------|-------------|----------------------|
| 1 | Immediate Escalation | 1,618 (5.8%) | 0.541 | 4,603 ppl/sqmi | 3.76 |
| 2 | Targeted Support | 13,518 (48.4%) | 0.755 | 109 ppl/sqmi | 2.03 |
| 3 | Standard Review | 12,814 (45.8%) | 0.256 | 147 ppl/sqmi | 2.15 |

**Output:** `county_triage_scored.csv` — 28,667 rows with `cluster`, `priority`, `priority_label` appended.

---

#### `03__2_evaluation.ipynb` — Business Validation

Checks whether the clustering results are operationally coherent. PA obligations are **not used in clustering** — they serve here as an out-of-sample validation signal only.

- **Feature profile by priority:** Confirms expected ordering across SVI, density, and prior disaster count. Flags that Priority 2 carries the highest mean SVI (0.755), driven by the composite weighting scheme favouring density for Priority 1.
- **PA obligation by priority:** Priority 1 counties show mean obligations of $0.89M vs. $0.13M (P2) and $0.15M (P3) — directionally aligned. The evaluation code flags a non-monotonic ordering and surfaces it as a limitation.
- **State-level distribution:** Bar chart of Priority 1 county-disaster appearances by state — identifies chronically high-exposure states for operational planning.

---

## 🟠 Phase 2 — Disaster Funding Tier Prediction

> **"At the moment of declaration, which DRF funding tier will this disaster reach?"**

The total federal PA cost of a disaster is not known for 12–18 months. This model provides an early structured estimate to anchor budget reservations before project applications are filed.

**Funding tiers (2019 CPI-adjusted dollars):**

| Tier | Range | Label | Scope |
|------|-------|-------|-------|
| 0 | < $1M | Minor | DRF ✓ |
| 1 | $1M – $50M | Moderate | DRF ✓ |
| 2 | $50M – $500M | Major | DRF ✓ |
| 3 | > $500M | Catastrophic | Congressional Supplemental — **excluded** |

Tier 3 is excluded because Catastrophic events trigger Emergency Supplemental Appropriations — a legislative process that is not predictable from declaration-time features.

### Notebooks

---

#### `01_data_loading_merging (2).ipynb` — Data Loading & Disaster-Level Aggregation

- Loads FEMA PA Projects (810,656 rows × 25 cols) and Disaster Declarations (69,766 rows).
- Filters declarations to `paProgramDeclared == 1`, deduplicates to one row per `disasterNumber`, retains `incidentBeginDate` and `incidentEndDate`.
- Builds 5-digit FIPS from `stateNumberCode` + `countyCode`.
- Loads and cleans three county-level enrichment files: ACS median income, ACS poverty rate, FEMA NRI (population, risk score, risk rating).
- **Disaster-level aggregation:** Groups 810,656 project rows by `disasterNumber` to produce **1,766 disaster-level records** with total federal obligations, project count, number of affected counties, and population-weighted socioeconomic averages.

**Output:** `merged_disaster_level.csv` — 1,766 rows × 14 columns.

---

#### `02_cleaning_feature_engineering (2).ipynb` — Cleaning & Feature Engineering

- **Date parsing:** Converts all date strings to `datetime` objects for time feature derivation.
- **Null handling:** Median imputation for county-level numeric features (~3% of counties without Census/NRI match).
- **Canonical incident type mapping:** Collapses 26 raw FEMA spellings into 8 stable categories (Flood, Hurricane, Severe Storm, Winter Storm, Tornado, Wildfire, Earthquake, Other). Without this, one-hot encoding treats `"Severe Storm(s)"` and `"Severe Storms"` as distinct features.
- **Time features:** `declaration_lag_days`, `incident_duration_days`, `incident_season`, `incident_year`.
- **`prior_disasters_5yr`:** Rolling 5-year count of PA disasters in the same state — captures chronically disaster-prone states.
- **CPI-U adjustment:** All obligations inflated to 2019 dollars using BLS annual CPI-U averages (1998–2025). Ensures a $40M disaster in 1998 is classified as Moderate, not Minor, relative to today's cost levels.
- **Target creation:** `funding_tier` (0–3) via `pd.cut` on CPI-adjusted obligations. Tier 3 rows kept in the file but excluded at modelling time.

**Output:** `cleaned_disaster_level.csv` — 1,766 rows × 21 columns, no nulls.

---

#### `03_eda (2).ipynb` — Exploratory Data Analysis

Validates that the selected features actually discriminate funding tiers before modelling begins.

- **Class balance:** Shows Tier 1 (Moderate) dominates at 73.4% — motivates use of `class_weight="balanced"` and weighted F1 as the selection metric.
- **Mean federal share by incident type:** Confirms hurricanes far outpace other types; establishes `incidentType` as a high-information feature.
- **Tier distribution by incident type:** Stacked bar showing which disaster types most often reach Major or higher.
- **Temporal funding profile:** Stacked area chart by year showing that obligations are clustered around major hurricane seasons — directly motivates the **temporal train/test split** rather than a random one.

---

#### `04_modeling_disaster_level (2).ipynb` — Model Training & Selection

Core modelling notebook for Phase 2.

- **Feature set (11 features, all declaration-time):** `incidentType`, `stateAbbreviation`, `incident_season` (categorical); `declaration_lag_days`, `incident_duration_days`, `n_counties`, `prior_disasters_5yr`, `population`, `median_income`, `poverty_rate`, `risk_score` (numeric).
- **Temporal split:** Train on pre-2016 (1,085 events), validate on 2016–2017 (92), test on 2018+ (518). A random split would leak future climate and policy trends into training.
- **Preprocessing pipeline:** Categorical → constant imputation (`'Unknown'`) + one-hot encoding; Numeric → median imputation + StandardScaler. Wrapped in `sklearn.Pipeline` — transformers are fit only on training data.
- **Models trained:** Stratified Baseline, Logistic Regression, Random Forest, Gradient Boosting (default + `RandomizedSearchCV` tuned over 30 candidates with `TimeSeriesSplit(n_splits=5)`).
- **Bootstrap stability:** 50 resamples on training set, fixed test set → LR: mean F1 = 0.679, std = 0.014, 95% CI [0.654, 0.706].

**Results summary:**

| Model | Val F1 | Test F1 | Test Acc. |
|-------|--------|---------|-----------|
| Baseline (Stratified) | 0.677 | 0.519 | 0.570 |
| **Logistic Regression ✓** | **0.786** | **0.690** | 0.685 |
| Random Forest | 0.741 | 0.559 | 0.670 |
| Gradient Boosting | 0.763 | 0.675 | 0.712 |
| GradBoosting (Tuned) | 0.772 | 0.683 | 0.726 |

Logistic Regression is selected on validation F1. Its performance advantage over ensemble methods on this dataset reflects the small training set (1,085 rows) and the concentration of signal in a few high-information categorical features.

**Output:** `best_disaster_model.pkl` — serialised best pipeline for evaluation in notebook 05.

---

#### `05_evaluation_interpretation.ipynb` — Evaluation & Interpretation

Final evaluation on the held-out test set (2018+). Test metrics reported here for the first and only time.

- **Threshold calibration:** Default argmax achieves Major (Tier 2) recall of 0.480. Lowering the Tier 2 decision boundary to 0.375 (selected on validation set) raises Major recall to 0.510. This is the operationally critical metric — a missed Major disaster represents a ~$250M DRF under-reservation.
- **Per-class report (calibrated):** Minor 0.44/0.64, Moderate 0.81/0.74, Major 0.54/0.51 (precision/recall).
- **Val–test gap analysis:** LR gap = 0.096 — flagged as "Investigate". Attributable to distributional shift: the 2016–2017 validation window was quiet; the 2018–2025 test window includes record hurricane seasons and COVID-19 declarations.
- **Cost of misclassification:** Using tier midpoints ($0.5M, $25.5M, $275M), the total DRF under-reservation exposure across the test set is estimated at **$14.05 billion**, driven by 45 Major disasters misclassified as Moderate.
- **SHAP:** Planned but not executable due to scikit-learn version mismatch between the saved model artefact and the evaluation environment. Re-running notebook 04 with the current sklearn version resolves this.

---

## 🛠️ `utils.py`

Shared utility functions used across multiple notebooks:

| Function | Description |
|----------|-------------|
| `data_summary()` | Prints shape, null counts, and basic stats for any DataFrame |
| `get_season()` | Maps a month integer to Winter / Spring / Summer / Fall |
| `add_prior_disasters()` | Rolling 5-year state-level disaster count with temporal ordering |
| `classification_metrics()` | Wrapper returning weighted F1, accuracy, and full classification report |
| `DISASTER_BINS` | Tier cut boundaries in 2019 CPI-adjusted dollars |
| `DISASTER_LABELS` | Tier label dictionary (0–3) |

---

## ⚙️ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn requests
```

> **Note on reproducibility:** The saved model pickle (`best_disaster_model.pkl`) was generated with scikit-learn 1.6.1. Running notebook 05 with a newer version will produce an `InconsistentVersionWarning`. Re-run notebook 04 to regenerate the pickle in your environment before running notebook 05.

---

## ⚠️ Key Limitations

- **Phase 1 silhouette at k=3 (0.379) is below k=4 (0.411)** — k is chosen partly on operational grounds.
- **Static SVI (2018)** — does not capture county-level changes over the 1998–2025 study window.
- **Val–test gap of 0.096** in Phase 2 — distributional shift between the 2016–2017 validation period and the 2018–2025 test period.
- **Obligations ≠ severity** — PA spending is shaped by administrative eligibility, not only physical damage. The model predicts funding tier, not disaster magnitude.
- **Small training set** — 1,085 disaster events after temporal split. Limits the complexity of models that can be trained without overfitting.

