# FEMA Disaster Intelligence System

### A Two-Phase Machine Learning Approach to Disaster Response and Budget Planning

MSc Business Analytics · Artificial Intelligence II · ESADE Business School

---

## Overview

When a disaster is declared, FEMA has to make decisions very quickly, often with limited information.

There are two main questions at that point:

1. Which counties need immediate attention and resources
2. How large the disaster will be in terms of funding

The issue is that the real cost of a disaster is only known much later, once projects are submitted and processed.

This project builds a two-phase machine learning system to support both decisions using only data available at the time of declaration. No post-event information is used.

Phase 1 focuses on prioritizing counties.
Phase 2 focuses on estimating the overall funding tier of the disaster.

| Phase   | Approach     | Model                 | Unit                |
| ------- | ------------ | --------------------- | ------------------- |
| Phase 1 | Unsupervised | KMeans                | County and disaster |
| Phase 2 | Supervised   | Classification models | Disaster            |

---

## Repository structure

```
phase1_clustering/
    01_data_loading.ipynb
    02_clustering.ipynb
    03_evaluation.ipynb

phase2_supervised/
    01_data_loading.ipynb
    02_cleaning_feature_engineering.ipynb
    03_eda.ipynb
    04_modeling.ipynb
    05_evaluation.ipynb

utils.py
```

---

## Data sources

All datasets are public and combined using county FIPS codes.

* FEMA Public Assistance projects
* FEMA disaster declarations
* CDC Social Vulnerability Index
* Census data (income and poverty)
* FEMA National Risk Index

These sources provide both disaster-level information and county-level context.

---

## Phase 1 — County triage

Goal: decide which counties should be prioritized right after a disaster is declared.

There is no target variable for this problem, so a supervised approach is not possible. Instead, KMeans is used to group counties based on similarity.

The clustering is based on three variables:

* SVI score
* Population density
* Prior disasters in the last five years

Clusters are then ranked and mapped to three priority levels.

### Output

Each county-disaster combination is assigned a priority:

* Priority 1 — immediate escalation
* Priority 2 — targeted support
* Priority 3 — standard review

The separation is mainly driven by population density and exposure.

### Validation

There is no ground truth, so validation is done using FEMA obligations.

Priority 1 counties tend to show higher spending, which supports the clustering results, even if not perfectly.

---

## Phase 2 — Funding tier prediction

Goal: estimate the total cost tier of a disaster at the moment of declaration.

The model predicts three tiers:

* Minor
* Moderate
* Major

Catastrophic events are excluded because they follow a different funding process.

### Data preparation

* Project-level data is aggregated to disaster level
* Features are built using only declaration-time information
* Values are adjusted for inflation to ensure consistency

### Models

Several models are tested:

* Logistic regression
* Random forest
* Gradient boosting

Logistic regression performs best and is selected.

This is likely due to the relatively small dataset and the structure of the features.

---

## Evaluation

Evaluation is done on post-2018 data.

Main observations:

* The model improves over the baseline
* Moderate events are predicted reliably
* Major events are harder to detect

A key issue is underestimation of large disasters, which has direct budget implications.

When translating errors into cost, the model shows around 14 billion in under-reservation exposure across the test set, mostly from Major events predicted as Moderate.

---

## Utilities

The `utils.py` file includes shared functions:

* Data summaries
* Feature engineering helpers
* Evaluation metrics
* Tier definitions

---

## Requirements

```
pip install pandas numpy scikit-learn matplotlib seaborn requests
```

If there are version issues with the saved model, re-run the modeling notebook to regenerate it.

---

## Limitations

* Clustering in Phase 1 is strongly influenced by population density
* SVI is static and does not change over time
* No physical intensity variables are included
* Data distribution changes over time affect performance
* The dataset for Phase 2 is relatively small

Also, funding is not a perfect proxy for disaster severity, since it depends on administrative processes.

---

## Final note

The two phases are designed to work together.

Phase 2 estimates the overall size of the disaster.
Phase 1 helps decide how to allocate resources within that disaster.

Together, they provide a structured way to support early decision making with limited information.

