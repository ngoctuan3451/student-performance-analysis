# Student Performance — Data Wrangling, EDA & Classification

**A full data-analysis cycle in pandas and scikit-learn: clean a messy student dataset, explore what drives grades, and train models to predict a student's grade class.**

---

## Problem

Given ~2,100 student academic records, can we (a) clean the real-world data-entry errors, (b) understand which factors relate to academic performance, and (c) train a model that predicts a student's `GradeClass` (A–F, encoded 0–4) from their study habits and attendance?

## Data

A student-records dataset (`Student_List_A2.csv`) with: `StudentID`, `Age`, `StudyTimeWeekly`, `Absences`, `ParentalSupport`, `GPA`, and the target `GradeClass`. A separate hold-out file (`Student_List_A2_Submission.csv`) is used to generate final predictions. Part B uses a [Kaggle student-performance dataset](https://www.kaggle.com/datasets/ilayaraja07/data-cleaning-feature-imputation) for an unsupervised clustering exercise.

## What I did

**1. Data wrangling**
- Identified 21 missing `StudyTimeWeekly` values and imputed them with the column median.
- Caught an impossible outlier (`Absences = -122`) — a data-entry sign error — and removed it after validating the plausible range (0–365).
- Reconciled a labelling inconsistency: recomputed grade from the GPA cut-offs and found **100 mislabelled `GradeClass` values**, then corrected them (trusting GPA as the continuous source of truth over the derived label).

**2. Exploratory data analysis**
- Class distribution: heavily imbalanced (GradeClass 4 ≈ 54% of students).
- Correlations: `StudyTimeWeekly` ↔ `GPA` = **+0.18** (weak positive); `Absences` ↔ `GPA` = **−0.72** (strong negative).
- Grouped summary stats (mean / median / std / IQR) of GPA and Absences per grade class.
- Noted that correlation ≠ causation, and listed confounders and better measurements that would strengthen any causal claim.

**3. Supervised learning**
- Deliberate feature selection (dropped `StudentID` as an identifier and `GPA` as it defines the label — avoiding leakage).
- 70/30 **stratified** train/test split; standardised features with `StandardScaler` (fit on train only).
- Trained and compared an **SVM** and a **Decision Tree**, with confusion matrices and a bias/variance discussion.
- Generated predictions on the hold-out submission set.

**4. Unsupervised learning (Part B)**
- K-means clustering on exam scores, choosing **k = 3** from cluster separation.

## Key findings

- **Attendance is the strongest signal.** Absences correlate −0.72 with GPA — far stronger than study time (+0.18).
- **SVM edged out the Decision Tree on overall accuracy (75% vs 71%)**, but the tree handled the rare minority class (GradeClass 0) better — the SVM ignored it entirely. A reminder that accuracy alone hides minority-class failure on imbalanced data.
- The most common confusion in both models was adjacent classes (C ↔ D), consistent with overlapping student profiles whose real difference lives in the excluded GPA signal.

## How to run

```bash
pip install pandas scikit-learn matplotlib
jupyter notebook Assign1.ipynb
```

Run the cells top to bottom; the CSVs sit alongside the notebook.

## Repository structure

```
.
├── Assign1.ipynb                       # the full analysis
├── Student_List_A2.csv                 # main dataset
├── Student_List_A2_Submission.csv      # hold-out set for predictions
└── Students_Performance_knn.csv        # Part B clustering dataset
```

## Built with

Python · pandas · scikit-learn · matplotlib

## Author

**Tuan Ngoc Chu** — FIT1043 Introduction to Data Science, Monash University.
