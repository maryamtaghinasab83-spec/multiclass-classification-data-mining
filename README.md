# Multi-Class Classification on Imbalanced Data

An end-to-end data mining pipeline built for a Data Mining course: exploratory data analysis, preprocessing, feature selection, and classifier benchmarking on a large, imbalanced tabular dataset.

## Overview

- **Dataset**: ~100,000 samples, 34 features, imbalanced target classes (course-provided data, not included in this repository)
- **Goal**: benchmark classical ML classifiers and improve performance through data cleaning and feature selection, evaluated with Macro F1-score (chosen specifically to account for class imbalance)

## Pipeline

1. **Exploratory Data Analysis (EDA)**: distribution checks, missing values, target distribution, duplicate detection (138 duplicate rows removed)
2. **Preprocessing pipeline**: imputation, scaling, and categorical encoding
3. **Baseline modeling**: Logistic Regression, Decision Tree, Random Forest
4. **Feature selection**: ranked features by Random Forest importance and reduced the feature set to the top 25
5. **Final model**: Random Forest trained on the reduced feature set, with preprocessing and feature selection saved together in a single scikit-learn Pipeline

## Results

| Model | Features | Macro F1 | Accuracy |
|---|---|---|---|
| Baseline models | 34 | see `model_comparison_baseline.csv` | see file |
| **Random Forest (final)** | **25** | **≈ 0.49** | **≈ 0.68** |

Full comparison tables are in `model_comparison_baseline.csv` and `final_model_comparison.csv`.

## Repository Contents

- `data_mining_group_28_final.ipynb`: the full analysis, with all cell outputs saved (viewable directly on GitHub without running anything)
- `final_report_group_28.pdf`: the final written report
- `requirements.txt`: Python dependencies
- `*.csv`: summary tables (missing values, unique values, target distribution/correlation, model comparisons, feature importance, selected top-25 features)

The raw training data and the trained model file are not included here (course data and large binary file). To re-run the notebook, place the course dataset next to it as `group_28_train.csv`.

## Tech Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Jupyter Notebook

## How to Run

```bash
python -m pip install -r requirements.txt
jupyter notebook data_mining_group_28_final.ipynb
```

## Team and My Role

Group project (2 members). I led the project's methodology end to end: problem framing, EDA, missing-value and outlier handling, preprocessing pipeline design, evaluation criteria selection, model implementation, feature selection, results analysis, and the final report.

## Author

Maryam Taghinasab (with a project partner), Data Mining course, University of Tehran
