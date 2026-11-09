# Bank Customer Churn Prediction

**Binary Classification | Python | scikit-learn | TripleTen MBA — Sprint 10**

## Project Overview

Customer churn is a major challenge for banks: retaining existing customers can be more cost-effective than acquiring new ones. This project builds and compares machine learning classifiers to identify customers who are likely to leave a bank, supporting more targeted retention efforts.

**Project objective:** achieve an **F1-score greater than 0.59** on a held-out test set while examining precision, recall, and ROC-AUC.

## Dataset and Business Problem

The dataset contains **10,000 customer records** and **14 original columns**, including credit score, geography, gender, age, tenure, balance, number of products, credit-card ownership, activity status, and estimated salary.

The target, `exited`, represents whether a customer left the bank:

| Target | Meaning | Records | Share |
|---|---|---:|---:|
| `0` | Customer stayed | 7,963 | 79.63% |
| `1` | Customer left | 2,037 | 20.37% |

The `tenure` column initially contains **909 missing values**. Because the target classes are imbalanced, accuracy alone is insufficient to evaluate churn detection.

**Dataset:** [`datasets/Churn.csv`](datasets/Churn.csv)

## Approach

1. **Exploratory analysis:** inspected the dataset, missing values, duplicates, features, and target distribution.
2. **Feature preparation:** standardized column names and excluded identifiers and surnames from model features.
3. **Stratified splitting:** created training (60%), validation (20%), and test (20%) subsets.
4. **Preprocessing:** imputed missing `tenure` values using the training-set median; one-hot encoded categorical features; scaled numerical features for Logistic Regression, using training-fitted transformations for validation and test data.
5. **Baseline models:** compared Logistic Regression and Random Forest.
6. **Class imbalance:** compared unmodified training data, manual oversampling, manual undersampling, and class weighting. The sampling procedures were implemented with pandas and scikit-learn's `shuffle` utility, not an imbalanced-learn sampler.
7. **Hyperparameter search:** evaluated candidate Logistic Regression and Random Forest configurations against the validation set, selecting by F1-score.
8. **Final evaluation:** evaluated the selected Random Forest on the held-out test set.

## Baseline Validation Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.806 | 0.564 | 0.194 | 0.289 | 0.758 |
| Random Forest | 0.860 | 0.739 | 0.479 | 0.581 | 0.845 |

Random Forest performed better than Logistic Regression at baseline, but neither baseline model achieved the required F1-score.

## Hyperparameter Selection

The strongest validation configuration was **Random Forest with balanced class weights**:

```python
RandomForestClassifier(
    random_state=67,
    class_weight="balanced",
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=1,
)
```

| Validation metric | Result |
|---|---:|
| Precision | 0.641 |
| Recall | 0.627 |
| **F1-score** | **0.634** |
| ROC-AUC | 0.847 |

The best recorded Logistic Regression validation F1-score was **0.497**, using balanced class weights, `C=0.1`, and the `liblinear` solver.

## Final Test Results

| Metric | Test result |
|---|---:|
| Accuracy | 0.849 |
| Precision | 0.635 |
| Recall | 0.613 |
| **F1-score** | **0.623** |
| **ROC-AUC** | **0.869** |

### Confusion Matrix

| Actual / Predicted | Stayed (`0`) | Left (`1`) |
|---|---:|---:|
| Stayed (`0`) | 1,448 | 144 |
| Left (`1`) | 158 | 250 |

Of the **408 customers who left** in the test set, the model correctly identified **250** and missed **158**. It also flagged **144 customers who stayed**. The test F1-score of **0.623** exceeds the project's **0.59** threshold.

These are the results saved in the project notebook; they are not a claim of performance on future production data.

## Business Interpretation and Next Steps

The classifier could help prioritize retention outreach for customers with elevated churn risk. The model still produces both false positives (unnecessary outreach) and false negatives (missed at-risk customers). Before deployment, a bank should assess the costs of each error type, consider adjusting the classification threshold, test on recent data, evaluate customer-segment performance, and monitor performance over time.

## Repository Structure

```text
bank-customer-churn-prediction/
├── datasets/
│   └── Churn.csv
├── notebook.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started — Anaconda and VS Code

**1. Clone the repository**

```bash
git clone https://github.com/pedrolocosta/bank-customer-churn-prediction.git
cd bank-customer-churn-prediction
```

**2. Create and activate a Conda environment**

```bash
conda create -n bank-churn python=3.12 -y
conda activate bank-churn
python -m pip install -r requirements.txt
```

**3. Run the notebook**

Open the repository folder in VS Code, open `notebook.ipynb`, select the `bank-churn` Python kernel, and run the cells from top to bottom. The dataset is expected at `datasets/Churn.csv`.

To inspect your installed package versions:

```bash
python -c "import sys, sklearn, pandas, numpy; print(sys.version); print('scikit-learn:', sklearn.__version__); print('pandas:', pandas.__version__); print('numpy:', numpy.__version__)"
```

> **Reproducibility note:** The notebook reports Python 3.12.10. The exact versions of all installed packages have not been independently verified. `requirements.txt` specifies dependencies and compatibility ranges rather than an exact, tested environment lockfile. Model outputs can differ across package versions.

## Technologies and Skills

- **Python, pandas, NumPy:** data inspection, cleaning, feature preparation, and manual resampling
- **scikit-learn:** Logistic Regression, Random Forest, stratified splitting, scaling, and classification metrics
- **Matplotlib / Seaborn:** exploratory visualization
- **Jupyter Notebook, Anaconda, VS Code:** local experimentation and analysis

## Author

**Pedro Oliveira Costa** — Data Science & Machine Learning Portfolio

- [GitHub](https://github.com/pedrolocosta)
- [LinkedIn](https://www.linkedin.com/in/pedrolocosta/)
- [OCAI Engineering](https://ocaiengineering.com/)

*Educational project developed as part of the TripleTen MBA in Artificial Intelligence & Data Science.*
