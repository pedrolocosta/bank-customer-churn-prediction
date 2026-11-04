# Bank Customer Churn Prediction

**Machine Learning Classification | Python | scikit-learn | TripleTen MBA — Sprint 10**

## Project Overview

Customer churn is a significant challenge for financial institutions. Identifying customers who are likely to leave can help a bank prioritize retention initiatives and make better-informed decisions.

This project develops and evaluates binary classification models to predict whether a bank customer will leave. The primary objective is to achieve an **F1-score above 0.59**, while also examining precision, recall, and ROC-AUC to understand the models' strengths and limitations.

## Business Problem

The target variable, `exited`, indicates whether a customer left the bank:

- `0`: Customer stayed
- `1`: Customer left

Because customers who leave are the minority class, accuracy alone can be misleading. The project emphasizes **F1-score**, which balances precision and recall, and uses ROC-AUC as a complementary measure of ranking performance.

## Dataset

The dataset contains **10,000 customer records** and 14 original columns. It includes customer characteristics such as credit score, geography, gender, age, tenure, account balance, number of products, credit-card ownership, activity status, and estimated salary.

The target distribution is:

| Outcome | Customers | Share |
|---|---:|---:|
| Stayed (`0`) | 7,963 | 79.63% |
| Left (`1`) | 2,037 | 20.37% |

The original `tenure` column contains **909 missing values**. No fully duplicated rows or repeated customer IDs were found in the notebook's checks.

Dataset location: `datasets/Churn.csv`.

## Methodology

1. **Data inspection and cleaning:** Examined data types, missing values, duplicates, and class distribution; removed the row-number field and standardized column names.
2. **Feature selection:** Removed customer identifiers and surnames from the predictors; retained `exited` as the target.
3. **Stratified data splitting:** Allocated 60% of records to training, 20% to validation, and 20% to testing.
4. **Preprocessing:** Imputed missing tenure values using the training-set median, applied one-hot encoding to categorical variables, and standardized numerical variables for Logistic Regression. Training-derived preprocessing parameters were reused for validation and test data.
5. **Model comparison:** Evaluated Logistic Regression and Random Forest classifiers against the same validation metrics.
6. **Class-imbalance experiments:** Compared the original class distribution with oversampling, undersampling, and class weighting.
7. **Model evaluation:** Assessed F1-score, precision, recall, accuracy, ROC-AUC, and confusion matrices on held-out data.

## Baseline Validation Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.806 | 0.564 | 0.194 | 0.289 | 0.758 |
| Random Forest | 0.860 | 0.739 | 0.479 | 0.581 | 0.845 |

Random Forest outperformed Logistic Regression on the baseline validation F1-score and ROC-AUC. Both models initially fell below the required F1 threshold.

## Final Test Results

The notebook's final evaluation uses a Random Forest classifier with these settings:

```python
RandomForestClassifier(
    random_state=67,
    class_weight='balanced',
    n_estimators=100,
    max_depth=None,
    min_samples_leaf=3
)
```

| Metric | Test result |
|---|---:|
| Accuracy | 0.832 |
| Precision | 0.577 |
| Recall | 0.667 |
| **F1-score** | **0.619** |
| **ROC-AUC** | **0.869** |

### Confusion Matrix — Test Set

| Actual / Predicted | Stayed | Left |
|---|---:|---:|
| Stayed | 1,448 | 144 |
| Left | 158 | 250 |

The final model correctly identified **250 of 408 customers who left** (approximately **66.7% recall**). It also flagged 144 customers who ultimately stayed and missed 158 customers who left.

**The recorded test F1-score of 0.619 exceeds the project's minimum requirement of 0.59.** The ROC-AUC of 0.869 suggests useful discrimination between customers who stay and those who leave. These metrics are measured on the project's held-out test set and do not establish production performance.

## Technical Note: Hyperparameter Search

The notebook explores model hyperparameters and class-balancing strategies. However, the current tuning cell places model fitting and scoring outside the innermost parameter loop, so its recorded tuning results **do not constitute a complete evaluation of every proposed parameter combination**. The tuning loop should be corrected and rerun before claiming that the selected configuration is optimal. The final test results above are reported as recorded in the notebook, not as evidence of a fully validated hyperparameter search.

## Business Implications

A churn-risk classifier could help a bank identify customers for targeted retention outreach. Before operational deployment, further work should include:

- Correcting and rerunning hyperparameter tuning, ideally with a repeatable cross-validation process.
- Evaluating different classification thresholds based on the relative costs of missed churners and unnecessary outreach.
- Checking model performance across relevant customer segments.
- Validating on more recent, representative data and monitoring performance over time.

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

## Getting Started (Anaconda + VS Code)

Clone the repository:

```bash
git clone https://github.com/pedrolocosta/bank-customer-churn-prediction.git
cd bank-customer-churn-prediction
```

Create and activate a Conda environment:

```bash
conda create -n bank-churn python=3.11 -y
conda activate bank-churn
python -m pip install -r requirements.txt
```

Open the folder in VS Code, open `notebook.ipynb`, and select the Python kernel from the `bank-churn` Conda environment. Run the notebook cells in order. The notebook expects the dataset at `datasets/Churn.csv`.

## Tools and Skills

- **Python, pandas, NumPy:** Data inspection, cleaning, and transformation
- **scikit-learn:** Logistic Regression, Random Forest, preprocessing, splitting, and evaluation
- **imbalanced-learn:** Random oversampling and undersampling
- **Matplotlib:** Exploratory visualization
- **Jupyter Notebook, Anaconda, VS Code:** Reproducible local development workflow

## Author

**Pedro Oliveira Costa**  
Data Science & Machine Learning Portfolio  

- [GitHub](https://github.com/pedrolocosta)
- [LinkedIn](https://www.linkedin.com/in/pedrolocosta/)
- [OCAI Engineering](https://ocaiengineering.com/)

*Educational project developed as part of the TripleTen MBA in Artificial Intelligence & Data Science.*
