# Credit Card Fraud Detection

This machine-learning project detects fraudulent credit card transactions using Logistic Regression and XGBoost classification models.

## Dataset

The project uses the well-known **Credit Card Fraud Detection** dataset, originally released through a collaboration between Worldline and the Machine Learning Group at Université Libre de Bruxelles (ULB). It contains transactions made by European cardholders over two days in September 2013.

Dataset source: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud/data)

- **284,807 transactions** and **31 columns**
- **492 fraudulent transactions** (only about **0.17%** of the data)
- `Class` is the target: `0` represents a legitimate transaction and `1` represents fraud
- `Time` records the seconds elapsed since the first transaction
- `Amount` contains the transaction value
- `V1` to `V28` are anonymized numerical features produced using Principal Component Analysis (PCA)

Because the dataset is extremely imbalanced, the notebook creates a balanced sample before splitting the data into training and testing sets. It then trains and evaluates Logistic Regression and XGBoost models using metrics such as accuracy, precision, recall, F1-score, and cross-validation accuracy.

> **Note:** Accuracy alone can be misleading for highly imbalanced fraud data, so precision, recall, F1-score, and the precision-recall curve are especially important when comparing models.
