# Machine Learning Assignment 2 – Classification Models + Streamlit App

## a) Problem statement
Build and compare multiple classification models on a single dataset. The task involves training six different supervised learning algorithms and evaluating their performance using standard classification metrics. A Streamlit application is developed to allow interactive dataset selection, model comparison, and visualization of detailed performance results.
The goal is to analyze and compare model performance in terms of Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

## b) Dataset description
Dataset: UCI Breast Cancer Wisconsin (Diagnostic)  
- Source: UCI Machine Learning Repository
- Kaggle Link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- Type: Binary classification (Benign vs Malignant)  
- Instances: 569  
- Features: 30 numeric features  
- Target column: diagnosis (or equivalent)

## c) Models used + evaluation metrics
Models:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

The following evaluation metrics were computed for each model:
- Accuracy
- Area Under ROC Curve (AUC)
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient (MCC)

### Metrics comparison table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression      | 0.96     | 0.99 | 0.98      | 0.93   | 0.95 | 0.92 |
| Decision Tree            | 0.93     | 0.92 | 0.90      | 0.90   | 0.90 | 0.85 |
| kNN                      | 0.96     | 0.98 | 0.97      | 0.90   | 0.94 | 0.91 |
| Naive Bayes (Gaussian)   | 0.92     | 0.99 | 0.92      | 0.86   | 0.89 | 0.83 |
| Random Forest (Ensemble) | 0.97     | 0.99 | 1.00      | 0.93   | 0.96 | 0.94 |
| XGBoost (Ensemble)       | 0.96     | 0.99 | 1.00      | 0.90   | 0.95 | 0.93 |


### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression      | Provides a strong baseline with high accuracy and AUC. Performs well due to near-linear separability of features after scaling.                              |
| Decision Tree            | Lower performance compared to ensemble methods. May overfit depending on tree depth and is sensitive to small variations in data.                            |
| kNN                      | Performs comparably to Logistic Regression when features are properly scaled. Sensitive to distance metric and value of k.                                   |
| Naive Bayes              | Fast and computationally efficient. Slightly lower recall due to independence assumption among features.                                                     |
| Random Forest (Ensemble) | Achieved the best overall performance. Robust to noise and captures non-linear relationships effectively. Shows strong balance between precision and recall. |
| XGBoost (Ensemble)       | Comparable to Random Forest with very high AUC. Handles complex decision boundaries efficiently and performs well with tuning.                               |

### Overall Conclusion
Among the evaluated models, Random Forest and XGBoost demonstrated the strongest overall performance across all evaluation metrics. Logistic Regression and kNN also performed competitively, indicating that the dataset is well-structured and exhibits strong separability. Decision Tree and Naive Bayes showed comparatively lower performance due to model-specific assumptions and variance characteristics.

In the context of medical diagnosis, minimizing false negatives is critical. Ensemble methods such as Random Forest and XGBoost provide a strong balance between sensitivity and specificity, making them suitable for high-stakes classification problems.

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
