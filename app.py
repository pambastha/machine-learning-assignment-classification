import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from model.train_and_eval import train_eval_all

@st.cache_resource(show_spinner=False)
def train_models_cached(df, target_col, test_size):
    return train_eval_all(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=42
    )

# Page config
st.set_page_config(page_title="Machine Learning Assignment 2 - Classifiers", layout="wide")
st.title("Machine Learning Assignment 2 – Classification Models Demo")

st.markdown("""
### Steps to Use the Application
1. Choose a dataset in the *Dataset Selection* section (built-in dataset or CSV upload).
2. Select the target (label) column. If an ID or high-cardinality column is selected, the application will display a validation message.
3. Adjust the holdout split slider (e.g., 0.2 means 80% train and 20% test).
4. Review the metrics table for all models.
5. Select a model to view the confusion matrix, classification report, ROC curve (binary), and feature importance (tree-based).
""")

# Dataset Selection
st.subheader("Dataset Selection")

use_builtin = st.checkbox("Use Built-in Breast Cancer Dataset (sklearn)")

if use_builtin:
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["diagnosis"] = data.target

    # Convert numeric labels to readable form
    df["diagnosis"] = df["diagnosis"].map({0: "malignant", 1: "benign"})

    st.success(
        "Using sklearn built-in Breast Cancer dataset. "
        "To use a different dataset, uncheck this option and upload a CSV file."
    )

else:
    uploaded = st.file_uploader(
        "Or Upload a CSV file (recommended dataset size: 500–10,000 rows)",
        type=["csv"]
    )

    if uploaded is None:
        st.info("Upload a CSV or select built-in dataset above.")
        st.stop()

    df = pd.read_csv(uploaded)


df = df.dropna(axis=1, how="all").copy()  # drop fully-empty columns

st.subheader("Preview")
st.dataframe(df.head(10), use_container_width=True)

cols = df.columns.tolist()

# Auto-select 'diagnosis' column if present
default_index = 0
for i, col in enumerate(cols):
    if col.lower() == "diagnosis":
        default_index = i
        break

st.markdown("### Select Target Column")
target_col = st.selectbox(
    "Choose the column representing the classification label:",
    options=cols,
    index=default_index
)

# Clean target: drop NaNs and strip whitespace so " " doesn't become a separate class
df = df.dropna(subset=[target_col]).copy()
df[target_col] = df[target_col].astype(str).str.strip()
df = df[df[target_col] != ""].copy()

st.subheader("Target distribution check")
vc = df[target_col].value_counts(dropna=False)
st.write(vc.head(50))
st.write(f"Unique classes = {vc.shape[0]}")
st.write(f"Min class count = {int(vc.min())}")
st.write(f"Max class count = {int(vc.max())}")

# Guardrails to prevent selecting ID-like target
if vc.shape[0] > 20:
    st.error(
        f"Target '{target_col}' has {vc.shape[0]} unique classes. "
        "This looks like an ID/high-cardinality column. “Please pick the true label column (not an identifier column)."
    )
    st.stop()

# Filter out classes with <2 samples 
min_required = 2
valid_classes = vc[vc >= min_required].index
df = df[df[target_col].isin(valid_classes)].copy()

vc2 = df[target_col].value_counts(dropna=False)
st.caption("After filtering classes with <2 samples:")
st.write(vc2)

if vc2.shape[0] < 2:
    st.error(
        "After cleaning/filtering, target has < 2 classes. "
        "Please choose a different target column or dataset."
    )
    st.stop()


# Train & Evaluate
test_size = st.slider(
    "Test Size (holdout proportion)",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05,
    help="Proportion of dataset reserved for evaluation. Example: 0.2 means 80% train, 20% test."
)

with st.spinner("Training + evaluating all models..."):
    results_df, fitted, le = train_models_cached(
        df, target_col, test_size
    )

st.subheader("Metrics Comparison Table (All Models)")
st.dataframe(results_df, use_container_width=True)

model_names = results_df["ML Model Name"].tolist()

st.markdown("### Select Model for Detailed Analysis")
chosen = st.selectbox("Choose a trained model to view detailed report", options=model_names)

# Detailed split (stratified when valid)
X = df.drop(columns=[target_col])
y = df[target_col]

vc_detail = y.value_counts(dropna=False)
stratify_arg = y if int(vc_detail.min()) >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42,
    stratify=stratify_arg
)

# Encode targets exactly like training
y_test_enc = le.transform(y_test)

pipe = fitted[chosen]
y_pred_enc = pipe.predict(X_test)
y_pred = le.inverse_transform(y_pred_enc)

# Detailed Results UI
st.subheader(f"Detailed Results – {chosen}")

c1, c2 = st.columns([1.15, 0.85])

with c1:
    st.markdown("### Confusion Matrix")

    labels = le.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        ax=ax
    )

    ax.set_title(f"{chosen} – Confusion Matrix", fontsize=14)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Actual Label", fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)


# Classification Report
with c2:
    st.markdown("### Classification Report")
    st.text(classification_report(y_test, y_pred, zero_division=0))

# ROC Curve (Binary only)
st.markdown("---")
st.subheader("ROC Curve")

if len(le.classes_) == 2:
    positive_label = le.classes_[1]
    st.caption(f"Positive class for ROC/AUC is treated as: **{positive_label}** (LabelEncoder class index 1).")

    y_score = None
    if hasattr(pipe, "predict_proba"):        
        try:
            proba = pipe.predict_proba(X_test)
        except Exception:
            proba = None

        # Expect shape (n,2) for binary
        if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
            y_score = proba[:, 1]
    elif hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(X_test)
        y_score = scores  # roc_curve can take decision scores too

    if y_score is None:
        st.info("ROC Curve not available for this model (no predict_proba/decision_function).")
    else:
        fpr, tpr, _ = roc_curve(y_test_enc, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random baseline")

        ax.set_title(f"{chosen} – ROC Curve", fontsize=14, pad=12)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)
else:
    st.info("ROC Curve shown only for binary classification in this app.")

# Feature Importance (Tree/Ensemble)
st.markdown("---")
st.subheader("Model Explainability")

clf = pipe.named_steps["clf"]

if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_

    # Get feature names after preprocessing (ColumnTransformer + OneHot)
    try:
        feature_names = pipe.named_steps["prep"].get_feature_names_out()
    except Exception:
        feature_names = np.array([f"f{i}" for i in range(len(importances))])

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(15)

    st.write("Top features used by the model:")
    st.dataframe(feat_df, use_container_width=True)

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1])
    ax.set_title(f"{chosen} – Top Feature Importances", fontsize=14, pad=12)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    st.pyplot(fig)
else:
    st.info("Feature importance is available only for tree-based models (Decision Tree, Random Forest, XGBoost).")


# Download metrics
st.markdown("---")
st.download_button(
    "Download metrics table as CSV",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="metrics_comparison.csv",
    mime="text/csv",
)
