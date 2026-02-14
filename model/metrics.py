import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef
)

def compute_auc(y_true, y_proba, is_multiclass: bool, n_classes: int):
    """
    Computes ROC-AUC when probabilities are available.
    Returns NaN if AUC is not defined for the current split/model output.
    """
    if y_proba is None:
        return np.nan

    y_proba = np.asarray(y_proba)

    try:
        if not is_multiclass:
            # Expect shape (n,2). Use proba for positive class (1)
            if y_proba.ndim == 2 and y_proba.shape[1] >= 2:
                return roc_auc_score(y_true, y_proba[:, 1])
            # If model returns only 1 column or 1D scores
            if y_proba.ndim == 1:
                return roc_auc_score(y_true, y_proba)
            return np.nan
        else:
            # Multiclass needs (n_samples, n_classes)
            if y_proba.ndim != 2 or y_proba.shape[1] != n_classes:
                return np.nan
            return roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        return np.nan


def compute_all_metrics(y_true, y_pred, y_proba, is_multiclass: bool, n_classes: int):
    avg = "weighted" if is_multiclass else "binary"
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(compute_auc(y_true, y_proba, is_multiclass, n_classes)),
        "Precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
    return metrics
