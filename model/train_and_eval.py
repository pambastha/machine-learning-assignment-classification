import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from .preprocessing import build_preprocessor, split_data
from .models_factory import get_models
from .metrics import compute_all_metrics

def train_eval_all(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    # Drop fully-empty columns 
    df = df.dropna(axis=1, how="all").copy()

    # Drop NA target rows (prevents 'nan' becoming a class)
    df = df.dropna(subset=[target_col]).copy()

    # Split (safe stratify inside split_data)
    X_train, X_test, y_train, y_test = split_data(df, target_col, test_size, random_state)

    # Encode labels so that all models (esp. XGBoost) behave consistently
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    n_classes = len(le.classes_)
    is_multiclass = n_classes > 2

    preprocessor = build_preprocessor(X_train)
    models = get_models(random_state=random_state)

    results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        if model is None:
            continue

        pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("clf", model)
        ])

        pipe.fit(X_train, y_train_enc)
        y_pred_enc = pipe.predict(X_test)
        
        try:
            y_proba = pipe.predict_proba(X_test)
        except Exception:
            y_proba = None

        metrics = compute_all_metrics(
            y_true=y_test_enc,
            y_pred=y_pred_enc,
            y_proba=y_proba,
            is_multiclass=is_multiclass,
            n_classes=n_classes
        )

        results[name] = metrics
        fitted_pipelines[name] = pipe

    results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "ML Model Name"})
    return results_df, fitted_pipelines, le
