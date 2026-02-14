import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Builds a preprocessing pipeline:
    - Numeric: median impute + standard scale
    - Categorical: most-frequent impute + one-hot encode
    """
    X2 = X.copy()

    # Safe numeric conversion for object columns
    for col in X2.columns:
        if X2[col].dtype == "object":
            converted = pd.to_numeric(X2[col], errors="ignore")
            X2[col] = converted

    num_cols = X2.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X2.columns if c not in num_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int = 42
):
    """
    It splits data with stratification only when its valid.
    It also drops NA targets to prevent accidental "nan class".
    """

    df = df.dropna(subset=[target_col]).copy()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    vc = y.value_counts(dropna=False)
    min_count = int(vc.min()) if len(vc) else 0

    stratify_arg = y if min_count >= 2 else None

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg
    )
