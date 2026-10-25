"""Task 5: split and preprocess."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, PolynomialFeatures,
                                   RobustScaler, StandardScaler)

from . import config

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Return X_train, X_test, y_train, y_test.

    Stratified on config.TARGET, seeded with config.SEED.
    Neither the target nor 'mag' may remain in X.
    """
    y = df[config.TARGET]
    X = df.drop(columns=[config.TARGET, "mag"], errors="ignore")
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=config.SEED)


def build_preprocessor(scaler: str = "robust", degree: int = 1) -> ColumnTransformer:
    """ColumnTransformer over config.NUMERIC and config.NOMINAL.

    numeric: median imputation, polynomial features when degree > 1 (no bias
             column), then a scaler chosen by name ('standard', 'minmax', 'robust')
    nominal: most-frequent imputation, then one-hot (handle_unknown='ignore')
    """
    if scaler not in SCALERS:
        raise ValueError(f"Unknown scaler {scaler!r}; choose from {sorted(SCALERS)}")
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    steps = [("impute", SimpleImputer(strategy="median"))]
    if degree > 1:
        steps.append(("poly", PolynomialFeatures(degree, include_bias=False)))
    steps.append(("scale", SCALERS[scaler]()))
    numeric = Pipeline(steps)
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, config.NUMERIC),
        ("nom", nominal, config.NOMINAL),
    ], remainder="drop")
