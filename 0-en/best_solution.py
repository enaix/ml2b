import os
import numpy as np
import pandas as pd
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


def _ensure_category(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Convert specified columns to pandas category dtype."""
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def train(X_train: pd.DataFrame, y_train: pd.DataFrame) -> Any:
    """
    Fit a LightGBM regressor.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame): Training target (single column).

    Returns:
        dict: Contains the fitted model and list of categorical columns.
    """
    # Ensure y_train is a 1‑d array
    y = y_train.squeeze()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    X_train = _ensure_category(X_train.copy(), cat_cols)

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        learning_rate=0.05,
        n_estimators=5000,
        num_leaves=31,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y, categorical_feature=cat_cols)
    return {"model": model, "cat_cols": cat_cols}


def prepare_val(X_val: pd.DataFrame, train_output: Any) -> Any:
    """
    Align validation features with training categorical columns.

    Args:
        X_val (pd.DataFrame): Validation features.
        train_output (Any): Output from `train`.

    Returns:
        pd.DataFrame: Processed validation features.
    """
    cat_cols = train_output["cat_cols"]
    return _ensure_category(X_val.copy(), cat_cols)


def predict(train_output: Any, prepare_val_output: Any) -> np.ndarray:
    """
    Generate predictions for processed validation/test data.

    Args:
        train_output (Any): Output from `train`.
        prepare_val_output (Any): Processed features.

    Returns:
        np.ndarray: Predicted values.
    """
    model = train_output["model"]
    return model.predict(prepare_val_output)


def run(
    X_train: pd.DataFrame, y_train: pd.DataFrame, X_val: pd.DataFrame
) -> np.ndarray:
    """
    Entrypoint for the submission: train on X_train/y_train and predict on X_val.

    Returns:
        np.ndarray: Predictions for X_val.
    """
    train_output = train(X_train, y_train)
    prepared_X_val = prepare_val(X_val, train_output)
    return predict(train_output, prepared_X_val)


if __name__ == "__main__":
    # Load training data
    train_path = os.path.join("input", "train.csv")
    df = pd.read_csv(train_path)

    target_col = "Unique Headcount"
    X = df.drop(columns=[target_col])
    y = df[[target_col]]  # keep as DataFrame to match signature

    # Hold‑out split for evaluation
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get predictions on validation set
    val_preds = run(X_tr, y_tr, X_val)

    # Compute and print RMSE
    rmse = mean_squared_error(y_val.squeeze(), val_preds, squared=False)
    print(f"Hold‑out RMSE: {rmse:.5f}")

    # Retrain on full data for final submission
    full_train_output = train(X, y)

    # If test data exists, generate submission
    test_path = os.path.join("input", "test.csv")
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        test_features = test_df.copy()
        test_prepared = prepare_val(test_features, full_train_output)
        test_preds = predict(full_train_output, test_prepared)

        submission = pd.DataFrame(
            {
                "Id": (
                    test_df["Id"]
                    if "Id" in test_df.columns
                    else np.arange(len(test_df))
                ),
                target_col: test_preds,
            }
        )
        os.makedirs("working", exist_ok=True)
        submission_path = os.path.join("working", "submission.csv")
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
