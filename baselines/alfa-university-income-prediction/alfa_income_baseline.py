from typing import Any
import lightgbm as lgb
import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split


def process_col(df: pd.DataFrame, col: str, categories_map: dict = None):
    if df[col].dtype == 'object' or df[col].dtype == 'str':
        if categories_map and col in categories_map:
            # Use predefined categories from training
            df[col] = pd.Categorical(df[col], categories=categories_map[col])
        else:
            df[col] = df[col].astype('category')

def process_na(df: pd.DataFrame, col: str):
    if df[col].dtype != 'category':
        df[col] = df[col].fillna(-1)
    else:
        if df[col].isna().any():
            if 'missing' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(['missing'])
            df[col] = df[col].fillna('missing')

def train(X: pd.DataFrame, y: pd.DataFrame):
    """
    This function takes training data and returns the trained model and any intermediate variables
    """
    all_features = sorted(list(set(X.columns) - set(['client_id', 'feature_date'])))
    X_processed = X[all_features].copy()

    for col in all_features:
        process_col(X_processed, col)
        process_na(X_processed, col)

    categorical_features = X_processed.select_dtypes(include=['category']).columns.tolist()

    categories_map = {}
    for col in categorical_features:
        categories_map[col] = X_processed[col].cat.categories.tolist()

    # Create LightGBM dataset
    train_data = lgb.Dataset(
        X_processed,
        label=y,
        categorical_feature=categorical_features,
        free_raw_data=False
    )

    # Set parameters for regression
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'seed': 42,
        'num_threads': 1,
        'verbose': 1,
        'deterministic': True,
        'force_row_wise': True,
    }

    # Train the model
    num_boost_round = 500
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data],
        valid_names=['train']
    )

    return {
        'model': model,
        'feature_names': all_features,
        'categorical_features': categorical_features,
        'categories_map': categories_map,
        'num_boost_round': num_boost_round
    }

def prepare_val(train_output: Any, df: pd.DataFrame):
    """
    This function takes train function output and processed validation features

    train_output (Any): Output from the train function
    """
    feature_names = train_output['feature_names']
    categories_map = train_output['categories_map']
    X_val = df[feature_names].copy()

    for col in feature_names:
        process_col(X_val, col, categories_map)
        process_na(X_val, col)
    return X_val


def predict(train_output: Any, prepare_val_output: Any):
    """
    This function takes train and prepare_val functions outputs and generates the prediction for validation features

    Args:
        train_output (Any): Output from the train function
        prepare_val_output (Any): Output from the prepare_val function, which is the processed X_val dataframe
    """
    model = train_output['model']
    num_boost_round = train_output['num_boost_round']
    X_val = prepare_val_output

    # Make predictions
    predictions = model.predict(X_val, num_iteration=num_boost_round)
    return predictions


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    #path = "data/alfa-university-income-prediction/train.csv"
    path = "train.csv"
    df = pd.read_csv(path, sep=";", decimal=",", encoding="windows-1251")
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["w", "target"]), df[["w", "target"]], test_size=0.2, random_state=42, shuffle=True)
    tr_out = train(X_train, y_train["target"])
    val_out = prepare_val(tr_out, X_test)
    out = predict(tr_out, val_out)

    y_true = y_test['target']
    weights = y_test['w']
    print("wmae score :", (weights * np.abs(y_true - np.asarray(out))).mean())
