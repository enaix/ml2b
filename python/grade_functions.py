import common

import numpy as np
import pandas as pd
import sys
from typing import List, Dict, Tuple

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    precision_score,
    recall_score,
    matthews_corrcoef,
    balanced_accuracy_score
)

METRICS = {
    "roc_auc_score": roc_auc_score,
    "f1_score": f1_score,
    "accuracy_score": accuracy_score,
    "f1_score_avg_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
    "f1_score_avg_weighted": lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
    "precision_score_macro": lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
    "recall_score_macro": lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
    "root_mean_squared_error": root_mean_squared_error,
    "log_loss": log_loss,
    "mean_squared_error": mean_squared_error,
    "mean_absolute_error": mean_absolute_error,
    "matthews_corrcoef": matthews_corrcoef,
    "balanced_accuracy": balanced_accuracy_score,
}

# Default grader
# ==============

def grader_default(pred: pd.DataFrame, val: pd.DataFrame, comp: dict):
    """Default grader using specified metric from competition config"""
    metric_name = comp.get("metric", "accuracy_score")
    metric = METRICS.get(metric_name)

    if metric is None:
        common.report_error(f"grader_default() : internal error : metric not found : {metric_name}")
        common.graceful_exit(1)

    try:
        # Handle different input formats
        if isinstance(pred, pd.DataFrame):
            pred_values = pred.iloc[:, 0] if pred.shape[1] == 1 else pred.values.flatten()
        else:
            pred_values = pred

        if isinstance(val, pd.DataFrame):
            val_values = val.iloc[:, 0] if val.shape[1] == 1 else val.values.flatten()
        else:
            val_values = val

        score = metric(val_values, pred_values)
        return score
    except Exception as e:
        common.report_error(f"Grader execution failed : {sys.exc_info()}")
        return np.nan

    

# Custom graders
# ==============

def calculate_ap_at_k(y_true_tours: List[int], predicted_ranking: List[int], k: int = None) -> float:
    """
    Calculate Average Precision at K for a single biker
    
    Args:
        y_true_tours: List of tour_ids that the biker actually liked
        predicted_ranking: List of tour_ids in predicted preference order
        k: Maximum number of recommendations to consider (if None, use all)
    
    Returns:
        Average Precision at K score
    """
    # TODO add try catch
    if not y_true_tours:
        return 0.0
    
    if k is None:
        k = len(predicted_ranking)

    # Truncate predictions to k
    predicted_ranking = predicted_ranking[:k]

    gtp = len(y_true_tours)  # Ground Truth Positives
    y_true_set = set(y_true_tours)

    precision_sum = 0.0
    num_hits = 0

    for i, tour_id in enumerate(predicted_ranking, 1):
        if tour_id in y_true_set:
            num_hits += 1
            precision_at_i = num_hits / i
            precision_sum += precision_at_i

    if gtp == 0:
        return 0.0

    return precision_sum / gtp


def calculate_map_at_k(y_true_dict: Dict[int, List[int]], predictions_dict: Dict[int, List[int]], k: int = None) -> float:
    """
    Calculate Mean Average Precision at K across all bikers
    
    Args:
        y_true_dict: Dictionary mapping biker_id -> list of liked tour_ids
        predictions_dict: Dictionary mapping biker_id -> list of predicted tour_ids (ranked)
        k: Maximum number of recommendations to consider
    
    Returns:
        Mean Average Precision at K score
    """
    # TODO add try catch
    ap_scores = []
    
    for biker_id in y_true_dict:
        if biker_id not in predictions_dict:
            # If no predictions for this biker, AP = 0
            ap_scores.append(0.0)
            continue
        
        y_true_tours = y_true_dict[biker_id]
        predicted_ranking = predictions_dict[biker_id]
        
        ap_score = calculate_ap_at_k(y_true_tours, predicted_ranking, k)
        ap_scores.append(ap_score)
    
    return np.mean(ap_scores)


def grader_prml_nov2020(pred: pd.DataFrame, val: pd.DataFrame, comp: dict) -> float:
    """
    Grader for PRML Data Contest Nov 2020 - Tour Recommendation System
    
    Expected format:
    - pred: DataFrame with columns ['biker_id', 'tour_id', 'rank'] or ['biker_id', 'tour_id', 'score']
            where rank=1 is most preferred, or higher score means more preferred
    - val: DataFrame with validation data containing ground truth likes
           Expected columns: ['biker_id', 'tour_id', 'like', 'dislike']
    """
    
    try:
        # Extract ground truth - tours that bikers actually liked
        # A biker likes a tour if like=1 (and optionally dislike=0)
        if 'like' not in val.columns:
            common.report_error("Validation data missing 'like' column")
            return np.nan
        
        # Ground truth: tours where like=1
        ground_truth = val[val['like'] == 1].groupby('biker_id')['tour_id'].apply(list).to_dict()
        
        # Handle predictions based on format
        if 'rank' in pred.columns:
            # Format 1: Explicit ranking (rank=1 is best)
            pred_sorted = pred.sort_values(['biker_id', 'rank'])
            predictions = pred_sorted.groupby('biker_id')['tour_id'].apply(list).to_dict()
            
        elif 'score' in pred.columns:
            # Format 2: Scores (higher is better)
            pred_sorted = pred.sort_values(['biker_id', 'score'], ascending=[True, False])
            predictions = pred_sorted.groupby('biker_id')['tour_id'].apply(list).to_dict()
            
        elif len(pred.columns) == 3:
            # Format 3: Assume third column is score/rank
            score_col = pred.columns[2]
            if pred[score_col].min() >= 0 and pred[score_col].max() <= len(pred):
                # Looks like rank (small is better)
                pred_sorted = pred.sort_values(['biker_id', score_col])
            else:
                # Looks like score (high is better)
                pred_sorted = pred.sort_values(['biker_id', score_col], ascending=[True, False])
            predictions = pred_sorted.groupby('biker_id')['tour_id'].apply(list).to_dict()
            
        else:
            common.report_error(f"Unexpected prediction format. Columns: {pred.columns.tolist()}")
            return np.nan
        
        # Get K from competition config
        k = comp.get("map_k", 10)  # Default to MAP@10 if not specified
        
        # Calculate MAP@K
        map_score = calculate_map_at_k(ground_truth, predictions, k)
        
        print(f"MAP@{k} = {map_score:.4f}")
        print(f"Ground truth bikers: {len(ground_truth)}")
        print(f"Prediction bikers: {len(predictions)}")
        
        return map_score
        
    except Exception as e:
        common.report_error(f"PRML Nov 2020 grader execution failed: {str(e)}")
        # TODO remove ALL print statements
        print(f"Prediction columns: {pred.columns.tolist() if isinstance(pred, pd.DataFrame) else 'Not DataFrame'}")
        print(f"Validation columns: {val.columns.tolist() if isinstance(val, pd.DataFrame) else 'Not DataFrame'}")
        print(f"Prediction shape: {pred.shape if hasattr(pred, 'shape') else 'No shape'}")
        print(f"Validation shape: {val.shape if hasattr(val, 'shape') else 'No shape'}")
        return np.nan


def grader_binary_classification_from_ranking(pred: pd.DataFrame, val: pd.DataFrame, comp: dict) -> float:
    """
    Alternative grader that converts the ranking problem to binary classification
    for easier debugging and comparison with other metrics
    """
    
    try:
        # Convert ranking predictions to binary predictions
        # Assume top-K ranked items are predicted as positive
        k_threshold = comp.get("binary_threshold", 5)
        
        # Create binary predictions
        if 'rank' in pred.columns:
            pred_binary = pred.copy()
            pred_binary['pred_like'] = (pred_binary['rank'] <= k_threshold).astype(int)
        elif 'score' in pred.columns:
            # Use top K scores as positive predictions
            pred_binary = pred.copy()
            pred_binary['rank'] = pred_binary.groupby('biker_id')['score'].rank(method='first', ascending=False)
            pred_binary['pred_like'] = (pred_binary['rank'] <= k_threshold).astype(int)
        else:
            common.report_error("Cannot convert predictions to binary format")
            return np.nan
        
        # Merge with validation data
        merged = pd.merge(pred_binary, val, on=['biker_id', 'tour_id'], how='inner')
        
        if len(merged) == 0:
            common.report_error("No matching records between predictions and validation")
            return np.nan
        
        # Calculate binary classification metrics
        y_true = merged['like'].values
        y_pred = merged['pred_like'].values
        
        # Use specified metric or default to accuracy
        metric_name = comp.get("metric", "accuracy_score")
        if metric_name == "roc_auc_score":
            score = roc_auc_score(y_true, y_pred)
        elif metric_name == "f1_score":
            score = f1_score(y_true, y_pred)
        else:
            score = accuracy_score(y_true, y_pred)
        
        print(f"Binary classification {metric_name}: {score:.4f}")
        print(f"Positive predictions: {y_pred.sum()}/{len(y_pred)}")
        print(f"Positive ground truth: {y_true.sum()}/{len(y_true)}")
        
        return score
        
    except Exception as e:
        common.report_error(f"Binary classification grader failed: {str(e)}")
        return np.nan


# Updated GRADERS registry
GRADERS = {
    "default": grader_default,
    "prml_nov2020": grader_prml_nov2020,
    "map_at_k": grader_prml_nov2020,  # Alias
    "recommendation": grader_prml_nov2020,  # Alias
    "binary_from_ranking": grader_binary_classification_from_ranking,
}

