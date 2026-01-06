from typing import Any
import sys
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    root_mean_squared_error,
    r2_score,
    mean_squared_log_error,
    fbeta_score
)
import python.common as common

def f1_score_multilabel(y_true, y_pred):
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    return f1_score(y_true_bin, y_pred_bin, average="samples")


def apk(actual, predicted, k=5):
    """Average precision at k for one sample."""
    if actual in predicted:
        return 1.0 / (predicted.index(actual) + 1)
    return 0.0


def mean_average_precision_k(y_true, y_pred_topk, k=5):
    """Mean average precision at k over all samples."""
    return np.mean([apk(a, p, k) for a, p in zip(y_true, y_pred_topk)])


def calculate_wae(y_pred: np.array, val: pd.DataFrame) -> float:
    """
    Calculate Weighted Mean Absolute Error For Income

    Args:
        pred : DataFrame with income predictions
        val: DataFrame with weights for evaluation and true targets

    Returns:
        Weighted Mean Absolute Error
    """

    if 'w' not in val.columns:
        weight_vals = np.ones(y_pred.shape[0])
    else:
        weight_vals = val['w'].values

    if 'target' not in val.columns:
        common.report_error("Validation data missing 'target' column")
        return np.nan
    y_true = val['target'].values
    if y_pred.shape != y_true.shape:
        common.report_error("The number of true values and predictions is not equal")
        return np.nan

    return (weight_vals * np.abs(y_true - y_pred)).mean()


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

    """
    Calculate RMSE For col_5 and col_8

    Args:
        pred : DataFrame with predictions for both columns
        val: DataFrame with true targets for both columns

    Returns:
        Mean RMSE value (1/2 * (RMSE_col5 + RMSE_col8))
    """
    try:
        # Convert predictions and validation data to the DF format
        val = _convert_to_dataframe_fin_eng(val)
        y_pred = _convert_to_dataframe_fin_eng(y_pred)

        if 'col_5' not in val.columns:
            # common.report_error("Validation data missing 'col_5' column")
            return np.nan
        if 'col_8' not in val.columns:
            # common.report_error("Validation data missing 'col_8' column")
            return np.nan
        y_true_5 = val['col_5'].values
        y_true_8 = val['col_8'].values

        y_pred_5 = y_pred.iloc[:, 0].values
        y_pred_8 = y_pred.iloc[:, 1].values
        # in description evaluation is defined by using RMSE without further comments
        # count RMSE on each column and count mean

        def rmse(y_true, y_pred):
            return np.sqrt(np.mean((y_true - y_pred) ** 2))

        col5_rmse = rmse(y_true_5, y_pred_5)
        col8_rmse = rmse(y_true_8, y_pred_8)

        return (col5_rmse + col8_rmse) / 2
    except Exception:
        # report_error(f"Recommender grader execution failed: {e}")
        return np.nan

def calculate_rmse(y_pred: pd.DataFrame, val: pd.DataFrame) -> float:
    """
    Calculate RMSE For col_5 and col_8

    Args:
        pred : DataFrame with predictions for 2 columns
        val: DataFrame with true targets for 2 columns

    Returns:
        Mean RMSE value over all 2 columns
    """
    try:
        if ('col_5' not in y_pred.columns or
                'col_8' not in y_pred.columns):
            common.report_error("Validation data missing target columns")
            return np.nan

        if 0 in val.columns or 1 in val.columns or 2 in val.columns:
            val.rename(columns={0: 'col_5', 1: 'col_8'},
                       inplace=True)

        if ('col_5' not in val.columns or
                'col_8' not in val.columns):
            common.report_error("Validation data missing target columns")
            return np.nan

        y_true_col_5 = val['col_5'].values
        y_true_col_8 = val['col_8'].values

        y_pred_col_5 = y_pred.iloc[:, 0].values
        y_pred_col_8 = y_pred.iloc[:, 1].values

        # count RMSE on each column and count mean
        col_5_rmse = np.sqrt(mean_squared_error(y_true_col_5, y_pred_col_5))
        col_8_rmse = np.sqrt(mean_squared_error(y_true_col_8, y_pred_col_8))

        return (col_5_rmse + col_8_rmse) / 2
    except Exception:
        return np.nan
    

def calculate_rmsle(y_pred: pd.DataFrame, val: pd.DataFrame) -> float:
    """
    Calculate RMSLE For target_carbon_monoxide, target_benzene and target_nitrogen_oxides

    Args:
        pred : DataFrame with predictions for 3 columns
        val: DataFrame with true targets for 3 columns

    Returns:
        Mean RMSLE value over all 3 columns
    """
    try:
        if ('target_carbon_monoxide' not in y_pred.columns or
                'target_benzene' not in y_pred.columns or
                'target_nitrogen_oxides' not in y_pred.columns):
            common.report_error("Validation data missing target columns")
            return np.nan

        if 0 in val.columns or 1 in val.columns or 2 in val.columns:
            val.rename(columns={0: 'target_carbon_monoxide', 1: 'target_benzene', 2: 'target_nitrogen_oxides'}, inplace=True)

        if ('target_carbon_monoxide' not in val.columns or
                'target_benzene' not in val.columns or
                'target_nitrogen_oxides' not in val.columns):
            common.report_error("Validation data missing target columns")
            return np.nan

        y_true_carbon_monoxide = val['target_carbon_monoxide'].values
        y_true_benzene = val['target_benzene'].values
        y_true_nitrogen_oxides = val['target_nitrogen_oxides'].values

        y_pred_carbon_monoxide = y_pred.iloc[:, 0].values
        y_pred_benzene = y_pred.iloc[:, 1].values
        y_pred_nitrogen_oxides = y_pred.iloc[:, 2].values

        # count RMSLE on each column and count mean
        carbon_monoxide_rmse = np.sqrt(mean_squared_log_error(y_true_carbon_monoxide, y_pred_carbon_monoxide))
        benzene_rmse = np.sqrt(mean_squared_log_error(y_true_benzene, y_pred_benzene))
        nitrogen_oxides_rmse = np.sqrt(mean_squared_log_error(y_true_nitrogen_oxides, y_pred_nitrogen_oxides))

        return (carbon_monoxide_rmse + benzene_rmse + nitrogen_oxides_rmse) / 3
    except Exception:
        return np.nan
    

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
    "f1_score_multilabel": f1_score_multilabel,
    "mean_average_precision": mean_average_precision_k,
    "r2_score": r2_score,
    "root_mean_squared_logarithmic_error_multitarget": calculate_rmsle,
    "root_mean_squared_error_multitarget": calculate_rmse
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
    except Exception:
        common.report_error(f"Grader execution failed : {sys.exc_info()}")
        return np.nan



# Custom graders
# ==============


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


def grader_multilabel(pred: pd.DataFrame, val: pd.DataFrame, comp: dict):
    """Default grader with multi-label string parsing"""
    metric_name = comp.get("metric", "accuracy_score")
    metric = METRICS.get(metric_name)
    
    if metric is None:
        common.report_error(f"grader_default() : internal error : metric not found : {metric_name}")
        common.graceful_exit(1)
    
    try:
        val_values = val
        if isinstance(pred, pd.DataFrame):
            pred_values = pred.iloc[:, 0].apply(_parse_multi_label_string_grader)
        else:
            pred_values = [_parse_multi_label_string_grader(v) for v in pred]
        
        score = metric(val_values, pred_values)
        return score
    except Exception:
        common.report_error(f"Grader execution failed : {sys.exc_info()}")
        return np.nan


def _parse_multi_label_string_grader(label_str):
    """Helper function for grader to parse multi-label strings"""
    # Same implementation as in DataLoader
    if isinstance(label_str, list):
        return label_str
            
    if not isinstance(label_str, str):
        return [str(label_str)]
            
    if label_str.startswith('[') and label_str.endswith(']'):
        try:
            import ast
            parsed = ast.literal_eval(label_str)
            return [str(item) for item in parsed]
        except:
            cleaned = label_str.strip('[]').replace("u'", "").replace("'", "").replace('"', '')
            return [item.strip() for item in cleaned.split(',') if item.strip()]
    
    return [label_str.strip()]


def grader_biker_recommender(pred: pd.DataFrame, val: pd.DataFrame, comp: dict):
    """
    Grader for recommender systems using Mean Average Precision (MAP).
    Only uses positive interactions (like=1) for evaluation.
    """
    try:
        # Convert predictions and validation data to the right format
        val = _convert_to_dataframe(val)
        pred = _convert_to_dataframe(pred)
        # Filter validation data to only include positive interactions (like=1)
        # Ignore dislikes and non-responses for MAP evaluation
        val_positives = val[val['like'] == 1]
        
        # Group positive validation data by biker_id
        val_grouped = val_positives.groupby('biker_id')['tour_id'].apply(list).to_dict()
        
        # Group predictions by biker_id to get ranked recommendations
        pred_grouped = pred.groupby('biker_id')['tour_id'].apply(list).to_dict()
        
        # Calculate AP for each biker with positive interactions
        ap_scores = []
        for biker_id, gt_tours in val_grouped.items():
            if biker_id in pred_grouped:
                gt_positives = set(gt_tours)
                recommendations = pred_grouped[biker_id]
                k = len(gt_tours)  # K is number of ground truth positives for this user
                ap = _calculate_ap(recommendations, gt_positives, k)
                ap_scores.append(ap)
        
        # Calculate MAP (mean of AP scores)
        if ap_scores:
            map_score = np.mean(ap_scores)
            return float(map_score)
        else:
            return 0.0
            
    except Exception as e:
        common.report_error(f"Recommender grader execution failed: {e}")
        return np.nan

def _calculate_ap(recommendations: List[int], gt_positives: set, k: int) -> float:
    """
    Calculate Average Precision for a single biker.
    k: number of ground truth positives (varies per user)
    """
    if not gt_positives or k == 0:
        return 0.0
    
    precision_values = []
    num_hits = 0
    
    for i, tour_id in enumerate(recommendations):
        if tour_id in gt_positives:
            num_hits += 1
            precision_at_i = num_hits / (i + 1)
            precision_values.append(precision_at_i)
        
        # Stop if we've found all ground truth positives
        if num_hits >= k:
            break
    
    if not precision_values:
        return 0.0
    
    return sum(precision_values) / k

def _convert_to_dataframe(data: Any) -> pd.DataFrame:
    """
    Convert various input formats to DataFrame.
    Handles both validation labels and prediction formats.
    """
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, np.ndarray):
        if data.shape[1] >= 2:
            # Assume array has at least [biker_id, tour_id] columns
            columns = ['biker_id', 'tour_id'] + [f'col_{i}' for i in range(2, data.shape[1])]
            return pd.DataFrame(data, columns=columns)
        else:
            raise ValueError("Array must have at least 2 columns for biker_id and tour_id")
    elif isinstance(data, list):
        if data and isinstance(data[0], (list, tuple)):
            # List of lists/tuples: [[biker_id, tour_id], ...]
            return pd.DataFrame(data, columns=['biker_id', 'tour_id'])
        else:
            raise ValueError("List must contain lists or tuples of [biker_id, tour_id]")
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")


def grader_multitarget(pred: pd.DataFrame, val: pd.DataFrame, comp: dict):
    """
    Specialized grader for multi-target tasks.
    """
    metric = METRICS.get(comp["metric"])
    if metric is None:
        common.report_error(f"grader_multitarget() : internal error : metric not found : {comp['metric']}")
        common.graceful_exit(1)

    try:
        # Handle different prediction formats
        if isinstance(pred, np.ndarray):
            pred = pd.DataFrame(pred)
        elif isinstance(pred, list):
            pred = pd.DataFrame(pred)

        # Handle different validation formats
        if isinstance(val, np.ndarray):
            val = pd.DataFrame(val)
        elif isinstance(val, list):
            val = pd.DataFrame(val)

        # Ensure both pred and val have the same shape
        min_rows = min(pred.shape[0], val.shape[0])
        min_cols = min(pred.shape[1], val.shape[1])
        
        pred_values = pred.iloc[:min_rows, :min_cols]
        val_values = val.iloc[:min_rows, :min_cols]

        score = metric(val_values, pred_values)
        return score
    except Exception:
        common.report_error(f"Multitarget grader execution failed : {sys.exc_info()}")
        return np.nan

def grader_classify_leaves(pred: pd.DataFrame, val: pd.DataFrame, comp: dict):
    """
    Specialized grader for image classification tasks that handles string labels and different prediction formats.
    """
    metric = METRICS.get(comp["metric"])
    if metric is None:
        common.report_error(f"grader_image_classification() : internal error : metric not found : {comp['metric']}")
        common.graceful_exit(1)

    try:
        # Handle different prediction formats
        if isinstance(pred, list):
            # Convert list to pandas Series
            pred = pd.Series(pred)
        elif isinstance(pred, np.ndarray):
            pred = pd.Series(pred)

        # Handle different validation formats
        if isinstance(val, pd.DataFrame):
            val = val.iloc[:, 0] if val.shape[1] == 1 else val
        elif isinstance(val, list):
            val = pd.Series(val)
        elif isinstance(val, np.ndarray):
            val = pd.Series(val)

        # Ensure both pred and val have the same length
        min_len = min(len(pred), len(val))
        pred = pred.iloc[:min_len] if hasattr(pred, 'iloc') else pred[:min_len]
        val = val.iloc[:min_len] if hasattr(val, 'iloc') else val[:min_len]

        # For image classification, we might need to handle string labels
        # Convert to string if needed for comparison
        if pred.dtype == 'object' and val.dtype == 'object':
            pred = pred.astype(str)
            val = val.astype(str)

        score = metric(val, pred)
        return score
    except Exception:
        common.report_error(f"Image classification grader execution failed : {sys.exc_info()}")
        return np.nan

def grader_photo_classification(pred: pd.DataFrame, val: pd.DataFrame, comp: dict):
    """
    Grader for multi-label image classification using sklearn's
    fbeta_score with micro averaging.
    """
    beta = comp.get("beta", 1.0)

    try:
        # Normalize predictions
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0]
        elif isinstance(pred, np.ndarray):
            pred = pd.Series(pred)
        elif isinstance(pred, list):
            pred = pd.Series(pred)

        # Normalize validation labels
        if isinstance(val, pd.DataFrame):
            val = val.iloc[:, -1]
        elif isinstance(val, np.ndarray):
            val = pd.Series(val)
        elif isinstance(val, list):
            val = pd.Series(val)

        # Align lengths
        min_len = min(len(pred), len(val))
        pred = pred.iloc[:min_len]
        val = val.iloc[:min_len]
        
        def to_label_list(x):
            if isinstance(x, (int, np.integer)):
                return [int(x)]
            if isinstance(x, str):
                return list(map(int, x.split()))
            if isinstance(x, (list, set)):
                return list(map(int, x))
            return []

        y_pred = pred.apply(to_label_list).tolist()
        y_true = val.apply(to_label_list).tolist()

        # Fit binarizer on ground truth labels
        mlb = MultiLabelBinarizer()
        y_true_bin = mlb.fit_transform(y_true)

        # Transform predictions using same label space
        y_pred_bin = mlb.transform(y_pred)

        return fbeta_score(
            y_true_bin,
            y_pred_bin,
            beta=beta,
            average="micro",
            zero_division=0
        )

    except Exception:
        common.report_error(
            f"Photo classification grader execution failed : {sys.exc_info()}"
        )
        return np.nan


def grader_sheep_classification(pred: pd.DataFrame, val: pd.DataFrame, comp: dict):
    """
    Specialized grader for sheep classification challenge 
    that handles string labels and different prediction formats.
    """
    metric_name = comp.get("metric", "f1_score")
    
    try:
        if isinstance(pred, list):
            pred = pd.Series(pred)
        elif isinstance(pred, np.ndarray):
            pred = pd.Series(pred)
        elif isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0] if pred.shape[1] >= 1 else pred

        # Handle different validation formats
        if isinstance(val, pd.DataFrame):
            val = val.iloc[:, 0] if val.shape[1] == 1 else val
        elif isinstance(val, list):
            val = pd.Series(val)
        elif isinstance(val, np.ndarray):
            val = pd.Series(val)

        # Ensure both pred and val have the same length
        min_len = min(len(pred), len(val))
        pred = pred.iloc[:min_len] if hasattr(pred, 'iloc') else pred[:min_len]
        val = val.iloc[:min_len] if hasattr(val, 'iloc') else val[:min_len]

        if pred.dtype == 'object' and val.dtype == 'object':
            pred = pred.astype(str)
            val = val.astype(str)

        # For f1_score with multi-class string labels, use average='macro'
        if metric_name == "f1_score":
            score = f1_score(val, pred, average='macro', zero_division=0)
        else:
            metric = METRICS.get(metric_name)
            if metric is None:
                common.report_error(f"grader_sheep_classification() : internal error : metric not found : {metric_name}")
                common.graceful_exit(1)
            score = metric(val, pred)
        
        return score
    except Exception as e:
        common.report_error(f"Sheep classification grader execution failed : {sys.exc_info()}")
        return np.nan


GRADERS = {
    "default": grader_default,
    "prml_nov2020": grader_prml_nov2020,
    "map_at_k": grader_prml_nov2020,  # Alias
    "recommendation": grader_prml_nov2020,  # Alias
    "binary_from_ranking": grader_binary_classification_from_ranking,
    "multilabel": grader_multilabel,
    "multitarget": grader_multitarget,
    "biker_recommender": grader_biker_recommender,
    "classify_leaves": grader_classify_leaves,
    "photo_classification": grader_photo_classification,
    "sheep_classification": grader_sheep_classification
}

