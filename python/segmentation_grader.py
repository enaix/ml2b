import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import sys

def rle_decode(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
    """
    Decode run-length encoded mask to binary mask.
    
    Args:
        mask_rle: Run-length encoded mask string
        shape: (height, width) of the image
        
    Returns:
        Binary mask as numpy array
    """
    if pd.isna(mask_rle) or mask_rle == '':
        return np.zeros(shape, dtype=np.uint8)
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(shape)


def rle_encode(mask: np.ndarray) -> str:
    """
    Encode binary mask to run-length encoded string.
    
    Args:
        mask: Binary mask as numpy array
        
    Returns:
        Run-length encoded string
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        IoU score
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_image_ap(gt_masks: List[np.ndarray], pred_masks: List[np.ndarray], 
                      iou_thresholds: List[float]) -> float:
    """
    Calculate Average Precision for a single image across IoU thresholds.
    
    Args:
        gt_masks: List of ground truth binary masks
        pred_masks: List of predicted binary masks
        iou_thresholds: List of IoU thresholds
        
    Returns:
        Average Precision score for the image
    """
    if len(gt_masks) == 0 and len(pred_masks) == 0:
        return 1.0 
    
    if len(gt_masks) == 0:
        return 0.0
    
    if len(pred_masks) == 0:
        return 0.0
    
    # calculate IoU matrix between all GT and predicted masks
    iou_matrix = np.zeros((len(gt_masks), len(pred_masks)))
    for i, gt_mask in enumerate(gt_masks):
        for j, pred_mask in enumerate(pred_masks):
            iou_matrix[i, j] = calculate_iou(gt_mask, pred_mask)
    
    ap_scores = []
    
    for threshold in iou_thresholds:
        matches = iou_matrix >= threshold
        
        tp = 0
        matched_gt = set()
        matched_pred = set()
        
        pred_ious = []
        for j in range(len(pred_masks)):
            max_iou = np.max(iou_matrix[:, j]) if len(gt_masks) > 0 else 0
            pred_ious.append((j, max_iou))
        pred_ious.sort(key=lambda x: x[1], reverse=True)
        
        for pred_idx, _ in pred_ious:
            best_gt_idx = -1
            best_iou = 0
            
            for gt_idx in range(len(gt_masks)):
                if gt_idx not in matched_gt and matches[gt_idx, pred_idx]:
                    if iou_matrix[gt_idx, pred_idx] > best_iou:
                        best_iou = iou_matrix[gt_idx, pred_idx]
                        best_gt_idx = gt_idx
            
            if best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
                tp += 1
        
        fp = len(pred_masks) - tp
        fn = len(gt_masks) - tp
        
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        
        ap_scores.append(precision)
    
    return np.mean(ap_scores)


def parse_prediction_format(pred_row: str) -> Tuple[str, int, int, List[str]]:
    """
    Parse prediction row format: 'image_id,width height,mask1 mask2 ...'
    
    Args:
        pred_row: Prediction string
        
    Returns:
        Tuple of (image_id, width, height, list_of_masks)
    """
    parts = pred_row.split(',', 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid prediction format: {pred_row}")
    
    image_id = parts[0]
    width, height = map(int, parts[1].split())
    masks = parts[2].split() if parts[2].strip() else []
    
    return image_id, width, height, masks


def grader_segmentation_map(pred: pd.DataFrame, val: pd.DataFrame, comp: dict) -> float:
    """
    Grader for neuronal cell segmentation using mean Average Precision.
    
    Args:
        pred: Predictions DataFrame with format 'image_id,width height,mask1 mask2 ...'
        val: Validation DataFrame with ground truth annotations
        comp: Competition configuration
        
    Returns:
        Mean Average Precision score
    """
    try:
        eval_details = comp.get("evaluation_details", {})
        iou_thresholds = eval_details.get("iou_thresholds", 
                                        [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        
        gt_by_image = {}
        for _, row in val.iterrows():
            image_parts = str(row['id']).split('_')
            if len(image_parts) >= 2:
                image_id = '_'.join(image_parts[:-1])
            else:
                image_id = str(row['id'])
                
            if image_id not in gt_by_image:
                gt_by_image[image_id] = {
                    'width': row['width'],
                    'height': row['height'], 
                    'masks': []
                }
            
            mask = rle_decode(str(row['annotation']), (row['height'], row['width']))
            gt_by_image[image_id]['masks'].append(mask)
        
        pred_by_image = {}
        if isinstance(pred, pd.DataFrame):
            pred_values = pred.iloc[:, 0] if pred.shape[1] == 1 else pred.values.flatten()
        else:
            pred_values = pred
        
        for pred_row in pred_values:
            if pd.isna(pred_row) or str(pred_row).strip() == '':
                continue
                
            try:
                image_id, width, height, mask_strs = parse_prediction_format(str(pred_row))
                
                masks = []
                for mask_str in mask_strs:
                    if mask_str.strip():
                        mask = rle_decode(mask_str, (height, width))
                        masks.append(mask)
                
                pred_by_image[image_id] = masks
                
            except Exception as e:
                print(f"Warning: Could not parse prediction row: {pred_row}. Error: {e}")
                continue
        
        image_aps = []
        for image_id, gt_data in gt_by_image.items():
            gt_masks = gt_data['masks']
            pred_masks = pred_by_image.get(image_id, [])
            
            image_ap = calculate_image_ap(gt_masks, pred_masks, iou_thresholds)
            image_aps.append(image_ap)
        
        for image_id, pred_masks in pred_by_image.items():
            if image_id not in gt_by_image:
                image_ap = 0.0
                image_aps.append(image_ap)
        
        if len(image_aps) == 0:
            return 0.0
        
        mean_ap = np.mean(image_aps)
        
        print(f"Evaluated {len(image_aps)} images")
        print(f"Mean Average Precision: {mean_ap:.4f}")
        print(f"IoU thresholds: {iou_thresholds}")
        
        return float(mean_ap)
        
    except Exception as e:
        print(f"Segmentation grader execution failed: {str(e)}")
        print(f"Prediction type: {type(pred)}")
        print(f"Validation type: {type(val)}")
        if hasattr(pred, 'shape'):
            print(f"Prediction shape: {pred.shape}")
        if hasattr(val, 'shape'):
            print(f"Validation shape: {val.shape}")
        return np.nan

def mean_average_precision_segmentation(y_true, y_pred):
    """
    Metric function for segmentation mAP.
    Note: This is a simplified version - full evaluation happens in grader_segmentation_map
    This function is used when grader is not available or for quick validation.
    """
    if not isinstance(y_true, (list, np.ndarray)) or not isinstance(y_pred, (list, np.ndarray)):
        return 0.0
    
    if len(y_true) != len(y_pred):
        return 0.0
    
    correct = sum(1 for i, j in zip(y_true, y_pred) if str(i).strip() == str(j).strip())
    return correct / len(y_true) if len(y_true) > 0 else 0.0
