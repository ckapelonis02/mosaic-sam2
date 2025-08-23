import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
import argparse

def read_masks(gt_mask_path, pred_mask_path):
    """
    Read ground-truth and predicted masks as binary numpy arrays.
    """
    gt_mask = 1 - cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    pred_mask = 1 - cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if gt_mask is None or pred_mask is None:
        raise ValueError(f"Invalid image path: {gt_mask_path} or {pred_mask_path}")
    
    gt_mask = (gt_mask > 128).astype(np.uint8)
    pred_mask = (pred_mask > 128).astype(np.uint8)
    
    return gt_mask, pred_mask

def evaluate_pred(gt_mask, pred_mask):
    """
    Compute common segmentation metrics for a single pair of masks.
    """
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    true_positive = intersection
    false_positive = pred_mask.sum() - true_positive
    false_negative = gt_mask.sum() - true_positive
    true_negative = np.logical_not(np.logical_or(gt_mask, pred_mask)).sum()
    
    iou = intersection / union if union != 0 else 0.0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0.0
    accuracy = (true_positive + true_negative) / gt_mask.size if gt_mask.size != 0 else 0.0
    dice = (2 * true_positive) / (2 * true_positive + false_positive + false_negative) if (2 * true_positive + false_positive + false_negative) != 0 else 0.0
    
    return {
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "Dice Coefficient": dice,
    }

def compute_metrics_batch(gt_masks_dir, pred_masks_dir):
    """
    Compute average metrics across all mask pairs in the directories.
    """
    metrics_list = []
    gt_files = set(os.listdir(gt_masks_dir))
    pred_files = set(os.listdir(pred_masks_dir))
    common_files = gt_files.intersection(pred_files)
    
    for filename in tqdm(common_files, desc="Evaluating masks"):
        gt_mask_path = os.path.join(gt_masks_dir, filename)
        pred_mask_path = os.path.join(pred_masks_dir, filename)
        gt_mask, pred_mask = read_masks(gt_mask_path, pred_mask_path)
        metrics_list.append(evaluate_pred(gt_mask, pred_mask))
        
    return pd.DataFrame(metrics_list).mean()

def main():
    mean_metrics = compute_metrics_batch("gt_dir", "pred_dir").round(3)
    
    print("\n### Metrics ###")
    print(mean_metrics)

if __name__ == "__main__":
    main()
