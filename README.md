# Mosaic-SAM2: Automatic Tesserae Segmentation Using SAM2

## Overview
**Mosaic-SAM2** is a PyTorch-based framework for segmenting mosaic images into individual tesserae using the **SAM2 model**.

## Requirements

### Python
- Python 3.10+ recommended.

### Conda environment & CLI setup
```bash
# Clone the repo
git clone https://github.com/ckapelonis02/mosaic-sam2.git
cd mosaic-sam2

# Create and activate conda environment
conda create -n mosaic-sam2 python=3.10 -y
conda activate mosaic-sam2

# Install package in editable mode
pip install -e .

# Build extensions
python setup.py build_ext --inplace
```

## Dataset

Structure:
```
data/
├── images/   # Original mosaic images in .jpg
└── masks/    # Binary masks (.png) with tesserae as white (255)
```

## Training

The `train_sam2` function is the core of fine-tuning SAM 2. For details check the `train.py` module.

```python
train_sam2(
    images_path=images_path,      # Path to training images
    masks_path=masks_path,        # Path to ground-truth masks
    epochs=10,                    # Number of training epochs
    grad_steps=4,                 # Gradient accumulation steps (effective batch size)
    log_dir=f"runs/log_run",      # Directory for logs and checkpoints
    predictor=predictor,          # SAM 2 predictor object
    optimizer=optimizer,          # Optimizer for training
    scheduler=scheduler,          # Learning rate scheduler
    seed=22,                      # Random seed for reproducibility
    train_percentage=0.8,         # Fraction of dataset used for training
    score_weight=0.2,             # Weight for the stability score in loss
    config_file=config_file,      # Path to config file
    ckpt_path=ckpt_path,          # Path to checkpoint
    points_per_side=16,           # (validation configs) Number of points per side for SAM 2 guided training
    points_per_batch=4,           # (validation configs) Number of points sampled per batch
    pred_iou_thresh=0.7,          # (validation configs) IoU threshold for predicted masks
    stability_score_thresh=0.7,   # (validation configs) Threshold for stability score
    stability_score_offset=1.0,   # (validation configs) Offset for stability score calculation
    mask_threshold=0.5,           # (validation configs) Threshold for binarizing predicted masks
)
```

## Testing / Inference

```
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,                    # The model SAM 2 object
    points_per_side=32,                  # Number of points sampled per side
    points_per_batch=4,                  # Number of points per batch
    pred_iou_thresh=0.8,                 # IoU threshold to accept predicted masks
    stability_score_thresh=0.8,          # Stability score threshold
    stability_score_offset=1.0,          # Offset for stability score calculation
    mask_threshold=0.5,                  # Threshold to binarize predicted masks
    box_nms_thresh=0.7,                  # NMS threshold for predicted boxes
    crop_n_layers=2,                     # Number of layers for cropping (multi-scale)
    crop_nms_thresh=0.7,                 # NMS threshold for crops
    crop_overlap_ratio=0.3,              # Overlap ratio between crops
    crop_n_points_downscale_factor=2,    # Downscale factor for points in crops
    point_grids=None,                    # Optional predefined point grids
    min_mask_region_area=25.0,           # Minimum area to keep a mask region
    output_mode="binary_mask",           # Output as binary mask
    use_m2m=False,                       # Whether to use mask-to-mask refinement
    multimask_output=True,               # Generate multiple mask proposals per object
    load_model="fine_tuned.torch"        # Optionally load a pretrained checkpoint
)

test_generator(
    mask_generator=mask_generator,
    images_path="PATH/TO/IMAGES",        # Directory containing test images
    output_path="PATH/TO/OUTPUT/MASKS",  # Where to save generated masks
    crops_csv_file="PATH/TO/CROPS.CSV",  # Optional CSV for specific crop regions
    max_mask_crop_region=0.1,            # Max fraction of image to crop for mask generation
    show_masks=False                     # Set True to visualize masks during generation
)
```

## CSV File Format for Mask Generation

The `test_generator` function can optionally take a **CSV file** to specify how images should be split into crops for mask generation. Each row corresponds to **one image** and specifies how many rows and columns to divide it into.

### Required Columns

| Column Name | Description |
|-------------|-------------|
| `file_name` | Name of the image file (without extension, e.g., `mosaic_001`) |
| `rows`      | Number of horizontal splits (rows) to divide the image into |
| `cols`      | Number of vertical splits (columns) to divide the image into |

- If a CSV is **not provided**, all images in the folder are assumed to have `rows=1` and `cols=1` (i.e., no cropping).  
- If `file_list` is provided, it overrides the CSV and assumes `rows=1`, `cols=1` for all listed images.  

### Example CSV

```csv
file_name,rows,cols
mosaic_001,2,2
mosaic_002,1,1
mosaic_003,3,2
```

## Evaluation of Predicted Masks

This script evaluates predicted segmentation masks against ground-truth masks using common segmentation metrics:
- Dice Coefficient
- Intersection over Union (IoU)
- Precision
- Recall
- Accuracy

### Usage

```python
# Example usage
from evaluate import compute_metrics_batch

gt_masks_dir = "path/to/ground_truth_masks"
pred_masks_dir = "path/to/predicted_masks"

# Compute average metrics across all mask pairs
results = compute_metrics_batch(gt_masks_dir, pred_masks_dir)
print(results)
```

- `gt_masks_dir`: Directory containing ground-truth masks.
- `pred_masks_dir`: Directory containing predicted masks.

### Example Output

```
IoU                0.842
Precision          0.875
Recall             0.810
Accuracy           0.950
Dice Coefficient   0.841
```
This summary gives a quick overview of segmentation performance for your dataset.

## Citation

TODO
