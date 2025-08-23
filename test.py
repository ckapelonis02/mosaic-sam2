import argparse
import os
import torch
import hydra
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.test_helper import test_generator
from sam2.train_helper import cleanup

# Clean up any previous training or inference state
cleanup()

# Configure CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] += ",expandable_segments:True"

# Initialize Hydra configuration
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_module("sam2", version_base="1.2")

# Build the SAM2 model
sam2_model = build_sam2(
    config_file = "../sam2_configs/sam2_hiera_l.yaml",
    ckpt_path = "/kaggle/input/segment-anything-2/pytorch/sam2-hiera-large/1/sam2_hiera_large.pt",
    device="cuda",
    apply_postprocessing=False
)

# Initialize the automatic mask generator
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=32,
    points_per_batch=4,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.8,
    stability_score_offset=1.0,
    mask_threshold=0.5,
    box_nms_thresh=0.7,
    crop_n_layers=2,
    crop_nms_thresh=0.7,
    crop_overlap_ratio=0.3,
    crop_n_points_downscale_factor=2,
    point_grids=None,
    min_mask_region_area=25.0,
    output_mode="binary_mask",
    use_m2m=False,
    multimask_output=True,
    # load_model=None
)

# Run inference and generate masks
test_generator(
    mask_generator=mask_generator,
    images_path="/kaggle/input/mosaic-dataset/data/test_data/images",
    output_path="/kaggle/input/mosaic-dataset/data/test_data/masks",
    crops_csv_file="/kaggle/input/mosaic-dataset/data/test_data/test_crops.csv",
    max_mask_crop_region=0.1,
    show_masks=False
)