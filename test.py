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

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run SAM2 inference on mosaic images")
parser.add_argument("--images_path", type=str, required=True, help="Path to input images")
parser.add_argument("--output_path", type=str, required=True, help="Path to save output masks")
parser.add_argument("--crops_csv_file", type=str, default=None, help="Optional CSV file with crop info")
parser.add_argument("--max_mask_crop_region", type=float, default=1.0, help="Maximum mask crop region (default=1.0)")
parser.add_argument("--show_masks", action="store_true", help="Display masks during processing (default=False)")
args = parser.parse_args()

# Build the SAM2 model
sam2_model = build_sam2(
    config_file="../sam2_configs/sam2_hiera_t.yaml",
    ckpt_path="checkpoints/sam2_hiera_tiny.pt",
    device="cuda",
    apply_postprocessing=False
)

# Initialize the automatic mask generator
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model,
    points_per_side=40,
    points_per_batch=8,
    pred_iou_thresh=0.75,
    stability_score_thresh=0.8,
    stability_score_offset=0.9,
    mask_threshold=0.4,
    box_nms_thresh=0.6,
    crop_n_layers=3,
    crop_nms_thresh=0.6,
    crop_overlap_ratio=0.2,
    crop_n_points_downscale_factor=1,
    point_grids=None,
    min_mask_region_area=10.0,
    output_mode="binary_mask",
    use_m2m=False,
    multimask_output=True
)

# Run inference and generate masks
test_generator(
    mask_generator=mask_generator,
    images_path=args.images_path,
    output_path=args.output_path,
    crops_csv_file=args.crops_csv_file,
    max_mask_crop_region=args.max_mask_crop_region,
    show_masks=args.show_masks
)
