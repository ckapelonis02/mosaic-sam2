import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from datetime import datetime
import hydra

from sam2.train_helper import cleanup, train_sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Clear any previous SAM 2 state and free resources
cleanup()

# Configure CUDA memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] += ",expandable_segments:True"

# Initialize Hydra configuration
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_module('sam2', version_base='1.2')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train SAM2 model on mosaic images")
parser.add_argument("--images_path", type=str, required=True, help="Path to training images")
parser.add_argument("--masks_path", type=str, required=True, help="Path to corresponding masks")
parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
parser.add_argument("--grad_steps", type=int, default=4, help="Number of gradient accumulation steps")
parser.add_argument("--train_percentage", type=float, default=0.8, help="Percentage of data used for training")
parser.add_argument("--score_weight", type=float, default=0.2, help="Weighting factor for the dice loss")
args = parser.parse_args()

# Load SAM 2 model from configuration and checkpoint
config_file = "../sam2_configs/sam2_hiera_t.yaml"
ckpt_path = "checkpoints/sam2_hiera_tiny.pt"
sam2_model = build_sam2(
    config_file=config_file,
    ckpt_path=ckpt_path,
    device="cuda",
    apply_postprocessing=False
)

# Initialize SAM2 predictor and enable training for mask decoder and prompt encoder
predictor = SAM2ImagePredictor(sam2_model)
predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)
# predictor.model.image_encoder.train(True)

# Set up optimizer and learning rate scheduler
optimizer = optim.AdamW(predictor.model.parameters(), lr=1e-5, weight_decay=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# Begin training loop
train_sam2(
    images_path=args.images_path,
    masks_path=args.masks_path,
    epochs=args.epochs,
    grad_steps=args.grad_steps,
    log_dir=f"runs/sam2_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    predictor=predictor,
    optimizer=optimizer,
    scheduler=scheduler,
    seed=22,
    train_percentage=args.train_percentage,
    score_weight=0.2,
    config_file=config_file,
    ckpt_path=ckpt_path,
    points_per_side=16,
    points_per_batch=4,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.7,
    stability_score_offset=1.0,
    mask_threshold=0.5,
)
