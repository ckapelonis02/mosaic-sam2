import hydra
import numpy as np
import torch
import random
import cv2
import os
import gc
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from torch.utils.tensorboard import SummaryWriter
from sam2.test_helper import *
from evaluate import *

def cleanup():
    """
    Clear GPU memory and Python garbage to prevent memory leaks.
    """
    gc.collect()
    torch.cuda.empty_cache()

def read_dataset(images_path, masks_path, file_names):
    """
    Generate dataset dictionary with paths to images and masks.
    
    Args:
        images_path: Directory containing images.
        masks_path: Directory containing corresponding masks.
        file_names: List of file names (without extensions) to include.

    Returns:
        List of dictionaries with 'image' and 'masks' keys.
    """
    return [
        {
            "image": os.path.join(images_path, f"{file}.jpg"),
            "masks": os.path.join(masks_path, f"{file}.png"),
        }
        for file in file_names
    ]

def read_batch(data_dict, index, max_masks=1, max_res=1024):
    """
    Read a single dataset entry and prepare masks and points for SAM 2 training.
    
    Args:
        data_dict: List of dataset dictionaries.
        index: Index of the entry to read.
        max_masks: Maximum number of masks to consider for this entry (-1 = all).
        max_res: Maximum resolution to resize image and masks.

    Returns:
        image: Resized RGB image.
        masks: Array of binary masks.
        points: Array of point prompts (centroids of masks).
        labels: Array of ones (positive labels for prompts).
    """
    ent = data_dict[index]
    img = cv2.imread(ent["image"])[..., ::-1]  # BGR -> RGB
    ann_map = cv2.imread(ent["masks"], cv2.IMREAD_GRAYSCALE)

    # Rescale image and mask to fit max_res
    r = np.min([max_res / img.shape[1], max_res / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

    ann_map = 255 - (ann_map > 127).astype(np.uint8) * 255  # Binarize mask

    masks, points = [], []
    
    # Find contours in the inverted mask (background=0)
    contours, _ = cv2.findContours(ann_map.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        if (i == max_masks): break
        if len(contour) >= 3:
            mask = np.zeros_like(ann_map)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            
            # Compute centroid for point prompt
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if (cx != 0 and cy != 0):
                    points.append([[cx, cy]])
                    masks.append(mask)

    return img, np.array(masks), np.array(points), np.ones([len(masks), 1])

def visualize_entry(img, masks, points):
    """
    Visualize an image with its masks and point prompts.
    """
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask)

    # Display combined inverted mask
    plt.figure(figsize=(8, 8))
    plt.imshow(combined_mask + 1, cmap='gray')
    plt.axis("on")
    plt.title("Inverted Combined Mask")
    try:
        plt.get_current_fig_manager().full_screen_toggle()
    except:
        pass
    plt.tight_layout()
    plt.show()

    # Display image with point prompts
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    for point in points:
        plt.plot(point[0][0], point[0][1], 'ro', markersize=7)
    plt.axis("on")
    plt.title("Image with Points")
    try:
        plt.get_current_fig_manager().full_screen_toggle()
    except:
        pass
    plt.tight_layout()
    plt.show()

def process_batch(predictor, image, masks, input_point, input_label, device="cuda"):
    """
    Run a forward pass on the SAM 2 model for a single image batch.
    
    Args:
        predictor: SAM2ImagePredictor instance.
        image: Input image (RGB).
        masks: Ground-truth masks.
        input_point: Point prompts (centroids).
        input_label: Labels for point prompts.
        device: Torch device to use.

    Returns:
        prd_mask: Predicted masks.
        prd_scores: Confidence scores for predicted masks.
        gt_mask: Ground-truth masks tensor.
    """
    if masks.shape[0] == 0:
        return None, None, None

    predictor.set_image(image)
    
    mask_input, unnorm_coords, labels, _ = predictor._prep_prompts(
        input_point, input_label, box=None, mask_logits=None, normalize_coords=True
    )

    sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
        points=(unnorm_coords, labels), boxes=None, masks=None
    )

    batched_mode = unnorm_coords.shape[0] > 1
    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=True,
        repeat_image=batched_mode,
        high_res_features=high_res_features
    )

    prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
    prd_mask = torch.sigmoid(prd_masks[:, 0].to(dtype=torch.float16))

    gt_mask = torch.tensor((masks / 255).astype(np.float16), device=device)

    return prd_mask, prd_scores, gt_mask

def compute_dice_loss(prd_mask, prd_scores, gt_mask, score_weight):
    """
    Compute Dice loss and score loss for mask prediction.
    
    Args:
        prd_mask: Predicted masks tensor.
        prd_scores: Predicted confidence scores.
        gt_mask: Ground-truth masks tensor.
        score_weight: Weight for score loss component.

    Returns:
        dice_loss: Dice loss component.
        total_loss: Combined Dice + score loss.
    """
    smooth = 1e-6
    pred = prd_mask.view(-1)
    target = gt_mask.view(-1)
    intersection = (pred * target).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    inter = (gt_mask * (prd_mask > 0.5)).sum(dim=[1, 2])
    union = gt_mask.sum(dim=[1, 2]) + (prd_mask > 0.5).sum(dim=[1, 2]) - inter
    iou = inter / (union + 1e-6)

    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

    return dice_loss, dice_loss + score_loss * score_weight

def train_sam2(
    images_path,
    masks_path,
    epochs,
    grad_steps,
    log_dir,
    predictor,
    optimizer,
    scheduler,
    seed,
    train_percentage,
    score_weight,
    config_file,
    ckpt_path,
    points_per_side,
    points_per_batch,
    pred_iou_thresh,
    stability_score_thresh,
    stability_score_offset,
    mask_threshold,
):
    """
    Main training loop for SAM 2 model on mosaic dataset.
    """
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    scaler = torch.cuda.amp.GradScaler()  # Mixed precision

    # Split dataset into training and validation
    file_names = [f.split(".")[0] for f in os.listdir(images_path) if f.endswith(".jpg")]
    train_size = int(train_percentage * len(file_names))
    train_files = file_names[:train_size]
    val_files = file_names[train_size:]

    train_data = read_dataset(images_path, masks_path, train_files)

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        train_dice, train_loss = 0, 0
        random.shuffle(train_data)

        print(f"\nEpoch {epoch+1}/{epochs}")
        writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)

        # Training loop
        for i in tqdm(range(train_size), desc="Training Progress"):
            with torch.cuda.amp.autocast():
                image, masks, input_point, input_label = read_batch(train_data, i)
                prd_mask, prd_scores, gt_mask = process_batch(predictor, image, masks, input_point, input_label)

                dice, loss = compute_dice_loss(prd_mask, prd_scores, gt_mask, score_weight)
                scaler.scale(loss).backward()

                if (i % grad_steps == 0):
                    scaler.step(optimizer)
                    scaler.update()
                    predictor.model.zero_grad()

                train_dice += dice.mean().item()
                train_loss += loss.item()

        train_mean_dice = train_dice / train_size
        train_mean_loss = train_loss / train_size

        torch.save(predictor.model.state_dict(), f"model_{epoch}.torch")
        writer.add_text("Checkpoint", f"Epoch {epoch+1}: New model saved", epoch)

        predictor.model.eval()

        # Rebuild SAM2 model for automatic mask generation
        sam2_model = build_sam2(
            config_file=config_file,
            ckpt_path=ckpt_path,
            device="cuda",
            apply_postprocessing=False
        )

        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            mask_threshold=mask_threshold,
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
            load_model=f"model_{epoch}.torch"
        )

        # Validate only on the validation split
        val_output_path = "validation_results_temp"
        test_generator(
            mask_generator=mask_generator,
            images_path=images_path,
            output_path=val_output_path,
            file_list=val_files,
            max_mask_crop_region=0.1,
            show_masks=False
        )

        val_results = compute_metrics_batch(masks_path, val_output_path).round(4)
        print(val_results)

        if os.path.exists(val_output_path):
            shutil.rmtree(val_output_path)

        # Log metrics
        writer.add_scalar("Loss/Train", train_mean_loss, epoch)
        writer.add_scalar("Dice/Train", train_mean_dice, epoch)
        for metric_name, metric_value in val_results.items():
            writer.add_scalar(f"Validation/{metric_name}", metric_value, epoch)

        print(f"Epoch {epoch+1}: Train Dice = {train_mean_dice:.4f}, Train Loss = {train_mean_loss:.4f}")
        predictor.model.train()
        scheduler.step()

    writer.close()
