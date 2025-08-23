import numpy as np
import time
from tqdm import tqdm
import cv2
import hydra
import matplotlib.pyplot as plt
import os
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import pandas as pd

# Function to visualize annotations (masks) on an image
def show_anns(anns, borders=True):
    if len(anns) == 0:  # Nothing to show
        return

    # Sort annotations by area (largest first)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # Create an RGBA image to overlay masks
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0  # Start fully transparent

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])  # Random color with 0.5 alpha
        img[m] = color_mask  # Apply mask color

        # Optionally draw contours (borders)
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Approximate contours for smoother borders
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=2) 

    ax.imshow(img)  # Display the overlay

# Function to generate masks for a dataset of images
def test_generator(
    mask_generator,
    images_path,
    output_path,
    crops_csv_file=None,
    max_mask_crop_region=0.1,
    overlap_ratio=0.1,
    show_masks=False,
    file_list=None
):
    # If a file list is provided, use it instead of reading CSV/folder
    if file_list is not None:
        df = pd.DataFrame({
            "file_name": file_list,
            "rows": [1] * len(file_list),
            "cols": [1] * len(file_list)
        })
    elif crops_csv_file is None:
        image_files = [f.split(".")[0] for f in os.listdir(images_path) if f.lower().endswith((".jpg", ".png"))]
        df = pd.DataFrame({
            "file_name": image_files,
            "rows": [1] * len(image_files),
            "cols": [1] * len(image_files)
        })
    else:
        df = pd.read_csv(crops_csv_file)

    file_names = df["file_name"]
    crop_rows = df["rows"]
    crop_cols = df["cols"]

    zipped = list(zip(file_names, crop_rows, crop_cols))

    for file_name, rows, cols in tqdm(zipped, desc="Processing", total=len(zipped)):
        img_path = os.path.join(images_path, f"{file_name}.jpg")
        pred_path = os.path.join(output_path, f"{file_name}.png")

        image = Image.open(img_path)
        width, height = image.size

        final_mask = np.zeros((height, width, 3), dtype=np.uint8)

        crop_regions = []
        cell_width, cell_height = width // cols, height // rows
        overlap_width = int(cell_width * overlap_ratio)
        overlap_height = int(cell_height * overlap_ratio)

        for i in range(rows):
            for j in range(cols):
                left = max(j * cell_width - overlap_width, 0)
                upper = max(i * cell_height - overlap_height, 0)
                right = min((j + 1) * cell_width + overlap_width, width)
                lower = min((i + 1) * cell_height + overlap_height, height)
                crop_regions.append((left, upper, right, lower))

        for crop in crop_regions:
            cropped_image = image.crop(crop)
            cropped_image_np = np.array(cropped_image.convert("RGB"))

            try:
                masks = mask_generator.generate(cropped_image_np)
            except IndexError:
                continue

            if show_masks:
                plt.figure(figsize=(12, 12))
                plt.imshow(cropped_image_np)
                show_anns(masks)
                plt.axis('off')
                plt.show()

            mask_overlay = np.zeros_like(cropped_image_np, dtype=np.uint8)
            max_area_threshold = (crop[2] - crop[0]) * (crop[3] - crop[1]) * max_mask_crop_region

            for mask in masks:
                mask_area = np.sum(mask['segmentation'])
                if mask_area < max_area_threshold:
                    mask_overlay[mask['segmentation']] = (255, 255, 255)

            x1, y1, x2, y2 = crop
            final_mask[y1:y2, x1:x2] = np.maximum(final_mask[y1:y2, x1:x2], mask_overlay)

        final_mask_pil = Image.fromarray(final_mask)
        os.makedirs(output_path, exist_ok=True)
        final_mask_pil.save(pred_path)
