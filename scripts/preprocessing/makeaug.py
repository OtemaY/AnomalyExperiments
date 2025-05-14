import os
import numpy as np
import torch
from PIL import Image, ImageStat
import torchvision.transforms as T
import kornia.augmentation as K

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
#src_dir = "/Users/oteyi561/Downloads/Thesis/concretedata/gen_normal"
src_dir = "/mnt/anom_proj/data/New/train"

#dst_dir = "/Users/oteyi561/Downloads/Thesis/concretedata/augnormalssym"
dst_dir = "/mnt/anom_proj/data/New/augnormalssym"

num_aug_per_image = 3
output_size = 256
pad_amount = 32  # Increase if needed

# -----------------------------------------------------------------------------
# Symmetric (mirror) padding function for PIL images
# -----------------------------------------------------------------------------
def symmetric_pad_pil(img, pad):
    """Pad a PIL image symmetrically (mirror boundary condition)."""
    arr = np.array(img)
    if arr.ndim == 2:  # Grayscale
        arr = np.expand_dims(arr, axis=-1)
    arr = np.pad(
        arr,
        ((pad, pad), (pad, pad), (0, 0)),
        mode='reflect'
    )
    arr = arr.squeeze() if arr.shape[2] == 1 else arr
    return Image.fromarray(arr.astype(np.uint8))

# -----------------------------------------------------------------------------
# Adaptive brightness jitter augmentation (still in PIL)
# -----------------------------------------------------------------------------
class AdaptiveBrightnessJitter:
    def __init__(self, min_rel=0.8, max_rel=1.2):
        self.min_rel = min_rel
        self.max_rel = max_rel

    def __call__(self, img):
        stat = ImageStat.Stat(img)
        mean_per_channel = stat.mean
        avg_brightness = sum(mean_per_channel) / 3.0
        norm = avg_brightness / 255.0
        weight = 1.0 - norm
        factor = self.min_rel * (1 - weight) + self.max_rel * weight
        return T.functional.adjust_brightness(img, factor)

# -----------------------------------------------------------------------------
# Kornia augmentation pipeline (works on torch.Tensor)
# -----------------------------------------------------------------------------
kornia_augment = torch.nn.Sequential(
    K.RandomResizedCrop(
        size=(output_size, output_size),
        scale=(0.5, 1.0),
        ratio=(0.9, 1.1),
        p=1.0
    ),
    K.RandomAffine(
        degrees=20,
        translate=(0.15, 0.15),
        scale=(0.8, 1.2),
        padding_mode='reflection',  # This is supported in Kornia 0.8.0
        p=1.0
    ),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.CenterCrop((output_size, output_size))
)

# -----------------------------------------------------------------------------
# Function to process and save images
# -----------------------------------------------------------------------------
def augment_and_save(src_dir, dst_dir, num_aug_per_image):
    os.makedirs(dst_dir, exist_ok=True)

    for fname in os.listdir(src_dir):
        src_path = os.path.join(src_dir, fname)
        if not os.path.isfile(src_path) or not fname.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        ):
            continue

        img = Image.open(src_path).convert('RGB')
        img = symmetric_pad_pil(img, pad_amount)  # Apply symmetric (mirror) padding

        base_name, ext = os.path.splitext(fname)
        orig_save_path = os.path.join(dst_dir, f"{base_name}_orig{ext}")
        img.save(orig_save_path)

        for i in range(1, num_aug_per_image + 1):
            img_tensor = T.functional.pil_to_tensor(img).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            with torch.no_grad():
                aug_tensor = kornia_augment(img_tensor)
            aug_tensor = aug_tensor.squeeze(0).clamp(0, 1)
            aug_img = T.functional.to_pil_image(aug_tensor)
            aug_img = AdaptiveBrightnessJitter(min_rel=0.8, max_rel=1.2)(aug_img)
            aug_save_path = os.path.join(dst_dir, f"{base_name}_aug{i}{ext}")
            aug_img.save(aug_save_path)

    print(f"Saved original + {num_aug_per_image} augmentations for each image to '{dst_dir}'")

import csv

def create_csv(dst_dir, csv_path):
    # List all image files in the augmented directory
    image_files = [
        f for f in os.listdir(dst_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
    ]
    # Write to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['split', 'filename', 'label'])  # Header
        for fname in image_files:
            writer.writerow(['train', fname, 0])

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    augment_and_save(src_dir, dst_dir, num_aug_per_image)
    # Path for your CSV file
    csv_output_path = os.path.join(dst_dir, "augmented_unsupe.csv")
    create_csv(dst_dir, csv_output_path)
    print(f"CSV file saved to {csv_output_path}")
