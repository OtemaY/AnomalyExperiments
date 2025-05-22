
import os
import random
import numpy as np
import torch
from PIL import Image, ImageStat
import torchvision.transforms as T
import kornia.augmentation as K

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
#src_dir = "/Users/oteyi561/Downloads/Thesis/concretedata/gen_mixed"
src_dir = "/mnt/anom_proj/data/New/super_train"

#dst_dir = "/Users/oteyi561/Downloads/Thesis/concretedata/superaugnormal"
dst_dir = "/mnt/anom_proj/data/New/superaugnormal"
pad_amount = 32  # padding before augmentation
# Overshoot ratio: % increase over matching majority class
overshoot_ratio = 0.8  # e.g. 0.1 = 10% more images

# -----------------------------------------------------------------------------
# Symmetric (mirror) padding for PIL images
# -----------------------------------------------------------------------------
def symmetric_pad_pil(img, pad):
    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[..., None]
    arr = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    if arr.shape[2] == 1:
        arr = arr.squeeze(2)
    return Image.fromarray(arr.astype(np.uint8))

# -----------------------------------------------------------------------------
# Brightness jitter in PIL
# -----------------------------------------------------------------------------
class AdaptiveBrightnessJitter:
    def __init__(self, min_rel=0.8, max_rel=1.2):
        self.min_rel, self.max_rel = min_rel, max_rel
    def __call__(self, img):
        stat = ImageStat.Stat(img)
        mean = sum(stat.mean) / len(stat.mean)
        factor = self.min_rel + (self.max_rel - self.min_rel) * (mean / 255.0)
        return T.functional.adjust_brightness(img, factor)

# -----------------------------------------------------------------------------
# Augmentation sequence (no resizing)
# -----------------------------------------------------------------------------
def get_augment_seq():
    return torch.nn.Sequential(
        K.RandomAffine(degrees=20,
                       translate=(0.15, 0.15),
                       scale=(0.8, 1.2),
                       padding_mode='reflection',
                       p=1.0),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5)
    )

# -----------------------------------------------------------------------------
# Main processing: compute dynamic targets, then augment
# -----------------------------------------------------------------------------
def augment_and_save(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    anomalies = [f for f in files if f.lower().startswith('pos')]
    normals = [f for f in files if not f.lower().startswith('pos')]

    num_norm = len(normals)
    num_anom = len(anomalies)
    total_real = num_norm + num_anom

    # Compute base target = max of class counts
    base_target = max(num_norm, num_anom)
    # Apply overshoot to increase dataset
    target_per_class = int(np.ceil(base_target * (1 + overshoot_ratio)))

    # Synthetic counts per class
    synth_norm_count = max(target_per_class - num_norm, 0)
    synth_anom_count = max(target_per_class - num_anom, 0)

    # Sample with replacement
    synth_norms = random.choices(normals, k=synth_norm_count) if synth_norm_count > 0 else []
    synth_anoms = random.choices(anomalies, k=synth_anom_count) if synth_anom_count > 0 else []

    # 1) Save originals unchanged
    for fname in files:
        img = Image.open(os.path.join(src_dir, fname)).convert('RGB')
        img.save(os.path.join(dst_dir, fname))

    # 2) Generate synthetic normals
    for idx, fname in enumerate(synth_norms, start=1):
        img = Image.open(os.path.join(src_dir, fname)).convert('RGB')
        w, h = img.size
        img_pad = symmetric_pad_pil(img, pad_amount)
        tensor = T.functional.pil_to_tensor(img_pad).float() / 255.0
        tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            aug = get_augment_seq()(tensor)
        aug = aug.squeeze(0).clamp(0, 1)
        pil_aug = T.functional.to_pil_image(aug)
        pil_aug = AdaptiveBrightnessJitter()(pil_aug)
        pil_aug = pil_aug.resize((w, h), Image.BILINEAR)
        base, ext = os.path.splitext(fname)
        pil_aug.save(os.path.join(dst_dir, f"{base}_synthnorm{idx}{ext}"))

    # 3) Generate synthetic anomalies
    for idx, fname in enumerate(synth_anoms, start=1):
        img = Image.open(os.path.join(src_dir, fname)).convert('RGB')
        w, h = img.size
        img_pad = symmetric_pad_pil(img, pad_amount)
        tensor = T.functional.pil_to_tensor(img_pad).float() / 255.0
        tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            aug = get_augment_seq()(tensor)
        aug = aug.squeeze(0).clamp(0, 1)
        pil_aug = T.functional.to_pil_image(aug)
        pil_aug = AdaptiveBrightnessJitter()(pil_aug)
        pil_aug = pil_aug.resize((w, h), Image.BILINEAR)
        base, ext = os.path.splitext(fname)
        pil_aug.save(os.path.join(dst_dir, f"{base}_synthanom{idx}{ext}"))

    total_synth = synth_norm_count + synth_anom_count
    print(f"Saved {total_real} originals and {total_synth} synthetic images; ~{total_real + total_synth} total.")

# -----------------------------------------------------------------------------
# CSV generation
# -----------------------------------------------------------------------------
def create_csv(dst_dir, csv_path):
    import csv
    files = [f for f in os.listdir(dst_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'filename', 'label'])
        for fn in files:
            label = 1 if fn.lower().startswith('pos') or '_synthanom' in fn else 0
            writer.writerow(['train', fn, label])

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    augment_and_save(src_dir, dst_dir)
    create_csv(dst_dir, os.path.join(dst_dir, 'super_augment.csv'))
    print('CSV saved.')
