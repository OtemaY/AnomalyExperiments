# This script splits the data for supervised learning

import os
import shutil
import random
import csv
from sklearn.model_selection import train_test_split

# Set actual paths
train_dir = "/mnt/anom_proj/data/original data/train"
val_dir = "/mnt/anom_proj/data/original data/validation"
output_base = "/mnt/anom_proj/data/original data/"

# Output directories
new_train_dir = os.path.join(output_base, "super_train")
new_val_dir = os.path.join(output_base, "super_val")

# Create new folders
os.makedirs(new_train_dir, exist_ok=True)
os.makedirs(new_val_dir, exist_ok=True)

# Assign label based on filename prefix
def label_from_filename(filename):
    if filename.startswith("neg_"):
        return 0
    elif filename.startswith("pos_"):
        return 1
    else:
        raise ValueError(f"Unknown label prefix in filename: {filename}")

# Collect all images with labels
all_images = []

# Add from train (all normal)
for fname in os.listdir(train_dir):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        label = label_from_filename(fname)  # will always be 0 for train
        all_images.append((os.path.join(train_dir, fname), fname, label))

# Add from validation (mixed)
for fname in os.listdir(val_dir):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        label = label_from_filename(fname)
        all_images.append((os.path.join(val_dir, fname), fname, label))

# Shuffle and split
random.seed(42)
random.shuffle(all_images)

file_paths = [i[0] for i in all_images]
filenames = [i[1] for i in all_images]
labels = [i[2] for i in all_images]

train_paths, val_paths, train_fnames, val_fnames, train_labels, val_labels = train_test_split(
    file_paths, filenames, labels, test_size=0.2, stratify=labels, random_state=42
)


test_dir = os.path.join(output_base, "test")


# Prepare CSV rows
csv_rows = []

def copy_and_log(paths, fnames, labels, split_name, target_dir):
    for src, fname, label in zip(paths, fnames, labels):
        dst = os.path.join(target_dir, fname)
        shutil.copy(src, dst)
        csv_rows.append([split_name, fname, label])

# Copy and log train/val
copy_and_log(train_paths, train_fnames, train_labels, "train", new_train_dir)
copy_and_log(val_paths, val_fnames, val_labels, "val", new_val_dir)

# Add test images to CSV (do not copy)
for fname in os.listdir(test_dir):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        try:
            label = label_from_filename(fname)
            csv_rows.append(["test", fname, label])
        except ValueError as e:
            print(f"Skipping file: {fname} — {e}")

# Write final CSV
csv_path = os.path.join(output_base, "super_labels.csv")
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "filename", "label"])
    writer.writerows(csv_rows)

print("✅ All done! original data splits created and test set added to CSV.")
print(f"📄 CSV saved to: {csv_path}")