# This file is used to the

import os
import shutil
import random
import csv
from pathlib import Path

def label_from_filename(filename):
    """
    Assign a label based on filename prefix.
    Filenames starting with 'neg' => 0, 'pos' => 1.
    """
    name = Path(filename).name.lower()
    if name.startswith('neg'):
        return 0
    elif name.startswith('pos'):
        return 1
    else:
        raise ValueError(f"Cannot determine label for file: {filename}")


def split_and_label_images(folder1, folder2, output_train, output_val,
                           labels_csv, split_ratio=0.8,
                           test_folder=None):
    """
    Combine images from two source folders, shuffle them, split into train/validation,
    optionally include unseen test data, and write a CSV of split,filename,label.
    """
    # Collect all image paths from source folders
    images = []
    for folder in (folder1, folder2):
        folder_path = Path(folder)
        if not folder_path.is_dir():
            raise ValueError(f"Source folder does not exist: {folder}")
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            images.extend(folder_path.glob(ext))

    # Shuffle the list
    random.shuffle(images)

    # Determine split sizes
    n_train = int(len(images) * split_ratio)
    train_images = images[:n_train]
    val_images = images[n_train:]

    # Create output directories
    Path(output_train).mkdir(parents=True, exist_ok=True)
    Path(output_val).mkdir(parents=True, exist_ok=True)

    # Copy train/val files
    for img in train_images:
        shutil.copy(img, Path(output_train) / img.name)
    for img in val_images:
        shutil.copy(img, Path(output_val) / img.name)

    # Collect test images if provided
    test_images = []
    if test_folder:
        test_path = Path(test_folder)
        if not test_path.is_dir():
            raise ValueError(f"Test folder does not exist: {test_folder}")
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            test_images.extend(test_path.glob(ext))

    # Write labels CSV
    csv_path = Path(labels_csv)
    with csv_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'filename', 'label'])
        # Train
        for img in train_images:
            writer.writerow(['train', img.name, label_from_filename(img.name)])
        # Validation
        for img in val_images:
            writer.writerow(['validation', img.name, label_from_filename(img.name)])
        # Test
        for img in test_images:
            writer.writerow(['test', img.name, label_from_filename(img.name)])

    print(f"Images split and labels written to {csv_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Shuffle images, split into train/val, include optional test set, and generate labels CSV."
    )
    parser.add_argument("--folder1", required=True, help="Path to the first source image folder.")
    parser.add_argument("--folder2", required=True, help="Path to the second source image folder.")
    parser.add_argument("--output_train", default="train", help="Output folder for training split.")
    parser.add_argument("--output_val", default="val", help="Output folder for validation split.")
    parser.add_argument("--labels_csv", default="labels.csv", help="Path for the labels CSV output.")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="Fraction of images to allocate to training (0-1).")
    parser.add_argument("--test_folder", default=None,
                        help="Optional path to an unseen test image folder to include in CSV.")

    args = parser.parse_args()
    split_and_label_images(
        args.folder1, args.folder2,
        args.output_train, args.output_val,
        args.labels_csv, args.split_ratio,
        args.test_folder
    )

if __name__ == "__main__":
    main()
