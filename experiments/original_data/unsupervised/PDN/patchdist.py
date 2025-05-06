# -*- coding: utf-8 -*-
"""Patch_Distillation.ipynb
 - A lightweight 4-layer Patch Description Network (PDN) with a 33×33 receptive field.
 - A deep pretrained teacher network (WideResNet-101) whose features are distilled into the PDN.
"""

# -----------------------
# Imports
# -----------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import csv
from datetime import datetime
import time
import seaborn as sns


# -----------------------
# PDN (Student) Definition
# -----------------------
# The PDN is a lightweight, fully convolutional network with 4 layers.
# With kernel_size=9 and stride=1 (with padding=4) in each layer,
# the receptive field of the final output is 9 + 3*(9-1) = 33.
class PDN(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(PDN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=9, stride=1, padding=4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=9, stride=1, padding=4)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4)
        # No activation after the final layer so that raw features are produced

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)
        return x


# -----------------------
# Teacher Feature Extractor Definition
# -----------------------
# We use a pretrained WideResNet-101 as the teacher.
# The final classifier is removed so that we obtain a feature map.
# Because the teacher network is deep, its spatial output is lower resolution.
# We add a 1x1 projection layer (to reduce the channel dimension to match the PDN)
# and upsample the result to the original image resolution.
class TeacherFeatureExtractor(nn.Module):
    def __init__(self, out_channels=64):
        super(TeacherFeatureExtractor, self).__init__()
        # Load a pretrained WideResNet-101
        teacher = models.wide_resnet101_2(pretrained=True)
        # Remove the classification head.
        # Here we take all layers up to (but not including) the average pooling.
        self.features = nn.Sequential(*list(teacher.children())[:-2])
        # The teacher's last convolution produces 2048 channels.
        self.proj = nn.Conv2d(2048, out_channels, kernel_size=1)
        # Freeze teacher parameters
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.proj.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x is expected to be a 3-channel image.
        feat = self.features(x)  # e.g. shape: [B, 2048, H_t, W_t]
        feat = self.proj(feat)  # shape becomes [B, out_channels, H_t, W_t]
        # Upsample to match the input resolution (which is also the PDN's output resolution)
        feat = nn.functional.interpolate(feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        return feat


# -----------------------
# Custom Dataset Definition (Remains Similar)
# -----------------------
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, return_filename=False):
        self.root_dir = root_dir
        self.transform = transform
        self.return_filename = return_filename
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img_path = os.path.join(self.root_dir, fname)
        # Load as grayscale for the PDN (1 channel)
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        if self.return_filename:
            return image, fname
        else:
            return image


# -----------------------
# Transforms & Device Setup
# -----------------------
# We use a resize transform (e.g. to 128x128 or 224x224).
# For distillation with the teacher network, a size like 224x224 is typical.
img_size = 224  # you can adjust this as needed
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Data Loading
# -----------------------
# Training dataset: assumed to be all normal images (for distillation)
train_data_path = "/mnt/anom_proj/data/New/train"  # update path if needed
train_dataset = CustomImageDataset(root_dir=train_data_path, transform=transform, return_filename=False)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Validation and Test datasets (with filenames for later anomaly detection)
val_data_path = "/mnt/anom_proj/data/New/validation"  # update path if needed
val_dataset = CustomImageDataset(root_dir=val_data_path, transform=transform, return_filename=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_data_path = "/mnt/anom_proj/data/New/test"  # update path if needed
test_dataset = CustomImageDataset(root_dir=test_data_path, transform=transform, return_filename=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -----------------------
# Model Instantiation
# -----------------------
# Instantiate the PDN (student) and the teacher feature extractor.
student = PDN(in_channels=1, out_channels=64).to(device)
teacher = TeacherFeatureExtractor(out_channels=64).to(device)

# Set teacher to evaluation mode (its parameters are frozen)
teacher.eval()

model = PDN().to(device)
model_name = model.__class__.__name__

print(f"Training {model_name} (distilling from WideResNet-101) on grayscale data...")

# -----------------------
# Distillation Training Setup
# -----------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(student.parameters(), lr=0.001)

train_losses = []
val_losses = []

best_val_loss = float('inf')
patience = 3
counter = 0

epochs = 30  # Adjust the number of distillation epochs as needed

train_start = time.time()

# -----------------------
# Training Loop (Distillation Phase)
# -----------------------
for epoch in range(epochs):
    student.train()
    total_train_loss = 0
    for images in train_dataloader:
        images = images.to(device)  # shape: [B,1,H,W]
        optimizer.zero_grad()
        # Student forward pass on grayscale input
        student_features = student(images)
        # For teacher, replicate grayscale to 3 channels
        images_3ch = images.repeat(1, 3, 1, 1)  # shape: [B,3,H,W]
        with torch.no_grad():
            teacher_features = teacher(images_3ch)
        loss = criterion(student_features, teacher_features)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_dataloader)

    # Validation (using training data as proxy if no separate distillation val set is available)
    student.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images in train_dataloader:
            images = images.to(device)
            student_features = student(images)
            images_3ch = images.repeat(1, 3, 1, 1)
            teacher_features = teacher(images_3ch)
            loss = criterion(student_features, teacher_features)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(train_dataloader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(student.state_dict(), "best_student.pth")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

train_end = time.time()
train_duration = train_end - train_start
print(f"\nDistillation training completed in {train_duration:.2f} seconds ({train_duration / 60:.2f} minutes)")

# -----------------------
# Loss Curves (Distillation Phase)
# -----------------------
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(12, 6))
epochs_run = len(train_losses)
plt.plot(range(1, epochs_run + 1), train_losses, marker='o', label="Training Loss")
plt.plot(range(1, epochs_run + 1), val_losses, marker='o', label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title(f"{model_name} Distillation Loss")
plt.title(f"{model.__class__.__name__} Distillation Loss over" f" {epochs_run} epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"results/plots/{model.__class__.__name__}_distillation_loss_curve.png")

plt.xlim(0, epochs_run + 1)
ymin = min(min(train_losses), min(val_losses)) - 0.1
ymax = max(max(train_losses), max(val_losses)) + 0.1
plt.ylim(ymin, ymax)

plt.show()

# -----------------------
# Threshold Calibration for Anomaly Detection
# -----------------------
# On normal training images, compute the distillation (MSE) error per image.
student.eval()
train_errors = []
with torch.no_grad():
    for images in train_dataloader:
        images = images.to(device)
        student_features = student(images)
        images_3ch = images.repeat(1, 3, 1, 1)
        teacher_features = teacher(images_3ch)
        # Compute per-image MSE (averaging over spatial dimensions and channels)
        errors = torch.mean((student_features - teacher_features) ** 2, dim=[1, 2, 3])
        train_errors.extend(errors.cpu().numpy())

train_errors = np.array(train_errors)
threshold = train_errors.mean() + 3 * train_errors.std()
print(f"Calibrated threshold based on training data: {threshold:.6f}")


# -----------------------
# Anomaly Detection Function (Distillation Error)
# -----------------------
def detect_anomalies_with_filenames(student, teacher, dataloader, threshold):
    results = []  # list of tuples: (image, filename, anomaly_flag)
    student.eval()
    with torch.no_grad():
        for batch in dataloader:
            images, filenames = batch
            images = images.to(device)
            student_features = student(images)
            images_3ch = images.repeat(1, 3, 1, 1)
            teacher_features = teacher(images_3ch)
            loss = torch.mean((student_features - teacher_features) ** 2, dim=[1, 2, 3])
            anomaly_flags = loss > threshold
            for img, fname, flag in zip(images.cpu(), filenames, anomaly_flags.cpu().numpy()):
                results.append((img, fname, flag))
    return results


# -----------------------
# Compute Anomaly Results on Validation and Test Sets
# -----------------------
val_results = detect_anomalies_with_filenames(student, teacher, val_dataloader, threshold)

inference_start = time.time()
test_results = detect_anomalies_with_filenames(student, teacher, test_dataloader, threshold)
inference_end = time.time()
inference_duration = inference_end - inference_start
print(f"\n Inference on test set completed in {inference_duration:.2f} seconds ({inference_duration / 60:.2f} minutes)")

val_anomaly_count = sum(1 for _, _, flag in val_results if flag)
test_anomaly_count = sum(1 for _, _, flag in test_results if flag)
print("Validation anomalies detected:", val_anomaly_count, "out of", len(val_dataset))
print("Test anomalies detected:", test_anomaly_count, "out of", len(test_dataset))

# -----------------------
# (Optional) Evaluation with Ground Truth Labels
# -----------------------
labels_csv_path = "/mnt/anom_proj/data/New/labels.csv"  # update if needed
labels_df = pd.read_csv(labels_csv_path)
label_dict = {row['filename']: row['label'] for _, row in labels_df.iterrows() if row['split'] == 'test'}

y_true = []
y_pred = []
for _, fname, is_anomaly in test_results:
    if fname in label_dict:
        y_true.append(label_dict[fname])
        y_pred.append(int(is_anomaly))

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\nEvaluation Metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")

cm = confusion_matrix(y_true, y_pred)


# -----------------------
# Random Sampling and Visualization (Optional)
# -----------------------
def plot_images(samples, title, save_folder="data_inspection", save_filename=None):
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(15, 3))
    for i, (img, fname, flag) in enumerate(samples):
        plt.subplot(1, len(samples), i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(fname, fontsize=8)
        plt.axis('off')
    plt.suptitle(title)
    if save_filename is None:
        save_filename = title.replace(" ", "_") + ".png"
    save_path = os.path.join(save_folder, save_filename)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.show()


def pick_random_samples(results, num_samples=5):
    anomalies = [res for res in results if res[2]]
    normals = [res for res in results if not res[2]]
    selected_anomalies = random.sample(anomalies, min(num_samples, len(anomalies))) if anomalies else []
    selected_normals = random.sample(normals, min(num_samples, len(normals))) if normals else []
    return selected_anomalies, selected_normals


val_anomalies, val_normals = pick_random_samples(val_results, num_samples=5)
plot_images(val_anomalies, f"{model.__class__.__name__} Validation Anomalies", save_filename=f"{model.__class__.__name__}_validation_anomalies.png")
plot_images(val_normals, f"{model.__class__.__name__} Validation Normals", save_filename=f"{model.__class__.__name__}_validation_normals.png")

# Randomly pick samples from test set and plot/save them
test_anomalies, test_normals = pick_random_samples(test_results, num_samples=5)
plot_images(test_anomalies, f"{model.__class__.__name__} Test Anomalies", save_filename=f"{model.__class__.__name__}_test_anomalies.png")
plot_images(test_normals, f"{model.__class__.__name__} Test Normals", save_filename=f"{model.__class__.__name__}_test_normals.png")


def save_results_to_csv(results, label_dict=None, split_name="test", output_path="results"):
    os.makedirs(output_path, exist_ok=True)
    filename = os.path.join(output_path, f"{split_name}_anomaly_results.csv")

    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename", "predicted_anomaly", "actual_label"]
        writer.writerow(header)

        for _, fname, is_anomaly in results:
            actual_label = label_dict[fname] if label_dict and fname in label_dict else "NA"
            writer.writerow([fname, int(is_anomaly), actual_label])

    print(f"Saved {split_name} results to {filename}")


save_results_to_csv(val_results, label_dict=None, split_name="validation")  # no labels for val
save_results_to_csv(test_results, label_dict=label_dict, split_name="test")  # uses labels


# -----------------------
# (Optional) Experiment Logging
# -----------------------


def log_experiment_run(model, criterion, optimizer, train_dataloader, val_dataloader, test_dataloader,
                       epoch, patience, train_duration, inference_duration,
                       threshold, acc, prec, rec, f1, cm,
                       save_model=True, save_paths=True, log_csv=True):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = model.__class__.__name__
    version_prefix = f"{model_name}_{timestamp}"
    if save_model:
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{version_prefix}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

        file_size_bytes = os.path.getsize(model_path)
        model_file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    else:
        model_path = "not_saved"
        model_file_size_mb = "not_saved"

    if save_paths:
        os.makedirs("results", exist_ok=True)
        cm_path = f"results/plots/confusion_matrix_{version_prefix}.png"
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"])
        plt.title(f"{model.__class__.__name__} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        print(f"Confusion matrix saved to {cm_path}")
    else:
        cm_path = "not_saved"
    model_parameters = sum(p.numel() for p in model.parameters())
    experiment_settings = {
        "version_id": version_prefix,
        "timestamp": timestamp,
        "model": model_name,
        "loss_function": criterion.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "batch_size": train_dataloader.batch_size,
        "epochs_run": epoch + 1,
        "patience": patience,
        "train_time_sec": round(train_duration, 2),
        "test_inference_time_sec": round(inference_duration, 2),
        "train_size": len(train_dataloader.dataset),
        "val_size": len(val_dataloader.dataset),
        "test_size": len(test_dataloader.dataset),
        "image_size": f"{img_size}x{img_size}",
        "threshold": round(threshold, 6),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "true_negatives": cm[0][0],
        "false_positives": cm[0][1],
        "false_negatives": cm[1][0],
        "true_positives": cm[1][1],
        "model_file": model_path,
        "confusion_matrix_path": cm_path,
        "model_parameters": model_parameters,
        "model_file_size_mb": model_file_size_mb
    }
    if log_csv:
        os.makedirs("logs", exist_ok=True)
        csv_path = "logs/experiment_log.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=experiment_settings.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(experiment_settings)
        print(f"Experiment logged to {csv_path}")
    return version_prefix


log_experiment_run(
    model=student,
    criterion=criterion,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    epoch=epoch,
    patience=patience,
    train_duration=train_duration,
    inference_duration=inference_duration,
    threshold=threshold,
    acc=acc,
    prec=prec,
    rec=rec,
    f1=f1,
    cm=cm,
    save_paths=True,
)
