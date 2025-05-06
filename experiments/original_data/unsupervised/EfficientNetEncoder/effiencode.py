# -*- coding: utf-8 -*-
"""EfficientAD.ipynb

Replicates the EfficientAD architecture with two branches:
  1. Distillation branch (PDN student distilled from WideResNet-101 teacher)
  2. Autoencoder branch capturing logical constraints.
At test time, the anomaly scores from both branches are fused.
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
import time
import csv
from datetime import datetime
import seaborn as sns


# -----------------------
# 1. Define the PDN (Student) for Distillation
# -----------------------
class PDN(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(PDN, self).__init__()
        # Four convolutional layers with kernel_size=9, stride=1, and padding=4 give a 33x33 receptive field.
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=9, stride=1, padding=4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=9, stride=1, padding=4)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4)
        # No activation after final layer

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)
        return x


# -----------------------
# 2. Define the Teacher Feature Extractor
# -----------------------
class TeacherFeatureExtractor(nn.Module):
    def __init__(self, out_channels=64):
        super(TeacherFeatureExtractor, self).__init__()
        # Load a pretrained WideResNet-101 (wide_resnet101_2 variant)
        teacher = models.wide_resnet101_2(pretrained=True)
        # Remove the classification head (take all layers except average pooling and FC)
        self.features = nn.Sequential(*list(teacher.children())[:-2])
        # Project the teacher's output (2048 channels) to match PDN (64 channels)
        self.proj = nn.Conv2d(2048, out_channels, kernel_size=1)
        # Freeze teacher parameters
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.proj.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x is expected to be a 3-channel image.
        feat = self.features(x)  # e.g., shape: [B, 2048, H_t, W_t]
        feat = self.proj(feat)  # shape: [B, 64, H_t, W_t]
        # Upsample to the input spatial size (to compare with PDN output)
        feat = nn.functional.interpolate(feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        return feat


# -----------------------
# 3. Define the Convolutional Autoencoder (Logical Anomaly Branch)
# -----------------------
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # e.g., 224 -> 112
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 112 -> 56
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 56 -> 28
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28 -> 56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 56 -> 112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 112 -> 224
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# -----------------------
# 4. Custom Dataset Definition
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
        # Load as grayscale for the PDN and autoencoder
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        if self.return_filename:
            return image, fname
        else:
            return image


# -----------------------
# 5. Transforms & Device Setup
# -----------------------
# For the teacher branch, a typical image size is 224x224.
img_size = 224
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# 6. Data Loading
# -----------------------
train_data_path = "/mnt/anom_proj/data/original data/train"  # Update path if needed
val_data_path = "/mnt/anom_proj/data/original data/validation"
test_data_path = "/mnt/anom_proj/data/original data/test"

train_dataset = CustomImageDataset(root_dir=train_data_path, transform=transform, return_filename=False)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = CustomImageDataset(root_dir=val_data_path, transform=transform, return_filename=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = CustomImageDataset(root_dir=test_data_path, transform=transform, return_filename=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -----------------------
# 7. Model Instantiation
# -----------------------
# Distillation branch components
student = PDN(in_channels=1, out_channels=64).to(device)
teacher = TeacherFeatureExtractor(out_channels=64).to(device)
teacher.eval()  # Teacher is frozen

# Autoencoder branch
autoencoder = ConvAutoencoder().to(device)

model_name = "EfficientAD_Fusion"

print(f"Training {model_name} with both distillation and autoencoder branches...")

# -----------------------
# 8. Optimizers and Loss Functions
# -----------------------
criterion = nn.MSELoss()

# Create separate optimizers for the student (distillation) and autoencoder branches
optimizer_student = optim.Adam(student.parameters(), lr=0.001)
optimizer_autoencoder = optim.Adam(autoencoder.parameters(), lr=0.001)

# -----------------------
# 9. Combined Training Loop
# -----------------------
epochs = 25  # Adjust as needed
train_losses_dist = []
train_losses_rec = []
patience = 3
counter = 0


train_start = time.time()

for epoch in range(epochs):
    student.train()
    autoencoder.train()
    total_loss_dist = 0.0
    total_loss_rec = 0.0

    for images in train_dataloader:
        images = images.to(device)  # Shape: [B, 1, H, W]

        # ----- Distillation Branch -----
        optimizer_student.zero_grad()
        # Student forward pass on grayscale images
        student_features = student(images)
        # Teacher expects 3-channel input, so replicate channels
        images_3ch = images.repeat(1, 3, 1, 1)
        with torch.no_grad():
            teacher_features = teacher(images_3ch)
        loss_dist = criterion(student_features, teacher_features)
        loss_dist.backward()
        optimizer_student.step()
        total_loss_dist += loss_dist.item()

        # ----- Autoencoder Branch -----
        optimizer_autoencoder.zero_grad()
        reconstructions = autoencoder(images)
        loss_rec = criterion(reconstructions, images)
        loss_rec.backward()
        optimizer_autoencoder.step()
        total_loss_rec += loss_rec.item()

    avg_loss_dist = total_loss_dist / len(train_dataloader)
    avg_loss_rec = total_loss_rec / len(train_dataloader)
    train_losses_dist.append(avg_loss_dist)
    train_losses_rec.append(avg_loss_rec)

    print(f"Epoch {epoch + 1}/{epochs} | Distillation Loss: {avg_loss_dist:.6f} | Reconstruction Loss: {avg_loss_rec:.6f}")

    # (Optional) Early stopping based on validation loss could be implemented separately for each branch.
    # Here, we simply print the training losses.

train_end = time.time()
train_duration = train_end - train_start
print(f"\nTraining completed in {train_duration:.2f} seconds ({train_duration / 60:.2f} minutes)")

# -----------------------
# 10. Plot Training Losses
# -----------------------
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(train_losses_dist, label="Distillation Loss")
plt.plot(train_losses_rec, label="Reconstruction Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{model_name} Training Losses")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/plots/training_loss_fusion.png")
plt.show()

# -----------------------
# 11. Threshold Calibration (Fusion of Anomaly Scores)
# -----------------------
# For each training image, compute:
#   - distillation_error: MSE between student and teacher features
#   - reconstruction_error: MSE between autoencoder reconstruction and input
# Then fuse: fused_score = alpha * distillation_error + (1 - alpha) * reconstruction_error
fusion_weight = 0.5  # alpha: weight for the distillation branch

student.eval()
autoencoder.eval()
train_fused_errors = []

with torch.no_grad():
    for images in train_dataloader:
        images = images.to(device)
        # Distillation error:
        student_features = student(images)
        images_3ch = images.repeat(1, 3, 1, 1)
        teacher_features = teacher(images_3ch)
        dist_error = torch.mean((student_features - teacher_features) ** 2, dim=[1, 2, 3])
        # Autoencoder reconstruction error:
        reconstructions = autoencoder(images)
        rec_error = torch.mean((reconstructions - images) ** 2, dim=[1, 2, 3])
        # Fused anomaly score:
        fused_error = fusion_weight * dist_error + (1 - fusion_weight) * rec_error
        train_fused_errors.extend(fused_error.cpu().numpy())

train_fused_errors = np.array(train_fused_errors)
threshold = train_fused_errors.mean() + 3 * train_fused_errors.std()
print(f"Calibrated fusion threshold based on training data: {threshold:.6f}")


# -----------------------
# 12. Anomaly Detection Function (Fusion of Both Branches)
# -----------------------
def detect_anomalies_with_filenames(student, teacher, autoencoder, dataloader, threshold, fusion_weight):
    results = []  # List of tuples: (image, filename, anomaly_flag)
    student.eval()
    autoencoder.eval()
    with torch.no_grad():
        for batch in dataloader:
            images, filenames = batch
            images = images.to(device)
            # Distillation branch error:
            student_features = student(images)
            images_3ch = images.repeat(1, 3, 1, 1)
            teacher_features = teacher(images_3ch)
            dist_error = torch.mean((student_features - teacher_features) ** 2, dim=[1, 2, 3])
            # Autoencoder branch error:
            reconstructions = autoencoder(images)
            rec_error = torch.mean((reconstructions - images) ** 2, dim=[1, 2, 3])
            # Fuse the errors:
            fused_error = fusion_weight * dist_error + (1 - fusion_weight) * rec_error
            anomaly_flags = fused_error > threshold
            for img, fname, flag in zip(images.cpu(), filenames, anomaly_flags.cpu().numpy()):
                results.append((img, fname, flag))
    return results


# -----------------------
# 13. Compute Anomaly Results on Validation and Test Sets
# -----------------------
val_results = detect_anomalies_with_filenames(student, teacher, autoencoder, val_dataloader, threshold, fusion_weight)


inference_start = time.time()
test_results = detect_anomalies_with_filenames(student, teacher, autoencoder, test_dataloader, threshold, fusion_weight)
inference_end = time.time()
inference_duration = inference_end - inference_start
print(f"\nInference on test set completed in {inference_duration:.2f} seconds ({inference_duration / 60:.2f} minutes)")

val_anomaly_count = sum(1 for _, _, flag in val_results if flag)
test_anomaly_count = sum(1 for _, _, flag in test_results if flag)
print("Validation anomalies detected:", val_anomaly_count, "out of", len(val_dataset))
print("Test anomalies detected:", test_anomaly_count, "out of", len(test_dataset))

# -----------------------
# 14. (Optional) Evaluation with Ground Truth Labels
# -----------------------
labels_csv_path = "/mnt/anom_proj/data/original data/labels.csv"  # Update path if needed
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
cm = confusion_matrix(y_true, y_pred)

print("\nEvaluation Metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")





# -----------------------
# 15. Random Sampling and Visualization (Optional)
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
plot_images(val_anomalies, f"{model_name} Validation Anomalies", save_filename=f"{model_name}_validation_anomalies.png")
plot_images(val_normals, f"{model_name} Validation Normals", save_filename=f"{model_name}_validation_normals.png")

test_anomalies, test_normals = pick_random_samples(test_results, num_samples=5)
plot_images(test_anomalies, f"{model_name} Test Anomalies", save_filename=f"{model_name}_test_anomalies.png")
plot_images(test_normals, f"{model_name} Test Normals", save_filename=f"{model_name}_test_normals.png")

# -----------------------
# 16. (Optional) Experiment Logging
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


# (For logging, we log the student branch parameters; you could extend this to log both branches if desired)
log_experiment_run(
    model=student,
    criterion=criterion,
    optimizer=optimizer_student,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    epoch=epochs - 1,
    patience=patience,
    train_duration=train_duration,
    inference_duration=inference_duration,
    threshold=threshold,
    acc=acc,
    prec=prec,
    rec=rec,
    f1=f1,
    cm=cm,
    save_paths=True
)
