# This script is for running a CNN Model for supervised learning
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from datetime import datetime
import numpy as np
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import csv
import seaborn as sns

# Mapping from CSV split names to actual folder names
SPLIT_TO_FOLDER = {
    'train': 'output_train',
    'validation': 'output_val'
}

class AnomalyDataset(Dataset):
    def __init__(self, csv_file, split, root_dir, transform=None, train_only_normals=True):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['split'] == split]
        if train_only_normals and split == 'train':
            self.data = self.data[self.data['label'] == 0]
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_filename = self.data.iloc[idx]['filename']
        folder = SPLIT_TO_FOLDER.get(self.split, self.split)
        img_path = os.path.join(self.root_dir, folder, img_filename)
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx]['label'])
        if self.transform:
            image = self.transform(image)
        return image, label

# ---- Data Preparation ----
img_size = 224
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

csv_file = "/mnt/anom_proj/data/New/split_labels.csv"
root_dir  = "/mnt/anom_proj/data/New"

train_dataset = AnomalyDataset(csv_file, 'train', root_dir, transform, train_only_normals=False)
val_dataset   = AnomalyDataset(csv_file, 'validation', root_dir, transform, train_only_normals=False)
test_dataset  = AnomalyDataset(csv_file, 'test', root_dir, transform, train_only_normals=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# ---- Model Setup ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomCNN(nn.Module):
    def __init__(self, img_size=224, num_pools=5):
        super(CustomCNN, self).__init__()
        # Five conv+pool blocks
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(2, 2)

        # compute spatial size after all the pooling layers
        # e.g. 224 // (2**5) == 7
        final_spatial = img_size // (2 ** num_pools)

        self.flatten     = nn.Flatten()
        self.fc1         = nn.Linear(16 * final_spatial * final_spatial, 128)
        self.hidden_relu = nn.ReLU()
        self.fc2         = nn.Linear(128, 1)
        self.sigmoid     = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.flatten(x)
        x = self.hidden_relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = CustomCNN().to(device)
model_name = model.__class__.__name__

# ---- Loss and Optimizer ----
criterion = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters(), lr=1e-4)

# ---- Train & Validation ----
train_losses, val_losses = [], []
num_epochs = 30
train_start = time.time()

for epoch in range(num_epochs):
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    # Training
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f'  Train Loss: {avg_train_loss:.4f}')

    # Validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            running_val_loss += criterion(outputs, labels).item()
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f'  Val   Loss: {avg_val_loss:.4f}')

train_duration = time.time() - train_start
torch.save(model.state_dict(), f"models/{model_name}_final.pth")

# ---- Plot Loss Curve ----
os.makedirs("results/plots", exist_ok=True)
plt.figure(figsize=(12,6))
plt.plot(range(1, num_epochs+1), train_losses, marker='o', label="Train Loss")
plt.plot(range(1, num_epochs+1), val_losses,   marker='o', label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title(f"{model_name} Loss over {num_epochs} Epochs")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(f"results/plots/{model_name}_loss_curve.png")
plt.show()

# ---- Testing & Metrics ----
print("\nModel Testing")
test_start = time.time()
model.eval()
running_test_loss, correct, total = 0.0, 0, 0
test_records = []

with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        running_test_loss += criterion(outputs, labels).item()
        preds = (outputs > 0.5).int()
        total += labels.size(0)
        correct += (preds == labels.int()).sum().item()

        start = idx * test_loader.batch_size
        fnames = test_dataset.data.iloc[start:start+images.size(0)]['filename'].tolist()
        for f, t, p in zip(fnames, labels.cpu().numpy().flatten(), preds.cpu().numpy().flatten()):
            test_records.append({"filename": f, "true_label": int(t), "predicted_label": int(p)})

avg_test_loss = running_test_loss / len(test_loader)
test_acc = 100 * correct / total
print(f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}%")
inference_duration = time.time() - test_start

y_true = [r["true_label"] for r in test_records]
y_pred = [r["predicted_label"] for r in test_records]
acc, prec, rec, f1 = (
    accuracy_score(y_true, y_pred),
    precision_score(y_true, y_pred, zero_division=0),
    recall_score(y_true, y_pred, zero_division=0),
    f1_score(y_true, y_pred, zero_division=0),
)
cm = confusion_matrix(y_true, y_pred)
print(f"\nAccuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)

os.makedirs('results/plots/test_results', exist_ok=True)
pd.DataFrame(test_records).to_csv(f"results/plots/test_results/{model_name}_test_results.csv", index=False)

# ---- Experiment Logging ----
patience =0
threshold = 0.0

def log_experiment_run(model, criterion, optimizer, train_dataloader, val_dataloader, test_dataloader,
                       epoch, patience, train_duration, inference_duration,
                       threshold, acc, prec, rec, f1, cm,
                       save_model=True, save_paths=True, log_csv=True, img_size=224):
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
        os.makedirs("results/plots", exist_ok=True)
        cm_path = f"results/plots/confusion_matrix_{version_prefix}.png"
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Anomaly"],
                    yticklabels=["Normal", "Anomaly"])
        plt.title(f"{model_name} Confusion Matrix")
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


# Call logging function
log_experiment_run(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    test_dataloader=test_loader,
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
    save_model=True,
    log_csv=True,
)
