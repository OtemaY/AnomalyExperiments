import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import time
from datetime import datetime
import numpy as np
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import csv
import seaborn as sns

# Mapping from CSV split names to actual folder names
SPLIT_TO_FOLDER = {
    'train': 'superaugnormal',
    'val': 'super_val',
}

class AnomalyDataset(Dataset):
    def __init__(self, csv_file, split, root_dir, transform=None, train_only_normals=True):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['split'] == split]

        if train_only_normals and split == 'train':
            self.data = self.data[self.data['label'] == 0]

        self.root_dir = root_dir
        self.transform = transform
        self.split = split  # store split for use in __getitem__

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_filename = self.data.iloc[idx]['filename']
        # Map split name to actual folder name
        folder = SPLIT_TO_FOLDER.get(self.split, self.split)  # fallback to split if no mapping

        img_path = os.path.join(self.root_dir, folder, img_filename)
        image = Image.open(img_path).convert('RGB')
        label = int(self.data.iloc[idx]['label'])

        if self.transform:
            image = self.transform(image)

        return image, label



# ---- Data Preparation ----
img_size=224

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Update paths to match your new file structure
csv_file = "/mnt/anom_proj/data/New/superaugnormal/super_augment.csv"
root_dir = "/mnt/anom_proj/data/New"  # Root directory for images (train, validation, etc.)

# Create datasets (train only normal images for training)
train_dataset = AnomalyDataset(csv_file=csv_file, split='train', root_dir=root_dir, transform=transform, train_only_normals=False)
val_dataset = AnomalyDataset(csv_file=csv_file, split='val', root_dir=root_dir, transform=transform, train_only_normals=False)

# Create DataLoader for batch loading
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ---- Test Dataset ----
# Add a test dataset after training and validation
test_dataset = AnomalyDataset(csv_file=csv_file, split='test', root_dir=root_dir, transform=transform, train_only_normals=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ---- Model Setup ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a pre-trained ResNet18 and modify the final fully connected layer
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 output classes: normal and anomaly
model = model.to(device)
model_name = "AugResNet" #model.__class__.__name__



# Extract labels from training dataset (make sure train_only_normals=False to include anomalies)
# Extract labels from training dataset (make sure train_only_normals=False to include anomalies)
train_labels = train_dataset.data['label'].values

# Count samples per class
class_counts = np.bincount(train_labels)
print(f"Class counts: {class_counts}")

# Compute class weights: inverse frequency
class_weights = 1. / class_counts
print(f"Class weights (inverse frequency): {class_weights}")

# Normalize weights (optional)
class_weights = class_weights / class_weights.sum()
print(f"Normalized class weights: {class_weights}")

# Convert to torch tensor and move to device
weights_tensor = torch.FloatTensor(class_weights).to(device)

# Define loss with class weights
criterion = nn.CrossEntropyLoss(weight=weights_tensor)


# ---- Loss and Optimizer ----
#criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss expects raw logits
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---- Lists to Store Losses ----
train_losses = []
val_losses = []

# ---- Training and Validation Loop ----
num_epochs = 1  # You can increase this for more training
train_start = time.time()

for epoch in range(num_epochs):
    print(f'Epoch [{epoch+1}/{num_epochs}]')

    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)  # Raw logits from the model
        loss = criterion(outputs, labels)  # Calculate the loss

        optimizer.zero_grad()
        loss.backward()  # Backpropagate the gradients
        optimizer.step()  # Update the model's parameters

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)  # Store the training loss for plotting
    print(f'Train Loss: {avg_train_loss:.4f}')

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():  # No need to compute gradients for validation
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)  # Get raw logits
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)  # Store the validation loss for plotting
    print(f'Val Loss: {avg_val_loss:.4f}')

train_end = time.time()
train_duration = train_end - train_start

# ---- Plotting Training and Validation Loss ----
os.makedirs("results/plots", exist_ok=True)
plt.figure(figsize=(12, 6))

# Number of epochs
epochs_run = len(train_losses)

# Plot the training and validation loss curves
plt.plot(range(1, epochs_run+1), train_losses, marker='o', label="Training Loss")
plt.plot(range(1, epochs_run+1), val_losses, marker='o', label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{model.__class__.__name__} Training vs Validation Loss over {epochs_run} epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig(f"results/plots/{model.__class__.__name__}_loss_curve.png")
plt.xlim(0, epochs_run + 1)

# Adjust y-axis to cover a bit beyond the min and max losses
ymin = min(min(train_losses), min(val_losses)) - 0.1
ymax = max(max(train_losses), max(val_losses)) + 0.1
plt.ylim(ymin, ymax)

# Show the plot
plt.show()

print('Training and Validation Complete.')


# ---- Testing and Saving Predictions ----
print("\nModel Testing")
test_start = time.time()

model.eval()
running_test_loss = 0.0
correct = 0
total = 0

test_records = []  # List of dicts: each dict will store filename, true label, predicted label

with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_test_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Get filenames from dataset
        batch_start_idx = idx * test_loader.batch_size
        filenames = test_dataset.data.iloc[batch_start_idx:batch_start_idx + images.size(0)]['filename'].tolist()

        # Save results to memory
        for fname, true_label, pred_label in zip(filenames, labels.cpu().numpy(), predicted.cpu().numpy()):
            test_records.append({
                "filename": fname,
                "true_label": int(true_label),
                "predicted_label": int(pred_label)
            })

avg_test_loss = running_test_loss / len(test_loader)
test_acc = 100 * correct / total
print(f'Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_acc:.2f}%')

test_end = time.time()
inference_duration = test_end - test_start
print('Testing Complete.')

# ---- Evaluation from test_records (without reading CSV) ----
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

y_true = [record["true_label"] for record in test_records]
y_pred = [record["predicted_label"] for record in test_records]

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

print("\nConfusion Matrix:")
print(cm)

# ---- Save Test Results to CSV ----
results_df = pd.DataFrame(test_records)
os.makedirs('results/plots/test_results', exist_ok=True)
results_csv_path = f"results/plots/test_results/{model_name}_test_results.csv"
results_df.to_csv(results_csv_path, index=False)

print(f"Test results saved to {results_csv_path}")

patience =0
threshold = 0.0

def log_experiment_run(model, criterion, optimizer, train_dataloader, val_dataloader, test_dataloader,
                       epoch, patience, train_duration, inference_duration,
                       threshold, acc, prec, rec, f1, cm,
                       save_model=True, save_paths=True, log_csv=True, img_size=224):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = "AugResNet" #model.__class__.__name__
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
