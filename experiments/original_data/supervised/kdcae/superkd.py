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
from sklearn.neighbors import KernelDensity  # for KDE


threshold =0
patience =0

# Mapping from CSV split names to actual folder names
SPLIT_TO_FOLDER = {
    'train': 'output_train',
    'validation': 'output_val',
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

# ---- KD-CAE MODEL DEFINITION ----
class KD_CAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(KD_CAE, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 112x112
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 56x56
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),# 28x28
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),#14x14
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(256*14*14, latent_dim),
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256*14*14),
            nn.Unflatten(1, (256,14,14)),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1), #28x28
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1),  #56x56
            nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1),   #112x112
            nn.ReLU(True),
            nn.ConvTranspose2d(32,3,4,2,1),    #224x224
            nn.Sigmoid(),  # normalized [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

# ---- Data Preparation ----
img_size = 224
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])
csv_file = "/mnt/anom_proj/data/New/split_labels.csv"
root_dir = "/mnt/anom_proj/data/New"

train_dataset = AnomalyDataset(csv_file, 'train', root_dir, transform, train_only_normals=False)
val_dataset   = AnomalyDataset(csv_file, 'validation', root_dir, transform, train_only_normals=False)
test_dataset  = AnomalyDataset(csv_file, 'test', root_dir, transform, train_only_normals=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# ---- Model Setup ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KD_CAE(latent_dim=128).to(device)
model_name = model.__class__.__name__

# ---- Loss and Optimizer ----
criterion = nn.MSELoss()          # reconstruction error
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---- Train/Val Loop ----
train_losses, val_losses = [], []
num_epochs = 30
train_start = time.time()

for epoch in range(num_epochs):
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    model.train()
    running_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)
        recon, _ = model(images)
        loss = criterion(recon, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train = running_loss / len(train_loader)
    train_losses.append(avg_train)
    print(f'Train Loss: {avg_train:.4f}')

    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            recon, _ = model(images)
            running_val += criterion(recon, images).item()
    avg_val = running_val / len(val_loader)
    val_losses.append(avg_val)
    print(f'Val Loss: {avg_val:.4f}')

train_end = time.time()
print('Training and Validation Complete.')

# ---- FIT KDE ON TRAIN LATENTS ----
model.eval()
latents = []
with torch.no_grad():
    for images, _ in train_loader:
        images = images.to(device)
        _, z = model(images)
        latents.append(z.cpu().numpy())
latents = np.concatenate(latents, axis=0)
kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(latents)

# ---- Plot Loss Curves ----
os.makedirs("results/plots", exist_ok=True)
plt.figure(figsize=(12, 6))
epochs_run = len(train_losses)
plt.plot(range(1, epochs_run+1), train_losses, marker='o', label="Training Loss")
plt.plot(range(1, epochs_run+1), val_losses, marker='o', label="Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.title(f"{model_name} Reconstruction Loss over {epochs_run} epochs")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(f"results/plots/{model_name}_loss_curve.png")
plt.show()

# ---- Testing and Anomaly Scoring ----
print("\nModel Testing")
test_start = time.time()
model.eval()

test_records = []
correct = 0
total = 0

with torch.no_grad():
    for idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        recon, z = model(images)
        # reconstruction error per sample
        errors = torch.mean((recon - images)**2, dim=[1,2,3]).cpu().numpy()
        # log-density per sample
        zs = z.cpu().numpy()
        logdens = kde.score_samples(zs)
        # anomaly score = error - log-density
        scores = errors - logdens
        preds = (scores > threshold).astype(int)

        total += labels.size(0)
        correct += (preds == labels.numpy()).sum()

        batch_start = idx * test_loader.batch_size
        fnames = test_dataset.data.iloc[batch_start:batch_start+len(scores)]['filename'].tolist()
        for f, t, p in zip(fnames, labels.numpy(), preds):
            test_records.append({"filename":f, "true_label":int(t), "predicted_label":int(p)})

avg_test_loss = np.mean(errors)  # approx avg reconstruction loss
test_acc = 100 * correct / total
print(f'Test Recon Loss: {avg_test_loss:.4f} | Test Accuracy: {test_acc:.2f}%')

# ---- Rest of your existing evaluation & logging (unchanged) ----
y_true = [r["true_label"] for r in test_records]
y_pred = [r["predicted_label"] for r in test_records]
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)
cm   = confusion_matrix(y_true, y_pred)

print("\nEvaluation Metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)

# Save test results
results_df = pd.DataFrame(test_records)
os.makedirs('results/plots/test_results', exist_ok=True)
results_csv_path = f"results/plots/test_results/{model_name}_test_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Test results saved to {results_csv_path}")


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
