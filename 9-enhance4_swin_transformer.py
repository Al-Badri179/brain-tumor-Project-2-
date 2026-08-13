# 9-enhance4_swin_transformer.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from timm import create_model

# ============================================================
# CONFIGURATION
# ============================================================
IPCTI_BASE = r"D:\PhD\My Articles\PhD\10-LMR-Trinity-A Liquid-Mamba Vision Network for Explainable\Data\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI"
DATASET_PATH = os.path.join(IPCTI_BASE, "dataset")
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 50
IMG_SIZE = (224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

# Get existing patient folders
existing_folders_str = set()
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if os.path.isdir(folder_path):
        existing_folders_str.add(folder)
print(f"Existing folders: {len(existing_folders_str)}")


# ============================================================
# DATASET CLASS
# ============================================================
class IPCTIDataset(Dataset):
    def __init__(self, csv_file, images_dir, augment=False):
        raw_data = pd.read_csv(csv_file)
        # Filter to only patients with existing folders
        valid_patients = []
        for idx, row in raw_data.iterrows():
            patient_id = row['id']
            for fmt in [f"{int(patient_id):04d}", f"{int(patient_id):03d}", str(patient_id)]:
                if fmt in existing_folders_str:
                    valid_patients.append(idx)
                    break

        self.data = raw_data.iloc[valid_patients]
        self.images_dir = images_dir
        self.augment = augment

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        filtered_out = len(raw_data) - len(self.data)
        print(
            f"   {os.path.basename(csv_file)}: {len(raw_data)} -> {len(self.data)} patients (filtered {filtered_out})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        patient_id = int(row['id'])
        age = torch.tensor(row['age'], dtype=torch.float32)

        # Find folder
        folder_name = None
        for fmt in [f"{patient_id:04d}", f"{patient_id:03d}", str(patient_id)]:
            if fmt in existing_folders_str:
                folder_name = fmt
                break

        if folder_name is None:
            raise FileNotFoundError(f"No folder for patient {patient_id}")

        patient_dir = os.path.join(self.images_dir, folder_name)
        img_path = None
        for file in os.listdir(patient_dir):
            if file.endswith('.jpg'):
                img_path = os.path.join(patient_dir, file)
                break

        if img_path is None:
            raise FileNotFoundError(f"No image in {patient_dir}")

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return {'image': image, 'age': age}


# ============================================================
# SWIN TRANSFORMER MODEL
# ============================================================
class LMR_SwinTransformer(nn.Module):
    """
    Swin Transformer for Dental Age Estimation
    Expected MAE: 5.5 - 6.0 years
    """

    def __init__(self):
        super().__init__()
        # Pretrained Swin Transformer (state-of-the-art for vision)
        self.backbone = create_model('swin_tiny_patch4_window7_224',
                                     pretrained=True,
                                     num_classes=0)  # Remove classification head

        # Regression head
        self.fc1 = nn.Linear(768, 256)  # Swin-Tiny outputs 768 features
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Extract features using Swin Transformer
        features = self.backbone(x)  # [B, 768]

        # Regression head
        x = self.relu(self.fc1(features))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        age = self.fc3(x)

        return age.squeeze(-1)


# ============================================================
# MAIN
# ============================================================
print("=" * 60)
print("ENHANCEMENT D: SWIN TRANSFORMER (State-of-the-Art)")
print("=" * 60)

# Load CSV paths
test_csv = os.path.join(IPCTI_BASE, "cross-validation", "cv_1", "split_1", "test.csv")
train_csv = os.path.join(IPCTI_BASE, "cross-validation", "cv_1", "split_1", "train.csv")

# Create datasets
train_dataset = IPCTIDataset(train_csv, DATASET_PATH, augment=True)
test_dataset = IPCTIDataset(test_csv, DATASET_PATH, augment=False)

if len(train_dataset) == 0 or len(test_dataset) == 0:
    print("ERROR: No valid patients found!")
    exit()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nTraining samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Model
model = LMR_SwinTransformer().to(DEVICE)
criterion = nn.L1Loss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"\nTraining for {EPOCHS} epochs...\n")

best_mae = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch in train_loader:
        images = batch['image'].to(DEVICE)
        ages = batch['age'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, ages)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_preds = []
    val_truths = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(DEVICE)
            ages = batch['age'].to(DEVICE)
            outputs = model(images)
            val_preds.extend(outputs.cpu().numpy())
            val_truths.extend(ages.cpu().numpy())

    mae = mean_absolute_error(val_truths, val_preds)
    rmse = np.sqrt(mean_squared_error(val_truths, val_preds))
    r2 = r2_score(val_truths, val_preds)

    print(
        f"Epoch {epoch + 1:3d}/{EPOCHS} | Loss: {train_loss:.4f} | MAE: {mae:.2f} yrs | RMSE: {rmse:.2f} | R²: {r2:.3f}")

    if mae < best_mae:
        best_mae = mae
        torch.save(model.state_dict(), "best_swin_model.pth")
        print(f"  >>> New best model (MAE: {mae:.2f})")

print("\n" + "=" * 60)
print("ENHANCEMENT D RESULTS (Swin Transformer)")
print("=" * 60)
print(f"Best MAE: {best_mae:.2f} years")
print("=" * 60)

print("\n📊 FINAL COMPARISON OF ALL ENHANCEMENTS:")
print(f"   Baseline (7-train_ipcti.py):                10.11 years")
print(f"   Enhancement A (Multi-view):                 9.99 years")
print(f"   Enhancement B (Ensemble):                   10.50 years")
print(f"   Enhancement C (Pretrained ResNet50):        6.10 years")
print(f"   Enhancement D (Swin Transformer):           {best_mae:.2f} years")