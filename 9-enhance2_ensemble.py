# 9-enhance2_ensemble.py - FINAL FIXED VERSION
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from PIL import Image
from torchvision import transforms

# ============================================================
# CONFIGURATION
# ============================================================
IPCTI_BASE = r"D:\PhD\My Articles\PhD\10-LMR-Trinity-A Liquid-Mamba Vision Network for Explainable\Data\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI"
DATASET_PATH = os.path.join(IPCTI_BASE, "dataset")
BATCH_SIZE = 8
IMG_SIZE = (224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")

# Get existing patient IDs from dataset (for verification)
existing_ids = set()
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if os.path.isdir(folder_path) and folder.isdigit():
        existing_ids.add(int(folder))
print(f"Existing patient folders: {len(existing_ids)}")


# ============================================================
# DATASET CLASS
# ============================================================
class IPCTIDataset(Dataset):
    def __init__(self, csv_file, images_dir):
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"   Loaded {len(self.data)} patients from {os.path.basename(csv_file)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        patient_id = int(row['id'])
        age = torch.tensor(row['age'], dtype=torch.float32)

        # Find image - try different formats
        img_path = None
        for fmt in [f"{patient_id:04d}", f"{patient_id:03d}", str(patient_id)]:
            patient_dir = os.path.join(self.images_dir, fmt)
            if os.path.exists(patient_dir):
                for file in os.listdir(patient_dir):
                    if file.endswith('.jpg'):
                        img_path = os.path.join(patient_dir, file)
                        break
                if img_path:
                    break

        if img_path is None:
            # One last recursive search
            for root, dirs, files in os.walk(self.images_dir):
                for file in files:
                    if file.endswith('.jpg') and str(patient_id) in root:
                        img_path = os.path.join(root, file)
                        break
                if img_path:
                    break

        if img_path is None:
            print(f"   WARNING: No image for patient {patient_id}")
            # Return dummy image
            dummy = torch.zeros(3, IMG_SIZE[0], IMG_SIZE[1])
            return {'image': dummy, 'age': age}

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return {'image': image, 'age': age}


# ============================================================
# MODEL
# ============================================================
class LightMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        res = x
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = torch.nn.functional.silu(x)
        x = self.out_proj(x)
        return x + res


class LMR_Trinity_Age(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 64, kernel_size=16, stride=16)
        self.mamba1 = LightMambaBlock(64)
        self.mamba2 = LightMambaBlock(64)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.mamba1(x)
        x = self.mamba2(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        age = self.fc3(x)
        return age.squeeze(-1)


# ============================================================
# MAIN ENSEMBLE
# ============================================================
print("=" * 60)
print("ENSEMBLE OF 5 SINGLE-VIEW MODELS")
print("=" * 60)

# Use the exact path to the test CSV
test_csv = os.path.join(IPCTI_BASE, "cross-validation", "cv_1", "split_1", "test.csv")
print(f"Test CSV: {test_csv}")
print(f"File exists: {os.path.exists(test_csv)}")

if not os.path.exists(test_csv):
    raise FileNotFoundError(f"Test CSV not found: {test_csv}")

# Verify first few IDs in CSV
df_check = pd.read_csv(test_csv)
print(f"CSV patient IDs (first 10): {df_check['id'].head(10).tolist()}")
print(f"CSV patient IDs (last 10): {df_check['id'].tail(10).tolist()}")

# Check if all CSV IDs exist in dataset
missing_ids = set(df_check['id']) - existing_ids
if missing_ids:
    print(f"⚠️ WARNING: {len(missing_ids)} patients not in dataset: {sorted(missing_ids)[:10]}")
else:
    print(f"✓ All {len(df_check)} patients exist in dataset")

# Load test dataset
test_dataset = IPCTIDataset(test_csv, DATASET_PATH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Test samples: {len(test_dataset)}")

if len(test_dataset) == 0:
    print("No test samples. Exiting.")
    exit()

# Load all 5 trained models
models = []
for fold in range(1, 6):
    model_file = f"best_model_fold_{fold}.pth"
    if os.path.exists(model_file):
        model = LMR_Trinity_Age().to(DEVICE)
        model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        model.eval()
        models.append(model)
        print(f"✅ Loaded: {model_file}")
    else:
        print(f"❌ Missing: {model_file}")

print(f"\nLoaded {len(models)} models")

if len(models) == 0:
    print("No models found. Please run 7-train_ipcti.py first.")
    exit()

# Ensemble prediction
all_preds = []
all_truths = []

print("\nRunning ensemble predictions...")

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        images = batch['image'].to(DEVICE)
        ages = batch['age'].to(DEVICE)

        # Get predictions from all 5 models
        batch_preds = []
        for model in models:
            output = model(images)
            batch_preds.append(output)

        # Average predictions
        ensemble_pred = torch.stack(batch_preds).mean(dim=0)

        all_preds.extend(ensemble_pred.cpu().numpy())
        all_truths.extend(ages.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"   Processed {batch_idx + 1} batches")

# Calculate metrics
mae = mean_absolute_error(all_truths, all_preds)
rmse = np.sqrt(mean_squared_error(all_truths, all_preds))
r2 = r2_score(all_truths, all_preds)

print("\n" + "=" * 60)
print("ENSEMBLE RESULTS")
print("=" * 60)
print(f"Mean Absolute Error (MAE):  {mae:.2f} years")
print(f"Root Mean Square Error:     {rmse:.2f} years")
print(f"R² Score:                   {r2:.3f}")
print("=" * 60)

print("\n📊 FINAL COMPARISON:")
print(f"   Baseline (7-train_ipcti.py):       10.11 years")
print(f"   Enhancement A (Multi-view):        9.99 years")
print(f"   Enhancement B (Ensemble):          {mae:.2f} years")

# Save results
with open("ensemble_results.txt", "w") as f:
    f.write("Ensemble Results (5 Single-View Models)\n")
    f.write("=" * 50 + "\n")
    f.write(f"MAE: {mae:.2f} years\n")
    f.write(f"RMSE: {rmse:.2f} years\n")
    f.write(f"R²: {r2:.3f}\n")

print("\n✅ Results saved to ensemble_results.txt")