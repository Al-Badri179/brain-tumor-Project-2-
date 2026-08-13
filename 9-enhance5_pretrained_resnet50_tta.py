# 9-enhance5_tta.py
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
import torchvision.models as models

# ============================================================
# CONFIGURATION
# ============================================================
IPCTI_BASE = r"D:\PhD\My Articles\PhD\10-LMR-Trinity-A Liquid-Mamba Vision Network for Explainable\Data\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI"
DATASET_PATH = os.path.join(IPCTI_BASE, "dataset")
BATCH_SIZE = 8
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
    def __init__(self, csv_file, images_dir):
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
# PRETRAINED MODEL (same as Enhancement C)
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


class LMR_Trinity_Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        # Pretrained ResNet50
        self.backbone = models.resnet50(weights='IMAGENET1K_V2')
        self.backbone.fc = nn.Identity()

        # Mamba layers
        self.mamba1 = LightMambaBlock(2048)
        self.mamba2 = LightMambaBlock(2048)

        # Global pooling and regression
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        features = self.backbone(x)  # [B, 2048]
        features = features.unsqueeze(1)  # [B, 1, 2048]

        x = self.mamba1(features)
        x = self.mamba2(x)

        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        age = self.fc3(x)

        return age.squeeze(-1)


# ============================================================
# TEST-TIME AUGMENTATION FUNCTION
# ============================================================
def predict_with_tta(model, image, device):
    """Apply multiple augmentations and average predictions"""
    predictions = []

    # 1. Original image
    pred = model(image.unsqueeze(0).to(device))
    predictions.append(pred.item())

    # 2. Horizontal flip
    flipped = torch.flip(image, dims=[1])  # [C, H, W] -> flip width
    pred = model(flipped.unsqueeze(0).to(device))
    predictions.append(pred.item())

    # 3. Small rotations
    for angle in [-5, 5, -10, 10]:
        rotated = transforms.functional.rotate(image, angle)
        pred = model(rotated.unsqueeze(0).to(device))
        predictions.append(pred.item())

    # 4. Slight brightness adjustment
    bright = transforms.functional.adjust_brightness(image, 1.1)
    pred = model(bright.unsqueeze(0).to(device))
    predictions.append(pred.item())

    # 5. Slight contrast adjustment
    contrast = transforms.functional.adjust_contrast(image, 1.1)
    pred = model(contrast.unsqueeze(0).to(device))
    predictions.append(pred.item())

    return np.mean(predictions)


def evaluate_with_tta(model, test_loader, device):
    """Evaluate model using Test-Time Augmentation"""
    all_preds = []
    all_truths = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            ages = batch['age'].to(device)

            # Apply TTA for each image in batch
            batch_preds = []
            for i in range(images.shape[0]):
                pred = predict_with_tta(model, images[i], device)
                batch_preds.append(pred)

            all_preds.extend(batch_preds)
            all_truths.extend(ages.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"   Processed {batch_idx + 1} batches")

    return all_preds, all_truths


# ============================================================
# MAIN
# ============================================================
print("=" * 60)
print("ENHANCEMENT E: TEST-TIME AUGMENTATION (TTA)")
print("=" * 60)

# Load CSV paths
test_csv = os.path.join(IPCTI_BASE, "cross-validation", "cv_1", "split_1", "test.csv")
train_csv = os.path.join(IPCTI_BASE, "cross-validation", "cv_1", "split_1", "train.csv")

# Create datasets
train_dataset = IPCTIDataset(train_csv, DATASET_PATH)
test_dataset = IPCTIDataset(test_csv, DATASET_PATH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nTraining samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Load best pretrained model
model = LMR_Trinity_Pretrained().to(DEVICE)

# Try to load the best model from Enhancement C
model_paths = ["best_pretrained_model.pth", "best_model_fold_3.pth", "best_model_fold_1.pth"]
model_loaded = False

for path in model_paths:
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"✅ Loaded model: {path}")
        model_loaded = True
        break

if not model_loaded:
    print("❌ No pretrained model found. Please run 9-enhance3_pretrained.py first.")
    exit()

model.eval()

# ============================================================
# BASELINE EVALUATION (without TTA)
# ============================================================
print("\n" + "=" * 60)
print("BASELINE EVALUATION (without TTA)")
print("=" * 60)

all_preds_baseline = []
all_truths = []

with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(DEVICE)
        ages = batch['age'].to(DEVICE)
        outputs = model(images)
        all_preds_baseline.extend(outputs.cpu().numpy())
        all_truths.extend(ages.cpu().numpy())

mae_baseline = mean_absolute_error(all_truths, all_preds_baseline)
rmse_baseline = np.sqrt(mean_squared_error(all_truths, all_preds_baseline))
r2_baseline = r2_score(all_truths, all_preds_baseline)

print(f"MAE (without TTA):  {mae_baseline:.2f} years")
print(f"RMSE:               {rmse_baseline:.2f} years")
print(f"R²:                 {r2_baseline:.3f}")

# ============================================================
# TTA EVALUATION (with Test-Time Augmentation)
# ============================================================
print("\n" + "=" * 60)
print("TTA EVALUATION (with Test-Time Augmentation)")
print("=" * 60)

all_preds_tta, all_truths = evaluate_with_tta(model, test_loader, DEVICE)

mae_tta = mean_absolute_error(all_truths, all_preds_tta)
rmse_tta = np.sqrt(mean_squared_error(all_truths, all_preds_tta))
r2_tta = r2_score(all_truths, all_preds_tta)

print(f"\nMAE (with TTA):     {mae_tta:.2f} years")
print(f"RMSE:               {rmse_tta:.2f} years")
print(f"R²:                 {r2_tta:.3f}")

# ============================================================
# RESULTS COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("ENHANCEMENT E RESULTS (Test-Time Augmentation)")
print("=" * 60)
print(f"Baseline (no TTA):  {mae_baseline:.2f} years")
print(f"With TTA:           {mae_tta:.2f} years")
print(f"Improvement:        {mae_baseline - mae_tta:.2f} years ({(mae_baseline - mae_tta) / mae_baseline * 100:.1f}%)")
print("=" * 60)

# ============================================================
# FINAL COMPARISON OF ALL ENHANCEMENTS
# ============================================================
print("\n📊 FINAL COMPARISON OF ALL METHODS:")
print("-" * 60)
print(f"   Baseline (7-train_ipcti.py):                10.11 years")
print(f"   Enhancement A (Multi-view):                 9.99 years")
print(f"   Enhancement B (Ensemble):                   10.50 years")
print(f"   Enhancement C (Pretrained ResNet50):        6.10 years")
print(f"   Enhancement D (Swin Transformer):           9.79 years")
print(f"   Enhancement E (Pretrained + TTA):           {mae_tta:.2f} years")
print("-" * 60)

# Find best method
results = {
    "Baseline": 10.11,
    "Multi-view": 9.99,
    "Ensemble": 10.50,
    "Pretrained ResNet50": 6.10,
    "Swin Transformer": 9.79,
    f"Pretrained + TTA": mae_tta
}

best_method = min(results, key=results.get)
print(f"\n🏆 BEST METHOD: {best_method} with MAE = {results[best_method]:.2f} years")

# Save results
with open("tta_results.txt", "w") as f:
    f.write("Test-Time Augmentation Results\n")
    f.write("=" * 40 + "\n")
    f.write(f"Baseline (no TTA): {mae_baseline:.2f} years\n")
    f.write(f"With TTA:          {mae_tta:.2f} years\n")
    f.write(f"Improvement:       {mae_baseline - mae_tta:.2f} years\n")
    f.write(f"Improvement %:     {(mae_baseline - mae_tta) / mae_baseline * 100:.1f}%\n")

print("\n✅ Results saved to tta_results.txt")