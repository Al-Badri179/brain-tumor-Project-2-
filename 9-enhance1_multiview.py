# 7_train_ipcti_multiview.py (FIXED)
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

# ============================================================
# CONFIGURATION
# ============================================================
IPCTI_BASE = r"D:\PhD\My Articles\PhD\10-LMR-Trinity-A Liquid-Mamba Vision Network for Explainable\Data\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI"
DATASET_PATH = os.path.join(IPCTI_BASE, "dataset")
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 100
NUM_WORKERS = 0
IMG_SIZE = (224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Data path: {DATASET_PATH}")
print(f"Exists: {os.path.exists(DATASET_PATH)}")

# Get existing patient folders (as integers)
existing_folders = set()
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if os.path.isdir(folder_path) and folder.isdigit():
        existing_folders.add(int(folder))

print(f"Existing patient folders: {len(existing_folders)}")
print(f"First 20 folders: {sorted(existing_folders)[:20]}")


# ============================================================
# MULTI-VIEW DATASET CLASS (with better patient filtering)
# ============================================================
class IPCTIDataset_Multiview(Dataset):
    def __init__(self, csv_file, images_dir, existing_patients, augment=False):
        # Load CSV
        self.raw_data = pd.read_csv(csv_file)
        # Filter to only patients that exist in folders
        self.data = self.raw_data[self.raw_data['id'].isin(existing_patients)]
        self.images_dir = images_dir
        self.augment = augment

        # Debug: print missing patients
        csv_ids = set(self.raw_data['id'].values)
        missing = csv_ids - existing_patients
        if missing:
            print(f"   ⚠️ Skipping {len(missing)} patients not found in folders: {sorted(missing)[:10]}...")

        # Training augmentation
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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

        print(f"   Original CSV: {len(self.raw_data)} rows -> Filtered: {len(self.data)} patients (Augment: {augment})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        patient_id = int(row['id'])
        age = torch.tensor(row['age'], dtype=torch.float32)

        # Get patient folder (try different formats)
        images = []

        # Try 4-digit format first
        for fmt in [f"{patient_id:04d}", f"{patient_id:05d}", str(patient_id)]:
            patient_dir = os.path.join(self.images_dir, fmt)
            if os.path.exists(patient_dir):
                for file in sorted(os.listdir(patient_dir)):
                    if file.endswith('.jpg'):
                        img_path = os.path.join(patient_dir, file)
                        img = Image.open(img_path).convert('RGB')
                        img = self.transform(img)
                        images.append(img)
                if images:
                    break

        # If still no images, search recursively
        if len(images) == 0:
            for root, dirs, files in os.walk(self.images_dir):
                for file in files:
                    if file.endswith('.jpg') and (str(patient_id) in root or file.startswith(str(patient_id))):
                        img_path = os.path.join(root, file)
                        img = Image.open(img_path).convert('RGB')
                        img = self.transform(img)
                        images.append(img)
                if images:
                    break

        if len(images) == 0:
            # Return a dummy image and print warning (instead of crashing)
            print(f"   ⚠️ Warning: No images found for patient {patient_id}, using black image")
            dummy = torch.zeros(3, IMG_SIZE[0], IMG_SIZE[1])
            images = [dummy, dummy, dummy, dummy]

        # Ensure we have 4 images (pad if necessary)
        while len(images) < 4:
            images.append(images[0] if images else torch.zeros(3, IMG_SIZE[0], IMG_SIZE[1]))

        # Use only first 4 images
        multiview = torch.stack(images[:4], dim=0)  # [4, 3, H, W]

        return {
            'image': multiview,
            'age': age,
            'patient_id': patient_id
        }


# ============================================================
# MULTI-VIEW MODEL
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


class LMR_Trinity_Age_Multiview(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=16, num_views=4):
        super().__init__()
        self.num_views = num_views

        # Shared patch embedding
        self.patch_embed = nn.Conv2d(3, 64, kernel_size=patch_size, stride=patch_size)

        # Shared Mamba processor (share weights across views)
        self.mamba_shared = nn.Sequential(
            LightMambaBlock(64),
            LightMambaBlock(64)
        )

        # Cross-view pooling and fusion
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Regression head
        self.fc1 = nn.Linear(64 * num_views, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: [B, num_views, 3, H, W]
        B, V, C, H, W = x.shape

        view_features = []
        for v in range(V):
            view = x[:, v, :, :, :]  # [B, 3, H, W]

            # Patch embed
            view_feat = self.patch_embed(view)  # [B, 64, H', W']

            # Flatten for Mamba
            view_feat = view_feat.flatten(2).transpose(1, 2)  # [B, seq_len, 64]

            # Apply shared Mamba
            view_feat = self.mamba_shared(view_feat)

            # Global pooling
            view_feat = view_feat.transpose(1, 2)  # [B, 64, seq_len]
            view_feat = self.global_pool(view_feat).squeeze(-1)  # [B, 64]
            view_features.append(view_feat)

        # Concatenate all views
        all_features = torch.cat(view_features, dim=1)  # [B, 64 * V]

        # Regression head
        x = torch.nn.functional.relu(self.fc1(all_features))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.dropout(x)
        age = self.fc4(x)

        return age.squeeze(-1)


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_fold(train_csv, test_csv, fold_name):
    print(f"\n{'=' * 60}")
    print(f"Multi-View Training Fold: {fold_name}")
    print(f"{'=' * 60}")

    train_dataset = IPCTIDataset_Multiview(train_csv, DATASET_PATH, existing_folders, augment=True)
    test_dataset = IPCTIDataset_Multiview(test_csv, DATASET_PATH, existing_folders, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Training patients: {len(train_dataset)}")
    print(f"Test patients: {len(test_dataset)}")

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print(f"Skipping fold {fold_name} - no data")
        return None

    model = LMR_Trinity_Age_Multiview(IMG_SIZE).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    best_mae = float('inf')
    best_epoch = 0

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nStarting training for {EPOCHS} epochs...\n")

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

        scheduler.step(mae)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | Loss: {train_loss:.4f} | MAE: {mae:.2f} yrs | RMSE: {rmse:.2f} | R²: {r2:.3f} | LR: {current_lr:.2e}")

        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"best_model_multiview_{fold_name}.pth")
            print(f"  >>> New best model (MAE: {mae:.2f}) at epoch {best_epoch}")

    print(f"\nFold {fold_name} completed. Best MAE: {best_mae:.2f} years at epoch {best_epoch}")

    return best_mae


# ============================================================
# MAIN: RUN CROSS-VALIDATION
# ============================================================
def run_cross_validation():
    cv_base = os.path.join(IPCTI_BASE, "cross-validation")

    all_results = []

    for fold in range(1, 6):
        fold_dir = os.path.join(cv_base, f"cv_{fold}")
        split_dir = os.path.join(fold_dir, "split_1")

        train_csv = os.path.join(split_dir, "train.csv")
        test_csv = os.path.join(split_dir, "test.csv")

        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            print(f"Missing CSV files for cv_{fold}")
            continue

        print(f"\n{'=' * 70}")
        print(f"MULTI-VIEW FOLD {fold}/5")
        print(f"{'=' * 70}")

        best_mae = train_fold(train_csv, test_csv, f"fold_{fold}")

        if best_mae is not None:
            all_results.append(best_mae)

    print(f"\n{'=' * 70}")
    print("MULTI-VIEW CROSS-VALIDATION RESULTS")
    print(f"{'=' * 70}")

    for i, mae in enumerate(all_results, 1):
        print(f"Fold {i}: MAE = {mae:.2f} years")

    if all_results:
        mean_mae = np.mean(all_results)
        std_mae = np.std(all_results)
        print(f"\nMean MAE across {len(all_results)} folds: {mean_mae:.2f} ± {std_mae:.2f} years")

        with open("cv_results_multiview.txt", "w") as f:
            f.write("Multi-View Cross-Validation Results\n")
            f.write(f"{'=' * 50}\n")
            for i, mae in enumerate(all_results, 1):
                f.write(f"Fold {i}: {mae:.2f} years\n")
            f.write(f"\nMean MAE: {mean_mae:.2f} ± {std_mae:.2f} years\n")
        print("\nResults saved to cv_results_multiview.txt")


if __name__ == "__main__":
    run_cross_validation()