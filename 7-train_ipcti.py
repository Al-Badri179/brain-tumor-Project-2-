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
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 100  # Increased for better convergence
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


# ============================================================
# DATASET CLASS WITH AUGMENTATION
# ============================================================
class IPCTIDataset(Dataset):
    def __init__(self, csv_file, images_dir, existing_patients, augment=False):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['id'].isin(existing_patients)]
        self.images_dir = images_dir
        self.augment = augment

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
            # Validation/Test - no augmentation
            self.transform = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        print(f"   {len(pd.read_csv(csv_file))} -> {len(self.data)} samples (Augment: {augment})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        patient_id = int(row['id'])
        age = torch.tensor(row['age'], dtype=torch.float32)

        # Try different folder name formats
        img_path = None
        for folder_name in [f"{patient_id:04d}", f"{patient_id:05d}", str(patient_id)]:
            patient_dir = os.path.join(self.images_dir, folder_name)
            if os.path.exists(patient_dir):
                for file in os.listdir(patient_dir):
                    if file.endswith('.jpg'):
                        img_path = os.path.join(patient_dir, file)
                        break
                if img_path:
                    break

        if img_path is None:
            for root, dirs, files in os.walk(self.images_dir):
                for file in files:
                    if file.endswith('.jpg') and str(patient_id) in root:
                        img_path = os.path.join(root, file)
                        break
                if img_path:
                    break

        if img_path is None:
            raise FileNotFoundError(f"Image not found for patient {patient_id}")

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return {'image': image, 'age': age}


# ============================================================
# MODEL (Same as before)
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
# CROSS-VALIDATION FUNCTION
# ============================================================
def train_fold(train_csv, test_csv, fold_name):
    print(f"\n{'=' * 60}")
    print(f"Training Fold: {fold_name}")
    print(f"{'=' * 60}")

    # Create datasets
    train_dataset = IPCTIDataset(train_csv, DATASET_PATH, existing_folders, augment=True)
    test_dataset = IPCTIDataset(test_csv, DATASET_PATH, existing_folders, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print(f"Skipping fold {fold_name} - no data")
        return None

    # Initialize model
    model = LMR_Trinity_Age().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    best_mae = float('inf')
    best_epoch = 0

    for epoch in range(EPOCHS):
        # Training
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

        # Validation
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

        # Learning rate scheduling
        scheduler.step(mae)

        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS} | Loss: {train_loss:.4f} | MAE: {mae:.2f} yrs | RMSE: {rmse:.2f} | R²: {r2:.3f} | LR: {current_lr:.2e}")

        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f"best_model_{fold_name}.pth")
            print(f"  >>> New best model (MAE: {mae:.2f}) at epoch {best_epoch}")

    print(f"\nFold {fold_name} completed. Best MAE: {best_mae:.2f} years at epoch {best_epoch}")

    return best_mae


# ============================================================
# MAIN: RUN CROSS-VALIDATION
# ============================================================
def run_cross_validation():
    cv_base = os.path.join(IPCTI_BASE, "cross-validation")

    # We'll use 5 folds, each with split_1
    all_results = []

    for fold in range(1, 6):
        fold_dir = os.path.join(cv_base, f"cv_{fold}")
        split_dir = os.path.join(fold_dir, "split_1")  # Use first split of each fold

        train_csv = os.path.join(split_dir, "train.csv")
        test_csv = os.path.join(split_dir, "test.csv")

        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            print(f"Missing CSV files for cv_{fold}")
            continue

        print(f"\n{'=' * 70}")
        print(f"FOLD {fold}/5")
        print(f"{'=' * 70}")

        best_mae = train_fold(train_csv, test_csv, f"fold_{fold}")

        if best_mae is not None:
            all_results.append(best_mae)

    # Summary
    print(f"\n{'=' * 70}")
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print(f"{'=' * 70}")

    for i, mae in enumerate(all_results, 1):
        print(f"Fold {i}: MAE = {mae:.2f} years")

    if all_results:
        mean_mae = np.mean(all_results)
        std_mae = np.std(all_results)
        print(f"\nMean MAE across {len(all_results)} folds: {mean_mae:.2f} ± {std_mae:.2f} years")
        print(f"{'=' * 70}")

        # Save results
        with open("cv_results.txt", "w") as f:
            f.write(f"Cross-Validation Results\n")
            f.write(f"{'=' * 50}\n")
            for i, mae in enumerate(all_results, 1):
                f.write(f"Fold {i}: {mae:.2f} years\n")
            f.write(f"\nMean MAE: {mean_mae:.2f} ± {std_mae:.2f} years\n")
        print("Results saved to cv_results.txt")
    else:
        print("No folds completed successfully.")


if __name__ == "__main__":
    run_cross_validation()