# 9-enhance7_advanced.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import random

# ============================================================
# CONFIGURATION
# ============================================================
IPCTI_BASE = r"D:\PhD\My Articles\PhD\10-LMR-Trinity-A Liquid-Mamba Vision Network for Explainable\Data\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI"
DATASET_PATH = os.path.join(IPCTI_BASE, "dataset")
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 80  # Increased
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIXUP_ALPHA = 0.4
GRADIENT_CLIP = 1.0  # Added
LABEL_SMOOTHING_SIGMA = 0.3  # Added

# Progressive resizing stages
STAGES = [
    {'size': (112, 112), 'epochs': 20},
    {'size': (168, 168), 'epochs': 20},
    {'size': (224, 224), 'epochs': 40},
]

print(f"Device: {DEVICE}")


# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# Get existing patient folders
existing_folders_str = set()
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if os.path.isdir(folder_path):
        existing_folders_str.add(folder)
print(f"Existing folders: {len(existing_folders_str)}")


# ============================================================
# LABEL SMOOTHING FUNCTION
# ============================================================
def smooth_target(y, sigma=0.3):
    """Add small Gaussian noise to target ages for regularization"""
    noise = torch.randn_like(y) * sigma
    return y + noise


# ============================================================
# MIXUP FUNCTION
# ============================================================
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# DATASET CLASS
# ============================================================
class IPCTIDatasetAugmented(Dataset):
    def __init__(self, csv_file, images_dir, img_size, augment=True):
        raw_data = pd.read_csv(csv_file)
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
        self.img_size = img_size

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=25),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.15, 0.15),
                    scale=(0.8, 1.2)
                ),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.2,
                    hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
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


class LMR_Trinity_Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V2')
        self.backbone.fc = nn.Identity()

        self.mamba1 = LightMambaBlock(2048)
        self.mamba2 = LightMambaBlock(2048)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        features = self.backbone(x)
        features = features.unsqueeze(1)

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
# TRAINING FUNCTION WITH ALL TECHNIQUES
# ============================================================
def train_advanced():
    print("=" * 60)
    print("ENHANCEMENT G: PROGRESSIVE RESIZING + ALL TECHNIQUES")
    print("=" * 60)

    test_csv = os.path.join(IPCTI_BASE, "cross-validation", "cv_1", "split_1", "test.csv")
    train_csv = os.path.join(IPCTI_BASE, "cross-validation", "cv_1", "split_1", "train.csv")

    best_overall_mae = float('inf')
    global_epoch = 0

    for stage_idx, stage in enumerate(STAGES):
        img_size = stage['size']
        stage_epochs = stage['epochs']

        print(f"\n{'=' * 50}")
        print(f"STAGE {stage_idx + 1}: Image Size {img_size[0]}x{img_size[1]} for {stage_epochs} epochs")
        print(f"{'=' * 50}")

        # Create datasets with current image size
        train_dataset = IPCTIDatasetAugmented(train_csv, DATASET_PATH, img_size, augment=True)
        test_dataset = IPCTIDatasetAugmented(test_csv, DATASET_PATH, img_size, augment=False)

        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print("ERROR: No valid patients found!")
            return None

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        # Initialize or load model
        if stage_idx == 0:
            model = LMR_Trinity_Pretrained().to(DEVICE)
        else:
            # Load previous stage's best model
            model.load_state_dict(torch.load("best_advanced_model.pth"))

        # Adjust learning rate for later stages
        if stage_idx == 0:
            lr = LEARNING_RATE
        elif stage_idx == 1:
            lr = LEARNING_RATE / 2
        else:
            lr = LEARNING_RATE / 5

        criterion = nn.L1Loss()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        best_stage_mae = float('inf')

        print(
            f"\nTraining with: MixUp={MIXUP_ALPHA}, Label Smoothing={LABEL_SMOOTHING_SIGMA}, Gradient Clip={GRADIENT_CLIP}")

        for epoch in range(stage_epochs):
            global_epoch += 1
            model.train()
            train_loss = 0

            for batch in train_loader:
                images = batch['image'].to(DEVICE)
                ages = batch['age'].to(DEVICE)

                # Apply label smoothing
                ages_smoothed = smooth_target(ages, LABEL_SMOOTHING_SIGMA)

                # Apply MixUp
                images, ages_a, ages_b, lam = mixup_data(images, ages_smoothed, MIXUP_ALPHA)

                optimizer.zero_grad()
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, ages_a, ages_b, lam)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

                optimizer.step()
                scheduler.step()

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
            current_lr = optimizer.param_groups[0]['lr']

            print(
                f"Epoch {global_epoch:3d} | Loss: {train_loss:.4f} | MAE: {mae:.2f} yrs | RMSE: {rmse:.2f} | R²: {r2:.3f} | LR: {current_lr:.2e}")

            if mae < best_stage_mae:
                best_stage_mae = mae
                best_overall_mae = mae
                torch.save(model.state_dict(), "best_advanced_model.pth")
                print(f"  >>> New best model (MAE: {mae:.2f})")

    print("\n" + "=" * 60)
    print("ENHANCEMENT G RESULTS (Progressive Resizing + All Techniques)")
    print("=" * 60)
    print(f"Best MAE: {best_overall_mae:.2f} years")
    print("=" * 60)

    return best_overall_mae


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    best_mae = train_advanced()

    if best_mae:
        print("\n📊 FINAL COMPARISON OF ALL METHODS:")
        print("-" * 65)
        print(f"   Baseline (7-train_ipcti.py):                   10.11 years")
        print(f"   Enhancement C (Pretrained ResNet50):           6.10 years")
        print(f"   Enhancement F (MixUp + Aggressive Aug):        5.46 years")
        print(f"   Enhancement G (Progressive Resizing + Advanced): {best_mae:.2f} years")
        print("-" * 65)