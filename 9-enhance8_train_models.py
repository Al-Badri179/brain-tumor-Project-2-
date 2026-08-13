# 9-enhance8_train_models.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import mean_absolute_error
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
EPOCHS = 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIXUP_ALPHA = 0.4

# Different seeds for different models
SEEDS = [42, 123, 456]

print(f"Device: {DEVICE}")

# Get existing patient folders
existing_folders_str = set()
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if os.path.isdir(folder_path):
        existing_folders_str.add(folder)
print(f"Existing folders: {len(existing_folders_str)}")


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
    def __init__(self, csv_file, images_dir, augment=True):
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

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=25),
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
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

        patient_dir = os.path.join(self.images_dir, folder_name)
        img_path = None
        for file in os.listdir(patient_dir):
            if file.endswith('.jpg'):
                img_path = os.path.join(patient_dir, file)
                break

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
# TRAINING FUNCTION
# ============================================================
def train_model_with_seed(seed, model_name):
    print(f"\n{'=' * 50}")
    print(f"Training Model with SEED = {seed}")
    print(f"{'=' * 50}")

    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load data
    train_csv = os.path.join(IPCTI_BASE, "cross-validation", "cv_1", "split_1", "train.csv")
    test_csv = os.path.join(IPCTI_BASE, "cross-validation", "cv_1", "split_1", "test.csv")

    train_dataset = IPCTIDatasetAugmented(train_csv, DATASET_PATH, augment=True)
    test_dataset = IPCTIDatasetAugmented(test_csv, DATASET_PATH, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = LMR_Trinity_Pretrained().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.L1Loss()

    best_mae = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            images = batch['image'].to(DEVICE)
            ages = batch['age'].to(DEVICE)

            images, ages_a, ages_b, lam = mixup_data(images, ages, MIXUP_ALPHA)

            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, ages_a, ages_b, lam)
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

        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), f"best_model_seed_{seed}.pth")

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}: MAE = {mae:.2f} years")

    print(f"\n✅ Best MAE for seed {seed}: {best_mae:.2f} years")
    return best_mae


# ============================================================
# MAIN
# ============================================================
print("=" * 60)
print("TRAINING 3 MODELS WITH DIFFERENT SEEDS")
print("=" * 60)

results = []
for seed in SEEDS:
    mae = train_model_with_seed(seed, f"seed_{seed}")
    results.append(mae)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Seed 42 MAE: {results[0]:.2f} years")
print(f"Seed 123 MAE: {results[1]:.2f} years")
print(f"Seed 456 MAE: {results[2]:.2f} years")
print(f"Average MAE: {np.mean(results):.2f} years")