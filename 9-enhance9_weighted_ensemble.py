# 9-enhance9_weighted_ensemble.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torchvision.models as tv_models  # Renamed to avoid conflict
from scipy.optimize import minimize

# ============================================================
# CONFIGURATION
# ============================================================
IPCTI_BASE = r"D:\PhD\My Articles\PhD\10-LMR-Trinity-A Liquid-Mamba Vision Network for Explainable\Data\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI"
DATASET_PATH = os.path.join(IPCTI_BASE, "dataset")
BATCH_SIZE = 16
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
            transforms.Resize((224, 224)),
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

        if img_path is None:
            raise FileNotFoundError(f"No image found for patient {patient_id}")

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return {'image': image, 'age': age}


# ============================================================
# MODEL (same architecture as training)
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
        # Use tv_models instead of models
        self.backbone = tv_models.resnet50(weights='IMAGENET1K_V2')
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
# WEIGHT OPTIMIZATION FUNCTION
# ============================================================
def optimize_weights(val_preds, val_truths):
    """Learn optimal weights for ensemble using scipy optimization"""
    val_preds = np.array(val_preds)  # [num_models, num_samples]
    val_truths = np.array(val_truths)

    def objective(weights):
        weights = np.abs(weights)  # Ensure non-negative
        weights = weights / weights.sum()  # Normalize to sum to 1
        ensemble_pred = np.average(val_preds, axis=0, weights=weights)
        mae = np.mean(np.abs(ensemble_pred - val_truths))
        return mae

    # Optimize weights
    initial_weights = np.ones(len(val_preds)) / len(val_preds)
    result = minimize(objective, initial_weights, method='Nelder-Mead',
                      options={'maxiter': 1000, 'disp': False})
    optimized_weights = np.abs(result.x)
    optimized_weights = optimized_weights / optimized_weights.sum()

    return optimized_weights


# ============================================================
# MAIN
# ============================================================
print("=" * 60)
print("WEIGHTED ENSEMBLE OF 3 MODELS")
print("=" * 60)

# Load test data
test_csv = os.path.join(IPCTI_BASE, "cross-validation", "cv_1", "split_1", "test.csv")
test_dataset = IPCTIDataset(test_csv, DATASET_PATH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"\nTest samples: {len(test_dataset)}")

# Load validation data for weight optimization (use train.csv)
val_csv = os.path.join(IPCTI_BASE, "cross-validation", "cv_1", "split_1", "train.csv")
val_dataset = IPCTIDataset(val_csv, DATASET_PATH)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Validation samples (for weight opt): {len(val_dataset)}")

# Load 3 trained models
loaded_models = []  # Renamed from 'models' to avoid conflict
seeds = [42, 123, 456]
model_paths = []

print("\nLoading models:")
for seed in seeds:
    model_path = f"best_model_seed_{seed}.pth"
    if os.path.exists(model_path):
        model = LMR_Trinity_Pretrained().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        loaded_models.append(model)
        model_paths.append(model_path)
        print(f"✅ Loaded: {model_path}")
    else:
        print(f"❌ Missing: {model_path}")

if len(loaded_models) != 3:
    print("\n❌ Not all models found. Please run 9-enhance8_train_models.py first.")
    exit()

print(f"\nLoaded {len(loaded_models)} models successfully.")

# ============================================================
# COLLECT VALIDATION PREDICTIONS FOR WEIGHT OPTIMIZATION
# ============================================================
print("\n📊 Collecting validation predictions for weight optimization...")

val_preds_per_model = [[] for _ in range(3)]
val_truths = []

with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        images = batch['image'].to(DEVICE)
        ages = batch['age'].to(DEVICE)

        for i, model in enumerate(loaded_models):
            preds = model(images)
            val_preds_per_model[i].extend(preds.cpu().numpy())

        val_truths.extend(ages.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"   Processed {batch_idx + 1} batches")

val_preds_per_model = np.array(val_preds_per_model)
print(f"Validation predictions shape: {val_preds_per_model.shape}")

# ============================================================
# OPTIMIZE WEIGHTS
# ============================================================
print("\n🔍 Optimizing ensemble weights...")
optimal_weights = optimize_weights(val_preds_per_model, val_truths)

print(f"\n📊 OPTIMAL WEIGHTS:")
for i, seed in enumerate(seeds):
    print(f"   Model {seed}: {optimal_weights[i]:.4f}")

# Calculate validation MAE with optimal weights
val_ensemble_pred = np.average(val_preds_per_model, axis=0, weights=optimal_weights)
val_mae = np.mean(np.abs(val_ensemble_pred - val_truths))
print(f"\nValidation MAE with optimal weights: {val_mae:.2f} years")

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
print("\n📈 Evaluating weighted ensemble on test set...")

# Store predictions from individual models
individual_preds = []
individual_maes = []

# Simple average ensemble
simple_ensemble_preds = []
# Weighted ensemble
weighted_ensemble_preds = []
all_truths = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        images = batch['image'].to(DEVICE)
        ages = batch['age'].to(DEVICE)

        batch_preds = []
        for i, model in enumerate(loaded_models):
            pred = model(images)
            pred_np = pred.cpu().numpy().flatten()
            batch_preds.append(pred_np)

            # Store for individual evaluation
            if batch_idx == 0:
                individual_preds.append([])
            individual_preds[i].extend(pred_np)

        # Simple average (equal weights)
        simple_avg = np.mean(batch_preds, axis=0)
        simple_ensemble_preds.extend(simple_avg.flatten())

        # Weighted average
        weighted_avg = np.average(batch_preds, axis=0, weights=optimal_weights)
        weighted_ensemble_preds.extend(weighted_avg.flatten())

        all_truths.extend(ages.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"   Processed {batch_idx + 1} batches")

# Calculate individual model MAEs
print("\n📊 Individual Model Performance on Test Set:")
for i, seed in enumerate(seeds):
    mae = mean_absolute_error(all_truths, individual_preds[i])
    individual_maes.append(mae)
    print(f"   Model {seed}: {mae:.2f} years")

# Calculate ensemble metrics
simple_mae = mean_absolute_error(all_truths, simple_ensemble_preds)
weighted_mae = mean_absolute_error(all_truths, weighted_ensemble_preds)
simple_rmse = np.sqrt(mean_squared_error(all_truths, simple_ensemble_preds))
weighted_rmse = np.sqrt(mean_squared_error(all_truths, weighted_ensemble_preds))
simple_r2 = r2_score(all_truths, simple_ensemble_preds)
weighted_r2 = r2_score(all_truths, weighted_ensemble_preds)

# Find best individual model
best_individual = min(individual_maes)
best_seed = seeds[individual_maes.index(best_individual)]

print("\n" + "=" * 60)
print("ENSEMBLE RESULTS")
print("=" * 60)
print(f"Best individual model (Seed {best_seed}):  {best_individual:.2f} years")
print(f"Simple Average Ensemble:                 {simple_mae:.2f} years")
print(f"Weighted Ensemble:                       {weighted_mae:.2f} years")
print("=" * 60)

if weighted_mae < best_individual:
    improvement = best_individual - weighted_mae
    percent = (improvement / best_individual) * 100
    print(f"\n✅ Improvement over best single model: {improvement:.2f} years ({percent:.1f}%)")
else:
    print(f"\n⚠️ Ensemble did not improve over best single model")

print("\n📊 DETAILED METRICS:")
print(f"{'Method':<30} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
print("-" * 60)
print(
    f"{'Best Individual (Seed ' + str(best_seed) + ')':<30} {best_individual:<10.2f} {individual_maes[individual_maes.index(best_individual)]:<10.2f} -")
print(f"{'Simple Average Ensemble':<30} {simple_mae:<10.2f} {simple_rmse:<10.2f} {simple_r2:<10.3f}")
print(f"{'Weighted Ensemble':<30} {weighted_mae:<10.2f} {weighted_rmse:<10.2f} {weighted_r2:<10.3f}")

# Save results
with open("weighted_ensemble_results.txt", "w") as f:
    f.write("Weighted Ensemble Results\n")
    f.write("=" * 50 + "\n")
    f.write(f"Optimal weights: {optimal_weights}\n")
    f.write(
        f"Individual models: Seed 42={individual_maes[0]:.2f}, Seed 123={individual_maes[1]:.2f}, Seed 456={individual_maes[2]:.2f}\n")
    f.write(f"Best individual MAE: {best_individual:.2f} years\n")
    f.write(f"Simple ensemble MAE: {simple_mae:.2f} years\n")
    f.write(f"Weighted ensemble MAE: {weighted_mae:.2f} years\n")
    f.write(f"Weighted ensemble RMSE: {weighted_rmse:.2f} years\n")
    f.write(f"Weighted ensemble R²: {weighted_r2:.3f}\n")

print("\n✅ Results saved to weighted_ensemble_results.txt")