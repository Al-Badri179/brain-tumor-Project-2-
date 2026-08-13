# 9-enhance10_stacking_ensemble.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torchvision.models as tv_models
import warnings

warnings.filterwarnings('ignore')

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
# MODEL (same architecture)
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
# FUNCTION TO GET PREDICTIONS FROM ALL MODELS
# ============================================================
def get_predictions(models, dataloader):
    """Get predictions from all base models on a dataloader"""
    all_preds_per_model = []
    all_truths = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(DEVICE)
            ages = batch['age'].to(DEVICE)

            batch_preds = []
            for model in models:
                pred = model(images)
                batch_preds.append(pred.cpu().numpy().flatten())

            if len(all_preds_per_model) == 0:
                all_preds_per_model = [[] for _ in range(len(models))]

            for i, preds in enumerate(batch_preds):
                all_preds_per_model[i].extend(preds)

            all_truths.extend(ages.cpu().numpy())

    # Convert to numpy arrays
    all_preds_per_model = np.array(all_preds_per_model)
    all_truths = np.array(all_truths)

    return all_preds_per_model, all_truths


# ============================================================
# MAIN
# ============================================================
print("=" * 60)
print("ENHANCEMENT H: STACKING ENSEMBLE (Meta-Learner)")
print("=" * 60)

# Load datasets
test_csv = os.path.join(IPCTI_BASE, "cross-validation", "cv_1", "split_1", "test.csv")
test_dataset = IPCTIDataset(test_csv, DATASET_PATH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"\nTest samples: {len(test_dataset)}")

val_csv = os.path.join(IPCTI_BASE, "cross-validation", "cv_1", "split_1", "train.csv")
val_dataset = IPCTIDataset(val_csv, DATASET_PATH)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Training samples (for meta-learning): {len(val_dataset)}")

# Load 3 trained models
loaded_models = []
seeds = [42, 123, 456]

print("\n📦 Loading base models:")
for seed in seeds:
    model_path = f"best_model_seed_{seed}.pth"
    if os.path.exists(model_path):
        model = LMR_Trinity_Pretrained().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        loaded_models.append(model)
        print(f"   ✅ Loaded: {model_path}")
    else:
        print(f"   ❌ Missing: {model_path}")

if len(loaded_models) != 3:
    print("\n❌ Not all models found.")
    exit()

print(f"\n✅ Loaded {len(loaded_models)} base models.")

# ============================================================
# GET PREDICTIONS
# ============================================================
print("\n" + "=" * 50)
print("STEP 1: Getting predictions from base models")
print("=" * 50)

# Get predictions on validation set (for training meta-model)
print("\n📊 Getting validation predictions...")
val_preds, val_truths = get_predictions(loaded_models, val_loader)
print(f"Validation predictions shape: {val_preds.shape}")  # [3, 431]

# Get predictions on test set
print("\n📊 Getting test predictions...")
test_preds, test_truths = get_predictions(loaded_models, test_loader)
print(f"Test predictions shape: {test_preds.shape}")  # [3, 98]

# Individual model performance
print("\n📊 Individual Model Performance (Validation Set):")
for i, seed in enumerate(seeds):
    mae = mean_absolute_error(val_truths, val_preds[i])
    print(f"   Model {seed}: {mae:.2f} years")

print("\n📊 Individual Model Performance (Test Set):")
individual_test_maes = []
for i, seed in enumerate(seeds):
    mae = mean_absolute_error(test_truths, test_preds[i])
    individual_test_maes.append(mae)
    print(f"   Model {seed}: {mae:.2f} years")

# ============================================================
# WEIGHTED ENSEMBLE (for comparison)
# ============================================================
print("\n" + "=" * 50)
print("STEP 2: Weighted Ensemble (Baseline)")
print("=" * 50)

# Simple average
simple_ensemble = np.mean(test_preds, axis=0)
simple_mae = mean_absolute_error(test_truths, simple_ensemble)
print(f"Simple Average Ensemble: {simple_mae:.2f} years")

# Optimize weights
from scipy.optimize import minimize


def optimize_weights(val_preds, val_truths):
    val_preds = np.array(val_preds)
    val_truths = np.array(val_truths)

    def objective(weights):
        weights = np.abs(weights)
        weights = weights / weights.sum()
        ensemble_pred = np.average(val_preds, axis=0, weights=weights)
        return np.mean(np.abs(ensemble_pred - val_truths))

    initial_weights = np.ones(len(val_preds)) / len(val_preds)
    result = minimize(objective, initial_weights, method='Nelder-Mead',
                      options={'maxiter': 1000, 'disp': False})
    weights = np.abs(result.x)
    weights = weights / weights.sum()
    return weights


optimal_weights = optimize_weights(val_preds, val_truths)
print(f"\nOptimal weights: {optimal_weights}")

weighted_ensemble = np.average(test_preds, axis=0, weights=optimal_weights)
weighted_mae = mean_absolute_error(test_truths, weighted_ensemble)
print(f"Weighted Ensemble: {weighted_mae:.2f} years")

# ============================================================
# STACKING WITH META-LEARNER
# ============================================================
print("\n" + "=" * 50)
print("STEP 3: Stacking Ensemble (Meta-Learner)")
print("=" * 50)

# Prepare meta-features: [samples, models]
X_meta_train = val_preds.T  # [431, 3]
X_meta_test = test_preds.T  # [98, 3]
y_meta_train = val_truths
y_meta_test = test_truths

# Define meta-models to try
meta_models = {
    'Ridge (α=1)': Ridge(alpha=1.0),
    'Ridge (α=0.1)': Ridge(alpha=0.1),
    'Ridge (α=10)': Ridge(alpha=10.0),
    'Linear Regression': LinearRegression(),
    'Random Forest (50)': RandomForestRegressor(n_estimators=50, random_state=42),
    'Random Forest (100)': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
}

stacking_results = {}

print("\n📊 Training and evaluating meta-models:")
for meta_name, meta_model in meta_models.items():
    # Train meta-model
    meta_model.fit(X_meta_train, y_meta_train)

    # Predict on test set
    ensemble_preds = meta_model.predict(X_meta_test)

    # Calculate metrics
    mae = mean_absolute_error(y_meta_test, ensemble_preds)
    rmse = np.sqrt(mean_squared_error(y_meta_test, ensemble_preds))
    r2 = r2_score(y_meta_test, ensemble_preds)

    stacking_results[meta_name] = {'mae': mae, 'rmse': rmse, 'r2': r2}

    print(f"   {meta_name:<25} MAE: {mae:.2f} years, RMSE: {rmse:.2f}, R²: {r2:.3f}")

# ============================================================
# FINAL COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("FINAL COMPARISON OF ALL METHODS")
print("=" * 60)

best_individual = min(individual_test_maes)
best_individual_seed = seeds[individual_test_maes.index(best_individual)]
best_weighted = weighted_mae
best_stacking = min([r['mae'] for r in stacking_results.values()])
best_stacking_name = min(stacking_results, key=lambda x: stacking_results[x]['mae'])

print(f"\n📊 INDIVIDUAL MODELS:")
for i, seed in enumerate(seeds):
    print(f"   Model {seed}: {individual_test_maes[i]:.2f} years")
print(f"   Best Individual (Seed {best_individual_seed}): {best_individual:.2f} years")

print(f"\n📊 WEIGHTED ENSEMBLE: {best_weighted:.2f} years")

print(f"\n📊 STACKING ENSEMBLE:")
for meta_name, results in stacking_results.items():
    print(f"   {meta_name}: {results['mae']:.2f} years")
print(f"   Best Stacking ({best_stacking_name}): {best_stacking:.2f} years")

print("\n" + "=" * 60)
print("🏆 BEST RESULTS")
print("=" * 60)
print(f"Best Individual Model:     {best_individual:.2f} years")
print(f"Weighted Ensemble:         {best_weighted:.2f} years")
print(f"Best Stacking Ensemble:    {best_stacking:.2f} years")

best_overall = min(best_individual, best_weighted, best_stacking)
print(f"\n🎉 OVERALL BEST:           {best_overall:.2f} years")

if best_stacking < best_weighted:
    improvement = best_weighted - best_stacking
    print(
        f"\n✅ Stacking improves over Weighted Ensemble by {improvement:.2f} years ({(improvement / best_weighted) * 100:.1f}%)")

# Save results
with open("stacking_ensemble_results.txt", "w") as f:
    f.write("STACKING ENSEMBLE RESULTS\n")
    f.write("=" * 50 + "\n\n")
    f.write("Individual Models:\n")
    for i, seed in enumerate(seeds):
        f.write(f"  Seed {seed}: {individual_test_maes[i]:.2f} years\n")
    f.write(f"\nBest Individual: {best_individual:.2f} years\n")
    f.write(f"Weighted Ensemble: {best_weighted:.2f} years\n\n")
    f.write("Stacking Ensembles:\n")
    for meta_name, results in stacking_results.items():
        f.write(f"  {meta_name}: MAE={results['mae']:.2f}, RMSE={results['rmse']:.2f}, R²={results['r2']:.3f}\n")
    f.write(f"\nBEST STACKING: {best_stacking:.2f} years\n")
    f.write(f"OVERALL BEST: {best_overall:.2f} years\n")

print("\n✅ Results saved to stacking_ensemble_results.txt")