# 8-evaluate_ipcti.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from

4 - config_ipcti
import *
from

5 - dataset_ipcti
import IPCTIDataset
from

6 - model_trinity_age
import LMR_Trinity_Age

print("=" * 70)
print("📊 FINAL EVALUATION")
print("=" * 70)

# Find test CSV
cv_base = os.path.join(IPCTI_BASE, "cross-validation")
test_csv = None

for fold in range(1, N_FOLDS + 1):
    fold_dir = os.path.join(cv_base, f"cv_{fold}")
    for split in range(1, N_SPLITS + 1):
        split_dir = os.path.join(fold_dir, f"split_{split}")
        potential_test = os.path.join(split_dir, "test.csv")
        if os.path.exists(potential_test):
            test_csv = potential_test
            break
    if test_csv:
        break

# Load test data
test_dataset = IPCTIDataset(test_csv, DATASET_PATH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = LMR_Trinity_Age(IMG_SIZE).to(DEVICE)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Predict
all_preds = []
all_truths = []

with torch.no_grad():
    for batch in test_loader:
        images = batch['image'].to(DEVICE)
        ages = batch['age'].to(DEVICE)

        outputs = model(images)
        all_preds.extend(outputs.cpu().numpy())
        all_truths.extend(ages.cpu().numpy())

# Calculate metrics
mae = mean_absolute_error(all_truths, all_preds)
rmse = np.sqrt(mean_squared_error(all_truths, all_preds))
r2 = r2_score(all_truths, all_preds)

print(f"\n{'=' * 70}")
print(f"📊 FINAL RESULTS FOR MANUSCRIPT")
print(f"{'=' * 70}")
print(f"Mean Absolute Error (MAE):  {mae:.2f} years")
print(f"Root Mean Square Error:     {rmse:.2f} years")
print(f"R² Score:                   {r2:.3f}")
print(f"{'=' * 70}")

# Plot predictions vs ground truth
plt.figure(figsize=(8, 6))
plt.scatter(all_truths, all_preds, alpha=0.5)
plt.plot([18, 60], [18, 60], 'r--', label='Perfect prediction')
plt.xlabel('Ground Truth Age (years)')
plt.ylabel('Predicted Age (years)')
plt.title(f'LMR-Trinity Age Estimation\nMAE = {mae:.2f} years, R² = {r2:.3f}')
plt.legend()
plt.savefig('age_estimation_results.png', dpi=300)
plt.show()

print(f"\n✅ Results saved to: age_estimation_results.png")