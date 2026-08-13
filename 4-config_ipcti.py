# 4-config_ipcti.py
import os
import torch

# Paths
IPCTI_BASE = r"D:\PhD\My Articles\PhD\10-LMR-Trinity-A Liquid-Mamba Vision Network for Explainable\Data\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI"
DATASET_PATH = os.path.join(IPCTI_BASE, "dataset")

# Training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 100
NUM_WORKERS = 4

# Image settings
IMG_SIZE = (224, 224)
D_MODEL = 128

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cross-validation
N_FOLDS = 5
N_SPLITS = 5

print(f"✅ Config loaded")
print(f"   Device: {DEVICE}")
print(f"   Image size: {IMG_SIZE}")
print(f"   Batch size: {BATCH_SIZE}")