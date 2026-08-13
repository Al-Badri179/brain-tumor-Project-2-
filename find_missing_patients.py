# check_patient_276.py
import os

DATASET_PATH = r"D:\PhD\My Articles\PhD\10-LMR-Trinity-A Liquid-Mamba Vision Network for Explainable\Data\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI\dataset"

# Check if patient 276 folder exists
folder_276 = os.path.join(DATASET_PATH, "0276")
print(f"Checking: {folder_276}")
print(f"Exists: {os.path.exists(folder_276)}")

if os.path.exists(folder_276):
    print(f"Contents: {os.listdir(folder_276)}")
else:
    # Check for folder without leading zeros
    folder_276_alt = os.path.join(DATASET_PATH, "276")
    print(f"\nChecking alternative: {folder_276_alt}")
    print(f"Exists: {os.path.exists(folder_276_alt)}")
    if os.path.exists(folder_276_alt):
        print(f"Contents: {os.listdir(folder_276_alt)}")

# Check what folders actually exist
print("\nSample of existing folders:")
folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
print(f"First 20: {sorted(folders)[:20]}")