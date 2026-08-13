# fix_test_csv.py
import os
import pandas as pd

IPCTI_BASE = r"D:\PhD\My Articles\PhD\10-LMR-Trinity-A Liquid-Mamba Vision Network for Explainable\Data\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI"
DATASET_PATH = os.path.join(IPCTI_BASE, "dataset")

# Get existing folders (patients with images)
existing_folders = set()
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if os.path.isdir(folder_path) and folder.isdigit():
        existing_folders.add(int(folder))

print(f"Existing patient folders: {len(existing_folders)}")
print(f"Sample folders: {sorted(existing_folders)[:10]}...")

# Find and filter test CSV
cv_base = os.path.join(IPCTI_BASE, "cross-validation")

for fold in range(1, 6):
    for split in range(1, 6):
        split_dir = os.path.join(cv_base, f"cv_{fold}", f"split_{split}")
        test_csv = os.path.join(split_dir, "test.csv")

        if os.path.exists(test_csv):
            df = pd.read_csv(test_csv)
            original_count = len(df)

            # Filter to only existing patients
            df_filtered = df[df['id'].isin(existing_folders)]
            filtered_count = len(df_filtered)

            # Save filtered version (NEW FILE - original untouched)
            filtered_path = test_csv.replace('.csv', '_filtered.csv')
            df_filtered.to_csv(filtered_path, index=False)

            print(f"\n📁 {test_csv}")
            print(f"   Original: {original_count} patients")
            print(f"   Filtered: {filtered_count} patients")
            print(f"   Removed: {original_count - filtered_count} patients (missing images)")
            print(f"   ✅ Created: {filtered_path}")

            # Show removed patients
            removed = set(df['id'].values) - existing_folders
            if removed:
                print(f"   Removed IDs: {sorted(removed)[:10]}...")

print("\n✅ Done! Original files unchanged. New '_filtered.csv' files created.")
