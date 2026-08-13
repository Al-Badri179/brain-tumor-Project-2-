# read_ipcti_labels.py
import os
import pandas as pd

IPCTI_BASE = r"D:\PhD\My Articles\PhD\10-LMR-Trinity-A Liquid-Mamba Vision Network for Explainable\Data\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI"

# Find all CSV files
print("=" * 70)
print("📊 READING IPCTI CSV FILES")
print("=" * 70)

csv_files = []
for root, dirs, files in os.walk(IPCTI_BASE):
    for file in files:
        if file.endswith('.csv'):
            full_path = os.path.join(root, file)
            csv_files.append(full_path)
            print(f"\n📁 Found: {os.path.relpath(full_path, IPCTI_BASE)}")

            try:
                df = pd.read_csv(full_path)
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {list(df.columns)}")
                print(f"\n   First 5 rows:")
                print(df.head())

                # Check for age column
                age_columns = [col for col in df.columns if 'age' in col.lower()]
                if age_columns:
                    print(f"\n   ✅ AGE COLUMN FOUND: '{age_columns[0]}'")
                    print(f"   Age range: {df[age_columns[0]].min()} - {df[age_columns[0]].max()} years")
                    print(f"   Mean age: {df[age_columns[0]].mean():.1f} years")
                else:
                    print(f"\n   ⚠️ No obvious 'age' column. Columns available: {list(df.columns)}")

            except Exception as e:
                print(f"   Error reading: {e}")

print("\n" + "=" * 70)