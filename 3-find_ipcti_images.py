# find_ipcti_images.py
import os

IPCTI_BASE = r"D:\PhD\My Articles\PhD\10-LMR-Trinity-A Liquid-Mamba Vision Network for Explainable\Data\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI"

print("=" * 70)
print("🔍 LOCATING IPCTI IMAGE FILES")
print("=" * 70)

image_count = 0
image_paths = []

for root, dirs, files in os.walk(IPCTI_BASE):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            full_path = os.path.join(root, file)
            image_paths.append(full_path)
            image_count += 1
            if image_count <= 20:
                print(f"   {os.path.relpath(full_path, IPCTI_BASE)}")

print(f"\n📊 Total images found: {image_count}")

if image_count == 0:
    print("\n⚠️ No images found in the IPCTI folder!")
    print("   The images may be stored elsewhere or need to be extracted.")
    print("   Check if there are zip files in the dataset folder.")
else:
    print("\n✅ Images found! Ready to proceed.")

print("\n" + "=" * 70)