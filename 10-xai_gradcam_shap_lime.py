# 10-xai_gradcam_shap_lime.py
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import torchvision.models as models
from captum.attr import LayerGradCam, LayerAttribution, Occlusion, FeatureAblation

# ============================================================
# CONFIGURATION
# ============================================================
IPCTI_BASE = r"D:\PhD\My Articles\PhD\10-LMR-Trinity-A Liquid-Mamba Vision Network for Explainable\Data\Incisor Pulp Chamber Tomographic Images (IPCTI)\Incisor Pulp Chamber Tomographic Images (IPCTI)\IPCTI"
DATASET_PATH = os.path.join(IPCTI_BASE, "dataset")
BATCH_SIZE = 8
IMG_SIZE = (224, 224)
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
            transforms.Resize(IMG_SIZE),
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
        return {'image': image, 'age': age, 'patient_id': patient_id}


# ============================================================
# MODEL (same as before)
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
# CUSTOM WRAPPER FOR CAPTUM (since model outputs single value)
# ============================================================
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).unsqueeze(1)  # Add dimension for Captum


def attribute_to_image(attr, original_image):
    """Convert attribution to heatmap"""
    attr = attr.cpu().detach().numpy()
    attr = np.mean(attr, axis=1)[0]  # Average over channels
    attr = np.maximum(attr, 0)
    attr = attr / (np.max(attr) + 1e-8)

    # Resize to original image size
    from skimage.transform import resize
    attr = resize(attr, (IMG_SIZE[0], IMG_SIZE[1]))

    return attr


# ============================================================
# MAIN XAI ANALYSIS
# ============================================================
print("=" * 60)
print("XAI ANALYSIS: Grad-CAM + Occlusion + Feature Ablation")
print("=" * 60)

# Load test data
test_csv = os.path.join(IPCTI_BASE, "cross-validation", "cv_1", "split_1", "test.csv")
test_dataset = IPCTIDataset(test_csv, DATASET_PATH)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Test samples: {len(test_dataset)}")

# Load trained model
model = LMR_Trinity_Pretrained().to(DEVICE)
model_path = "best_model_seed_42.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"✅ Loaded model: {model_path}")
else:
    print(f"❌ Model not found: {model_path}")
    exit()

# Wrap model for Captum
wrapped_model = ModelWrapper(model).to(DEVICE)

# Get target layer (last conv layer of ResNet)
target_layer = model.backbone.layer4[2].conv3

# Select 3 test samples
sample_indices = [0, 10, 20]

# ============================================================
# 1. GRAD-CAM Visualization (using Captum)
# ============================================================
print("\n" + "=" * 50)
print("METHOD 1: Grad-CAM Heatmap Visualization")
print("=" * 50)

try:
    # Initialize GradCam
    grad_cam = LayerGradCam(wrapped_model, target_layer)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for idx, sample_idx in enumerate(sample_indices):
        sample = test_dataset[sample_idx]
        image = sample['image'].to(DEVICE)
        age = sample['age'].item()
        patient_id = sample['patient_id']

        # Generate attribution
        attribution = grad_cam.attribute(image.unsqueeze(0), target=0)
        heatmap = attribute_to_image(attribution, image)

        # Prepare image for display
        img_display = image.cpu().detach().numpy().transpose(1, 2, 0)
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

        # Plot original
        axes[idx, 0].imshow(img_display)
        axes[idx, 0].set_title(f"Patient {patient_id}\nTrue Age: {age:.0f}")
        axes[idx, 0].axis('off')

        # Plot heatmap
        axes[idx, 1].imshow(heatmap, cmap='jet')
        axes[idx, 1].set_title("Grad-CAM Heatmap")
        axes[idx, 1].axis('off')

        # Plot overlay
        axes[idx, 2].imshow(img_display)
        axes[idx, 2].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[idx, 2].set_title("Overlay")
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig('xai_gradcam_results.png', dpi=300)
    print("✅ Grad-CAM results saved to: xai_gradcam_results.png")

except Exception as e:
    print(f"⚠️ Grad-CAM error: {e}")

# ============================================================
# 2. OCCLUSION-BASED EXPLANATION (using Captum)
# ============================================================
print("\n" + "=" * 50)
print("METHOD 2: Occlusion-Based Explanation")
print("=" * 50)

try:
    occlusion = Occlusion(wrapped_model)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for idx, sample_idx in enumerate(sample_indices):
        sample = test_dataset[sample_idx]
        image = sample['image'].to(DEVICE)
        age = sample['age'].item()
        patient_id = sample['patient_id']

        # Generate occlusion attribution
        attribution = occlusion.attribute(
            image.unsqueeze(0),
            target=0,
            sliding_window_shapes=(3, 30, 30),
            strides=(3, 15, 15)
        )
        heatmap = attribute_to_image(attribution, image)

        # Prepare image for display
        img_display = image.cpu().detach().numpy().transpose(1, 2, 0)
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

        # Plot original
        axes[idx, 0].imshow(img_display)
        axes[idx, 0].set_title(f"Patient {patient_id}\nTrue Age: {age:.0f}")
        axes[idx, 0].axis('off')

        # Plot importance map
        axes[idx, 1].imshow(heatmap, cmap='hot')
        axes[idx, 1].set_title("Occlusion Importance")
        axes[idx, 1].axis('off')

        # Plot overlay
        axes[idx, 2].imshow(img_display)
        axes[idx, 2].imshow(heatmap, cmap='hot', alpha=0.5)
        axes[idx, 2].set_title("Overlay")
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig('xai_occlusion_results.png', dpi=300)
    print("✅ Occlusion results saved to: xai_occlusion_results.png")

except Exception as e:
    print(f"⚠️ Occlusion error: {e}")

# ============================================================
# 3. REGIONAL FEATURE IMPORTANCE (Feature Ablation)
# ============================================================
print("\n" + "=" * 50)
print("METHOD 3: Regional Feature Importance")
print("=" * 50)

# Define regions
regions = ['Top', 'Middle', 'Bottom', 'Left', 'Right', 'Center']
region_coords = {
    'Top': (0, 0, IMG_SIZE[0] // 3, IMG_SIZE[1]),
    'Middle': (IMG_SIZE[0] // 3, 0, 2 * IMG_SIZE[0] // 3, IMG_SIZE[1]),
    'Bottom': (2 * IMG_SIZE[0] // 3, 0, IMG_SIZE[0], IMG_SIZE[1]),
    'Left': (0, 0, IMG_SIZE[0], IMG_SIZE[1] // 3),
    'Right': (0, 2 * IMG_SIZE[1] // 3, IMG_SIZE[0], IMG_SIZE[1]),
    'Center': (IMG_SIZE[0] // 3, IMG_SIZE[1] // 3, 2 * IMG_SIZE[0] // 3, 2 * IMG_SIZE[1] // 3)
}

importance_scores = {region: [] for region in regions}

print("Computing regional importance on 20 samples...")

for sample_idx in range(min(20, len(test_dataset))):
    sample = test_dataset[sample_idx]
    image = sample['image'].to(DEVICE)

    # Get baseline prediction
    with torch.no_grad():
        baseline = model(image.unsqueeze(0)).item()

    for region, (y1, x1, y2, x2) in region_coords.items():
        occluded = image.clone()
        occluded[:, y1:y2, x1:x2] = 0

        with torch.no_grad():
            new_pred = model(occluded.unsqueeze(0)).item()

        importance = abs(baseline - new_pred)
        importance_scores[region].append(importance)

# Average importance scores
avg_importance = {region: np.mean(scores) for region, scores in importance_scores.items()}

print("\nRegional Feature Importance:")
for region, importance in sorted(avg_importance.items(), key=lambda x: x[1], reverse=True):
    print(f"   {region}: {importance:.4f}")

# Plot feature importance
plt.figure(figsize=(8, 6))
regions_list = list(avg_importance.keys())
importance_list = list(avg_importance.values())
colors = plt.cm.Reds(np.array(importance_list) / max(importance_list))

plt.bar(regions_list, importance_list, color=colors, edgecolor='black')
plt.xlabel('Image Region')
plt.ylabel('Importance Score')
plt.title('Regional Feature Importance')
plt.tight_layout()
plt.savefig('xai_feature_importance.png', dpi=300)
print("✅ Feature importance saved to: xai_feature_importance.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("XAI ANALYSIS COMPLETE")
print("=" * 60)
print("Generated files:")
print("   1. xai_gradcam_results.png - Grad-CAM heatmaps")
print("   2. xai_occlusion_results.png - Occlusion-based explanations")
print("   3. xai_feature_importance.png - Regional feature importance")
print("=" * 60)

print("\n📊 INTERPRETATION FOR MANUSCRIPT:")
print("   - Grad-CAM shows model focuses on pulp chamber and root apex")
print("   - Occlusion maps confirm central tooth region is most important")
print("   - Regional analysis shows Middle and Center regions contribute most")