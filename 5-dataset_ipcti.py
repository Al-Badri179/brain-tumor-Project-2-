# 5-dataset_ipcti.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from

4 - config_ipcti
import IMG_SIZE


class IPCTIDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        patient_id = str(row['id'])
        age = torch.tensor(row['age'], dtype=torch.float32)

        # Find image for this patient
        img_path = None
        patient_dir = os.path.join(self.images_dir, patient_id.zfill(4))

        if os.path.exists(patient_dir):
            for file in os.listdir(patient_dir):
                if file.endswith('.jpg'):
                    img_path = os.path.join(patient_dir, file)
                    break

        if img_path is None:
            # Search recursively
            for root, dirs, files in os.walk(self.images_dir):
                for file in files:
                    if file.startswith(patient_id.zfill(4)) and file.endswith('.jpg'):
                        img_path = os.path.join(root, file)
                        break
                if img_path:
                    break

        if img_path is None:
            raise FileNotFoundError(f"Image not found for patient {patient_id}")

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return {
            'image': image,
            'age': age,
            'patient_id': patient_id
        }


print(f"✅ Dataset class loaded")