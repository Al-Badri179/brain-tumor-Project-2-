# 6-model_trinity_age.py
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = F.silu(x)
        x = self.out_proj(x)
        return x + res


class LMR_Trinity_Age(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=16):
        super().__init__()

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, 64, kernel_size=patch_size, stride=patch_size)

        # Mamba blocks
        self.mamba1 = LightMambaBlock(64)
        self.mamba2 = LightMambaBlock(64)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Regression head
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        x = self.mamba1(x)
        x = self.mamba2(x)

        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        age = self.fc3(x)

        return age.squeeze(-1)


print(f"✅ LMR-Trinity model loaded")