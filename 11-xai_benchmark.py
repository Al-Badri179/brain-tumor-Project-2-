# 12-xai_metrics_benchmark.py
import os
import torch
import numpy as np
import pandas as pd
from captum.attr import (
    LayerGradCam, IntegratedGradients, GradientShap,
    Lime, GuidedBackprop, FeatureAblation
)
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize


# ============================================================
# XAI METRICS COMPUTATION
# ============================================================

def compute_faithfulness(attribution, model, input_tensor, target, n_samples=20):
    """Faithfulness: Correlation between importance and prediction change"""
    importances = attribution.flatten()

    # Get baseline prediction
    with torch.no_grad():
        base_pred = model(input_tensor.unsqueeze(0))[target].item()

    # Perturb features and measure change
    pred_changes = []
    sorted_indices = np.argsort(-importances)[:100]  # Top 100 features

    for idx in sorted_indices:
        perturbed = input_tensor.clone()
        # Flatten and perturb
        flat = perturbed.flatten()
        flat[idx] = 0
        perturbed = flat.reshape(perturbed.shape)

        with torch.no_grad():
            new_pred = model(perturbed.unsqueeze(0))[target].item()

        pred_changes.append(abs(base_pred - new_pred))

    # Correlation between importance and prediction change
    correlation, _ = pearsonr(importances[sorted_indices], pred_changes)
    return max(0, correlation)  # Ensure non-negative


def compute_stability(attribution_fn, model, input_tensor, target, noise_level=0.01):
    """Stability: Consistency under small input perturbations"""
    attributions = []

    for _ in range(5):
        # Add noise
        noisy_input = input_tensor + torch.randn_like(input_tensor) * noise_level
        attr = attribution_fn(noisy_input.unsqueeze(0), target=target)
        attributions.append(attr.cpu().numpy().flatten())

    # Compute variance across runs
    attributions = np.array(attributions)
    variance = np.var(attributions, axis=0).mean()
    return variance


def compute_sparsity(attribution, threshold=0.05):
    """Sparsity: Percentage of near-zero attributions"""
    attr_flat = attribution.flatten()
    n_important = np.sum(np.abs(attr_flat) > threshold * np.max(np.abs(attr_flat)))
    sparsity = 1 - (n_important / len(attr_flat))
    return sparsity


def compute_completeness(attribution, model, input_tensor, target, baseline=None):
    """Completeness: Sum of attributions approximates prediction change"""
    if baseline is None:
        baseline = torch.zeros_like(input_tensor)

    with torch.no_grad():
        pred_input = model(input_tensor.unsqueeze(0))[target].item()
        pred_baseline = model(baseline.unsqueeze(0))[target].item()

    attr_sum = attribution.sum().item()
    pred_diff = pred_input - pred_baseline
    completeness = 1 - abs(attr_sum - pred_diff) / (abs(pred_diff) + 1e-8)
    return max(0, min(1, completeness))


def compute_continuity(attribution_map):
    """Continuity: Smoothness of attribution map (lower = smoother)"""
    # Convert to 2D if needed
    if len(attribution_map.shape) > 2:
        attr_2d = np.mean(attribution_map, axis=0)
    else:
        attr_2d = attribution_map

    # Compute gradient magnitude
    gy, gx = np.gradient(attr_2d)
    continuity = np.mean(np.sqrt(gx ** 2 + gy ** 2))
    return continuity


def compute_repeatability(attribution_fn, model, input_tensor, target, n_runs=3):
    """Repeatability: Consistency across multiple runs"""
    attributions = []
    for _ in range(n_runs):
        attr = attribution_fn(input_tensor.unsqueeze(0), target=target)
        attributions.append(attr.cpu().numpy().flatten())

    attributions = np.array(attributions)
    # Compute pairwise SSIM
    repeatability = 0
    count = 0
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            sim = np.corrcoef(attributions[i], attributions[j])[0, 1]
            repeatability += sim
            count += 1

    return repeatability / count if count > 0 else 0


# ============================================================
# MAIN BENCHMARK WITH METRICS
# ============================================================

def run_xai_metrics_benchmark(model, test_loader, device):
    """Run full XAI metrics benchmark"""

    results = {
        'Method': [],
        'Faithfulness': [],
        'Stability': [],
        'Sparsity': [],
        'Completeness': [],
        'Continuity': [],
        'Repeatability': []
    }

    methods = {
        'Grad-CAM': lambda x, t: grad_cam.attribute(x, target=t),
        'Guided Grad-CAM': lambda x, t: guided_backprop.attribute(x, target=t),
        'SHAP': lambda x, t: gradient_shap.attribute(x, target=t),
        'LIME': lambda x, t: lime.attribute(x, target=t),
        'Integrated Gradients': lambda x, t: integrated_gradients.attribute(x, target=t),
        'Our Ensemble': lambda x, t: ensemble_attribution(x, t)
    }

    # Initialize attribution methods
    target_layer = model.backbone.layer4[2].conv3
    grad_cam = LayerGradCam(ModelWrapper(model), target_layer)
    guided_backprop = GuidedBackprop(ModelWrapper(model))
    gradient_shap = GradientShap(ModelWrapper(model))
    lime = Lime(ModelWrapper(model))
    integrated_gradients = IntegratedGradients(ModelWrapper(model))

    # Get sample
    sample = next(iter(test_loader))
    image = sample['image'].to(device)
    age = sample['age'].to(device)
    target = 0  # Regression output index

    for method_name, method_fn in methods.items():
        print(f"Evaluating {method_name}...")

        # Compute attribution
        attribution = method_fn(image, target)

        # Compute metrics
        faithfulness = compute_faithfulness(attribution, model, image, target)
        stability = compute_stability(method_fn, model, image, target)
        sparsity = compute_sparsity(attribution.cpu().numpy())
        completeness = compute_completeness(attribution, model, image, target)
        continuity = compute_continuity(attribution.cpu().numpy())
        repeatability = compute_repeatability(method_fn, model, image, target)

        results['Method'].append(method_name)
        results['Faithfulness'].append(faithfulness)
        results['Stability'].append(stability)
        results['Sparsity'].append(sparsity)
        results['Completeness'].append(completeness)
        results['Continuity'].append(continuity)
        results['Repeatability'].append(repeatability)

    return pd.DataFrame(results)