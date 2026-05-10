import math
from collections import defaultdict

import numpy as np
import torch
from skimage.metrics import structural_similarity


def tensor_to_hwc_numpy(tensor):
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    return tensor.permute(1, 2, 0).numpy()


def compute_psnr(prediction, target):
    mse = torch.mean((prediction.detach().cpu() - target.detach().cpu()) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim(prediction, target):
    pred = tensor_to_hwc_numpy(prediction)
    gt = tensor_to_hwc_numpy(target)
    return float(structural_similarity(gt, pred, channel_axis=2, data_range=1.0))


def compute_lpips(prediction, target, lpips_model, device):
    pred = prediction.unsqueeze(0).to(device).clamp(0.0, 1.0) * 2.0 - 1.0
    gt = target.unsqueeze(0).to(device).clamp(0.0, 1.0) * 2.0 - 1.0
    with torch.no_grad():
        value = lpips_model(pred, gt)
    return float(value.mean().detach().cpu().item())


class MetricAccumulator:
    def __init__(self):
        self.values = defaultdict(list)

    def update(self, metrics):
        for key, value in metrics.items():
            if value is None:
                continue
            self.values[key].append(float(value))

    def summary(self):
        out = {}
        for key, values in self.values.items():
            arr = np.array(values, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            mean = float(arr.mean()) if len(finite) == len(arr) else float("inf")
            if key != "psnr" and len(finite) != len(arr):
                mean = float(finite.mean()) if len(finite) else None
            out[key] = {
                "mean": mean,
                "count": int(len(arr)),
            }
        return out


def compute_image_metrics(prediction, target, lpips_model=None, lpips_device=None):
    metrics = {
        "psnr": compute_psnr(prediction, target),
        "ssim": compute_ssim(prediction, target),
    }
    if lpips_model is not None:
        metrics["lpips"] = compute_lpips(prediction, target, lpips_model, lpips_device)
    return metrics


def normalized_coords_to_pixel(coords_norm, frame_geometries):
    """Convert normalized xy coordinates to original-image pixel coordinates."""
    coords_norm = coords_norm.detach().cpu().float()
    frame_geometries = frame_geometries.detach().cpu().float()
    scales = torch.stack(
        [frame_geometries[..., 1], frame_geometries[..., 0]],
        dim=-1,
    )
    return coords_norm * scales


def compute_trajectory_metrics(prediction_px, target_px):
    distances = torch.linalg.norm(
        prediction_px.detach().cpu().float() - target_px.detach().cpu().float(),
        dim=-1,
    )
    return {
        "ade_px": float(distances.mean().item()),
        "fde_px": float(distances[-1].item()),
    }
