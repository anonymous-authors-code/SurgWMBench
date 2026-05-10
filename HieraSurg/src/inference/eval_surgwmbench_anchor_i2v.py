import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio
import numpy as np
import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXTransformer3DModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from finetune.dataset.surgwmbench_anchor_dataset import (
    SurgWMBenchAnchorDataset,
    latent_frame_count,
    surgwmbench_anchor_collate,
)
from finetune.trajectory_head import SurgWMBenchTrajectoryHead, trajectory_checkpoint_file


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HieraSurg on SurgWMBench 20-anchor prediction.")
    parser.add_argument("--pretrained_model_name_or_path", required=True, help="Base CogVideoX/HieraSurg model path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir containing transformer/ or a transformer dir.")
    parser.add_argument("--trajectory-checkpoint", default=None, help="Optional path to trajectory_head.pt or its parent dir.")
    parser.add_argument("--disable_trajectory_head", action="store_true")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", default="manifests/val.jsonl")
    parser.add_argument("--output_dir", default="outputs/surgwmbench_anchor_i2v_eval")
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--context-anchors", type=int, default=5)
    parser.add_argument("--prediction-anchors", type=int, default=15)
    parser.add_argument("--model-num-frames", type=int, default=33)
    parser.add_argument("--eval-horizons", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--max-videos", type=int, default=8)
    parser.add_argument("--enable_slicing", action="store_true")
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument(
        "--inference-coord-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise std applied to normalized context coords at inference (trajectory head only).",
    )
    parser.add_argument(
        "--inference-coord-mask-prob",
        type=float,
        default=0.0,
        help="Per-context-point random mask probability applied at inference (trajectory head only).",
    )
    return parser.parse_args()


def augment_context_coords(
    context_coords_norm: torch.Tensor,
    noise_std: float,
    mask_prob: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if noise_std < 0:
        raise ValueError("--inference-coord-noise-std must be non-negative.")
    if mask_prob < 0 or mask_prob > 1:
        raise ValueError("--inference-coord-mask-prob must be in [0, 1].")

    coords = context_coords_norm
    if noise_std > 0:
        noise = torch.randn(coords.shape, generator=generator, device=coords.device, dtype=coords.dtype)
        coords = coords + noise * noise_std
        coords = coords.clamp(0, 1)

    mask = torch.ones(coords.shape[:2], device=coords.device, dtype=coords.dtype)
    if mask_prob > 0:
        rand = torch.rand(coords.shape[:2], generator=generator, device=coords.device, dtype=coords.dtype)
        mask = (rand >= mask_prob).to(dtype=coords.dtype)
    coords = coords * mask.unsqueeze(-1)
    return coords, mask


def transformer_checkpoint_path(checkpoint: str) -> Path:
    path = Path(checkpoint)
    if (path / "transformer").is_dir():
        return path / "transformer"
    return path


def resolve_trajectory_checkpoint(args: argparse.Namespace) -> Path:
    if args.trajectory_checkpoint:
        return trajectory_checkpoint_file(args.trajectory_checkpoint)
    checkpoint = Path(args.checkpoint)
    if checkpoint.name == "transformer":
        checkpoint = checkpoint.parent
    return trajectory_checkpoint_file(checkpoint)


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int,
    patch_size: int,
    attention_head_dim: int,
    device: torch.device,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)
    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )
    return freqs_cos.to(device=device), freqs_sin.to(device=device)


def encode_video_latents(vae: AutoencoderKLCogVideoX, video: torch.Tensor) -> torch.Tensor:
    video = video.to(device=vae.device, dtype=vae.dtype)
    encoded = vae.encode(video)
    latent_dist = encoded.latent_dist if hasattr(encoded, "latent_dist") else encoded[0]
    return latent_dist.sample() * vae.config.scaling_factor


@torch.no_grad()
def encode_padded_context_latents(
    vae: AutoencoderKLCogVideoX,
    context_frames: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    repeat_count = args.model_num_frames - args.context_anchors
    pad = context_frames[:, :, -1:].repeat(1, 1, repeat_count, 1, 1)
    conditioning_video = torch.cat([context_frames, pad], dim=2).to(device=device, dtype=torch.float32)
    context_latents = encode_video_latents(vae, conditioning_video).to(dtype=dtype).permute(0, 2, 1, 3, 4)
    context_latent_frames = latent_frame_count(args.context_anchors, vae.config.temporal_compression_ratio)
    return context_latents[:, :context_latent_frames]


def decode_video_latents(vae: AutoencoderKLCogVideoX, latents: torch.Tensor) -> torch.Tensor:
    latents = latents.permute(0, 2, 1, 3, 4)
    latents = latents / vae.config.scaling_factor
    return vae.decode(latents.to(dtype=vae.dtype)).sample


def zero_prompt_embeds(batch_size: int, model_config, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(
        (batch_size, model_config.max_text_seq_length, getattr(model_config, "text_embed_dim", 4096)),
        device=device,
        dtype=dtype,
    )


def prepare_extra_step_kwargs(scheduler, generator):
    import inspect

    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    return {"generator": generator} if accepts_generator else {}


@torch.no_grad()
def generate_future_anchors(
    transformer: CogVideoXTransformer3DModel,
    vae: AutoencoderKLCogVideoX,
    scheduler: CogVideoXDPMScheduler,
    context_frames: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    batch_size = context_frames.shape[0]
    context_video = context_frames
    repeat_count = args.model_num_frames - args.context_anchors
    pad = context_frames[:, :, -1:].repeat(1, 1, repeat_count, 1, 1)
    conditioning_video = torch.cat([context_video, pad], dim=2).to(device=device, dtype=torch.float32)

    context_latents = encode_video_latents(vae, conditioning_video).to(dtype=dtype).permute(0, 2, 1, 3, 4)
    context_latent_frames = latent_frame_count(args.context_anchors, vae.config.temporal_compression_ratio)
    latents = randn_tensor(context_latents.shape, generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma
    latents[:, :context_latent_frames] = context_latents[:, :context_latent_frames]

    prompt_embeds = zero_prompt_embeds(batch_size, transformer.config, device, dtype)
    timesteps, _ = retrieve_timesteps(scheduler, args.num_inference_steps, device, None)
    extra_step_kwargs = prepare_extra_step_kwargs(scheduler, generator)
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    image_rotary_emb = (
        prepare_rotary_positional_embeddings(
            height=args.height,
            width=args.width,
            num_frames=latents.shape[1],
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            patch_size=transformer.config.patch_size,
            attention_head_dim=transformer.config.attention_head_dim,
            device=device,
        )
        if transformer.config.use_rotary_positional_embeddings
        else None
    )

    old_pred_original_sample = None
    for i, t in enumerate(tqdm(timesteps, desc="Denoising", leave=False)):
        latent_model_input = scheduler.scale_model_input(latents, t)
        latent_model_input[:, :context_latent_frames] = context_latents[:, :context_latent_frames]
        timestep = t.expand(batch_size)
        noise_pred = transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0].float()

        if not isinstance(scheduler, CogVideoXDPMScheduler):
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        else:
            latents, old_pred_original_sample = scheduler.step(
                noise_pred,
                old_pred_original_sample,
                t,
                timesteps[i - 1] if i > 0 else None,
                latents,
                **extra_step_kwargs,
                return_dict=False,
            )
        latents = latents.to(dtype)
        latents[:, :context_latent_frames] = context_latents[:, :context_latent_frames]

    decoded = decode_video_latents(vae, latents).clamp(-1, 1)
    return decoded[:, :, args.context_anchors : args.context_anchors + args.prediction_anchors]


def tensor_to_uint8(frame: torch.Tensor, size: Tuple[int, int]) -> np.ndarray:
    frame = ((frame.float().cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
    image = Image.fromarray((frame * 255).astype(np.uint8))
    image = image.resize((size[1], size[0]), Image.Resampling.BICUBIC)
    return np.asarray(image).astype(np.uint8)


def load_original_uint8(dataset_root: str, relative_path: str) -> np.ndarray:
    with Image.open(Path(dataset_root) / relative_path) as image:
        return np.asarray(image.convert("RGB")).astype(np.uint8)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))


def simple_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = ((pred - mu_x) ** 2).mean()
    sigma_y = ((target - mu_y) ** 2).mean()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
    value = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(value.item())


def make_lpips(device: torch.device):
    try:
        import lpips

        return lpips.LPIPS(net="alex").to(device).eval()
    except Exception:
        return None


def compute_frame_metrics(
    pred_uint8: np.ndarray,
    target_uint8: np.ndarray,
    lpips_model,
    device: torch.device,
) -> Dict[str, Optional[float]]:
    pred = torch.from_numpy(pred_uint8).float().permute(2, 0, 1) / 255.0
    target = torch.from_numpy(target_uint8).float().permute(2, 0, 1) / 255.0
    result: Dict[str, Optional[float]] = {
        "psnr": psnr(pred, target),
        "ssim": simple_ssim(pred, target),
        "lpips": None,
    }
    if lpips_model is not None:
        with torch.no_grad():
            pred_lp = pred.unsqueeze(0).to(device) * 2 - 1
            target_lp = target.unsqueeze(0).to(device) * 2 - 1
            result["lpips"] = float(lpips_model(pred_lp, target_lp).item())
    return result


def save_video(path: Path, frames: List[np.ndarray], fps: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=10) as writer:
        for frame in frames:
            writer.append_data(frame)


def mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def coords_norm_to_px(coords_norm: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
    coords = coords_norm.float().cpu().numpy().copy()
    height, width = original_size
    coords[:, 0] *= width
    coords[:, 1] *= height
    return coords


def compute_trajectory_metrics(
    pred_coords_px: np.ndarray,
    target_coords_px: np.ndarray,
    pred_coords_norm: np.ndarray,
    target_coords_norm: np.ndarray,
    horizon: int,
) -> Dict[str, Optional[float]]:
    if horizon <= 0:
        return {"ade_px": None, "fde_px": None, "ade_norm": None, "fde_norm": None}
    pred_px = pred_coords_px[:horizon]
    target_px = target_coords_px[:horizon]
    pred_norm = pred_coords_norm[:horizon]
    target_norm = target_coords_norm[:horizon]
    distances_px = np.linalg.norm(pred_px - target_px, axis=-1)
    distances_norm = np.linalg.norm(pred_norm - target_norm, axis=-1)
    return {
        "ade_px": float(distances_px.mean()),
        "fde_px": float(distances_px[-1]),
        "ade_norm": float(distances_norm.mean()),
        "fde_norm": float(distances_norm[-1]),
    }


def make_trajectory_points(
    sampled_indices: List[int],
    coords_norm: np.ndarray,
    coords_px: np.ndarray,
    context_anchors: int,
    future_source: str = "predicted",
) -> List[Dict[str, Any]]:
    points = []
    for idx, (local_frame_idx, coord_norm, coord_px) in enumerate(zip(sampled_indices, coords_norm, coords_px)):
        points.append(
            {
                "anchor_idx": idx,
                "local_frame_idx": local_frame_idx,
                "coord_norm": [float(coord_norm[0]), float(coord_norm[1])],
                "coord_px": [float(coord_px[0]), float(coord_px[1])],
                "source": "context_input" if idx < context_anchors else future_source,
            }
        )
    return points


def main() -> None:
    args = get_args()
    if args.context_anchors + args.prediction_anchors != 20:
        raise ValueError("Expected 5 context anchors plus 15 prediction anchors.")
    if args.model_num_frames < 20 or (args.model_num_frames - 1) % 4 != 0:
        raise ValueError("--model-num-frames must be >=20 and satisfy (N - 1) % 4 == 0.")
    if any(horizon < 1 or horizon > args.prediction_anchors for horizon in args.eval_horizons):
        raise ValueError("--eval-horizons values must be between 1 and prediction_anchors.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.mixed_precision]
    trajectory_enabled = not args.disable_trajectory_head

    dataset = SurgWMBenchAnchorDataset(
        dataset_root=args.dataset_root,
        manifest=args.manifest,
        height=args.height,
        width=args.width,
        context_anchors=args.context_anchors,
        prediction_anchors=args.prediction_anchors,
        limit=args.num_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=surgwmbench_anchor_collate,
    )

    transformer = CogVideoXTransformer3DModel.from_pretrained(transformer_checkpoint_path(args.checkpoint), torch_dtype=dtype)
    vae = AutoencoderKLCogVideoX.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float32)
    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    trajectory_head = (
        SurgWMBenchTrajectoryHead.from_checkpoint(resolve_trajectory_checkpoint(args), map_location="cpu")
        if trajectory_enabled
        else None
    )
    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()
    transformer.to(device=device, dtype=dtype).eval()
    vae.to(device=device, dtype=torch.float32).eval()
    if trajectory_head is not None:
        trajectory_head.to(device=device, dtype=dtype).eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    lpips_model = make_lpips(device)
    metric_names = ["psnr", "ssim", "lpips"]
    if trajectory_enabled:
        metric_names.extend(["ade_px", "fde_px", "ade_norm", "fde_norm"])
    metrics_by_horizon: Dict[str, Dict[str, List[Optional[float]]]] = {
        f"horizon_{h}": {metric_name: [] for metric_name in metric_names} for h in args.eval_horizons
    }
    samples: List[Dict[str, Any]] = []
    saved_videos = 0

    for batch in tqdm(loader, desc="Evaluating"):
        context_frames = batch["context_frames"].to(device=device, dtype=torch.float32)
        pred_targets = generate_future_anchors(transformer, vae, scheduler, context_frames, args, device, dtype, generator)
        pred_future_coords_norm = None
        if trajectory_head is not None:
            context_latents = encode_padded_context_latents(vae, context_frames, args, device, dtype)
            context_coords_norm = batch["context_coords_norm"].to(device=device, dtype=dtype)
            context_coords_aug, context_coord_mask = augment_context_coords(
                context_coords_norm,
                args.inference_coord_noise_std,
                args.inference_coord_mask_prob,
                generator=generator,
            )
            with torch.no_grad():
                pred_future_coords_norm = trajectory_head(
                    context_latents, context_coords_aug, context_coord_mask
                ).float().cpu()

        for b in range(pred_targets.shape[0]):
            original_size = batch["original_size"][b]
            pred_original = [
                tensor_to_uint8(pred_targets[b, :, frame_idx], original_size)
                for frame_idx in range(args.prediction_anchors)
            ]
            target_original = [
                load_original_uint8(args.dataset_root, rel_path) for rel_path in batch["target_frame_paths"][b]
            ]

            sample_metrics: Dict[str, Dict[str, Optional[float]]] = {}
            for horizon in args.eval_horizons:
                horizon_key = f"horizon_{horizon}"
                frame_metrics = [
                    compute_frame_metrics(pred_original[i], target_original[i], lpips_model, device)
                    for i in range(horizon)
                ]
                aggregate = {
                    "psnr": mean_or_none([m["psnr"] for m in frame_metrics]),
                    "ssim": mean_or_none([m["ssim"] for m in frame_metrics]),
                    "lpips": mean_or_none([m["lpips"] for m in frame_metrics]),
                }
                trajectory_metrics: Dict[str, Optional[float]] = {}
                if pred_future_coords_norm is not None:
                    pred_future_norm_np = pred_future_coords_norm[b].numpy()
                    target_future_norm_np = batch["target_coords_norm"][b].numpy()
                    pred_future_px = coords_norm_to_px(pred_future_coords_norm[b], original_size)
                    target_future_px = batch["target_coords_px"][b].numpy()
                    trajectory_metrics = compute_trajectory_metrics(
                        pred_future_px,
                        target_future_px,
                        pred_future_norm_np,
                        target_future_norm_np,
                        horizon,
                    )
                sample_metrics[horizon_key] = {**aggregate, **trajectory_metrics}
                for metric_name, value in aggregate.items():
                    metrics_by_horizon[horizon_key][metric_name].append(value)
                for metric_name, value in trajectory_metrics.items():
                    metrics_by_horizon[horizon_key][metric_name].append(value)

            if args.save_videos and saved_videos < args.max_videos:
                pred_path = output_dir / "videos" / f"pred_{saved_videos:04d}.mp4"
                target_path = output_dir / "videos" / f"target_{saved_videos:04d}.mp4"
                save_video(pred_path, pred_original)
                save_video(target_path, target_original)
                saved_videos += 1

            sample = {
                "patient_id": batch["patient_id"][b],
                "trajectory_id": batch["trajectory_id"][b],
                "difficulty": batch["difficulty"][b],
                "sampled_indices": batch["sampled_indices"][b],
                "target_frame_paths": batch["target_frame_paths"][b],
                "metrics": sample_metrics,
            }
            if pred_future_coords_norm is not None:
                context_norm_np = batch["context_coords_norm"][b].numpy()
                context_px_np = batch["context_coords_px"][b].numpy()
                pred_full_norm = np.concatenate([context_norm_np, pred_future_coords_norm[b].numpy()], axis=0)
                pred_full_px = np.concatenate(
                    [context_px_np, coords_norm_to_px(pred_future_coords_norm[b], original_size)], axis=0
                )
                target_full_norm = batch["anchor_coords_norm"][b].numpy()
                target_full_px = batch["anchor_coords_px"][b].numpy()
                sample["predicted_trajectory"] = make_trajectory_points(
                    batch["sampled_indices"][b], pred_full_norm, pred_full_px, args.context_anchors
                )
                sample["target_trajectory"] = make_trajectory_points(
                    batch["sampled_indices"][b],
                    target_full_norm,
                    target_full_px,
                    args.context_anchors,
                    future_source="human_gt",
                )
            samples.append(sample)

    aggregate_metrics = {
        horizon: {metric: mean_or_none(values) for metric, values in metric_values.items()}
        for horizon, metric_values in metrics_by_horizon.items()
    }
    result = {
        "dataset_name": "SurgWMBench",
        "model": "HieraSurg anchor I2V",
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "trajectory_enabled": trajectory_enabled,
        "trajectory_checkpoint": str(resolve_trajectory_checkpoint(args)) if trajectory_enabled else None,
        "context_anchors": args.context_anchors,
        "prediction_anchors": args.prediction_anchors,
        "eval_horizons": args.eval_horizons,
        "metric_resolution": "original",
        "generated_resolution": [args.height, args.width],
        "inference_coord_noise_std": args.inference_coord_noise_std,
        "inference_coord_mask_prob": args.inference_coord_mask_prob,
        "metrics": aggregate_metrics,
        "lpips_available": lpips_model is not None,
        "num_samples": len(samples),
        "samples": samples,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if trajectory_enabled:
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
    print(json.dumps({"metrics": aggregate_metrics, "output": str(output_dir / "metrics.json")}, indent=2))


if __name__ == "__main__":
    main()
