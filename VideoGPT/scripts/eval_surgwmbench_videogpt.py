import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import lpips
import torch
import torch.nn.functional as F

from videogpt import VideoGPT
from videogpt.surgwmbench_data import (
    SurgWMBenchDataModule,
    load_rgb_image,
    restore_letterboxed_frame,
)
from videogpt.surgwmbench_metrics import (
    MetricAccumulator,
    compute_trajectory_metrics,
    compute_lpips,
    compute_psnr,
    compute_ssim,
    normalized_coords_to_pixel,
)


def move_tensors_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def get_hparam(args_obj, name, default=None):
    if isinstance(args_obj, dict):
        return args_obj.get(name, default)
    return getattr(args_obj, name, default)


def tensor_to_uint8_hwc(frame):
    frame = frame.detach().cpu().clamp(0.0, 1.0)
    return (frame.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")


def resize_for_lpips(frame, size):
    if size is None or size <= 0:
        return frame
    return F.interpolate(
        frame.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def build_metrics_json(
    args,
    model_args,
    overall,
    by_horizon,
    by_difficulty,
    trajectory_overall,
    trajectory_by_horizon,
    trajectory_by_difficulty,
    sample_artifacts,
    num_samples,
    predictions_path,
    has_trajectory_head,
):
    return {
        "dataset_name": "SurgWMBench",
        "baseline": "VideoGPT",
        "prediction_task": (
            "20_anchor_future_frame_and_trajectory_prediction"
            if has_trajectory_head
            else "20_anchor_future_frame_prediction"
        ),
        "data_track": "sparse_20_anchor",
        "manifest": args.manifest,
        "checkpoint": args.ckpt,
        "context_frames": args.context_frames,
        "horizons": args.horizons,
        "sequence_length": get_hparam(model_args, "sequence_length"),
        "resolution": get_hparam(model_args, "resolution"),
        "original_resolution_metrics": True,
        "trajectory_head": has_trajectory_head,
        "lpips_downsample": args.lpips_downsample or None,
        "eval_traj_noise_std": float(args.eval_noise_std),
        "eval_traj_mask_prob": float(args.eval_mask_prob),
        "num_samples": num_samples,
        "image_metrics_overall": overall.summary(),
        "metrics_by_horizon": {
            str(horizon): accumulator.summary()
            for horizon, accumulator in by_horizon.items()
        },
        "metrics_by_difficulty": {
            difficulty: {
                str(horizon): accumulator.summary()
                for horizon, accumulator in horizon_map.items()
            }
            for difficulty, horizon_map in by_difficulty.items()
        },
        "trajectory_metrics_overall": trajectory_overall.summary(),
        "trajectory_metrics_by_horizon": {
            str(horizon): accumulator.summary()
            for horizon, accumulator in trajectory_by_horizon.items()
        },
        "trajectory_metrics_by_difficulty": {
            difficulty: {
                str(horizon): accumulator.summary()
                for horizon, accumulator in horizon_map.items()
            }
            for difficulty, horizon_map in trajectory_by_difficulty.items()
        },
        "predictions_jsonl": str(predictions_path) if predictions_path is not None else None,
        "sample_artifacts": sample_artifacts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="manifests/test.jsonl")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/surgwmbench_videogpt_eval")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 15])
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument("--lpips-net", type=str, default="alex")
    parser.add_argument(
        "--lpips-downsample",
        type=int,
        default=0,
        help="Optional square size for LPIPS only. 0 keeps original resolution.",
    )
    parser.add_argument("--max-save-samples", type=int, default=16)
    parser.add_argument(
        "--eval-noise-std",
        type=float,
        default=0.0,
        help="Inference-time Gaussian noise std applied to context trajectory coords.",
    )
    parser.add_argument(
        "--eval-mask-prob",
        type=float,
        default=0.0,
        help="Inference-time per-anchor random mask probability for context trajectory.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    gpt = VideoGPT.load_from_checkpoint(args.ckpt, weights_only=False).to(device)
    gpt.eval()
    gpt.eval_traj_noise_std = float(args.eval_noise_std)
    gpt.eval_traj_mask_prob = float(args.eval_mask_prob)
    model_args = gpt.hparams["args"]

    sequence_length = int(get_hparam(model_args, "sequence_length"))
    model_context_frames = int(get_hparam(model_args, "n_cond_frames", args.context_frames))
    resolution = args.resolution or int(get_hparam(model_args, "resolution"))
    if args.context_frames != model_context_frames:
        raise ValueError(
            f"Eval context_frames={args.context_frames} does not match model n_cond_frames={model_context_frames}"
        )
    if max(args.horizons) > sequence_length - args.context_frames:
        raise ValueError("A requested horizon extends past the 20-anchor sequence")

    data_args = SimpleNamespace(
        dataset_root=args.dataset_root,
        train_manifest=args.manifest,
        val_manifest=args.manifest,
        test_manifest=args.manifest,
        sequence_length=sequence_length,
        resolution=resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_clips=args.max_clips,
    )
    loader = SurgWMBenchDataModule(data_args).test_dataloader()

    lpips_model = None
    if not args.no_lpips:
        lpips_model = lpips.LPIPS(net=args.lpips_net).to(device)
        lpips_model.eval()

    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    has_trajectory_head = bool(getattr(gpt, "use_trajectory_head", False))
    predictions_path = output_dir / "predictions.jsonl" if has_trajectory_head else None
    if not has_trajectory_head:
        print("checkpoint has no trajectory head; trajectory metrics will be empty")

    overall = MetricAccumulator()
    by_horizon = {horizon: MetricAccumulator() for horizon in args.horizons}
    by_difficulty = defaultdict(lambda: {horizon: MetricAccumulator() for horizon in args.horizons})
    trajectory_overall = MetricAccumulator()
    trajectory_by_horizon = {horizon: MetricAccumulator() for horizon in args.horizons}
    trajectory_by_difficulty = defaultdict(
        lambda: {horizon: MetricAccumulator() for horizon in args.horizons}
    )
    sample_artifacts = []
    num_samples = 0
    max_horizon = max(args.horizons)

    predictions_file = predictions_path.open("w") if predictions_path is not None else None
    with torch.no_grad():
        for batch in loader:
            batch = move_tensors_to_device(batch, device)
            batch_size = batch["video"].shape[0]
            samples = gpt.sample(batch_size, batch)
            trajectory_pred_norm = None
            if has_trajectory_head:
                trajectory_pred_norm = gpt.predict_trajectory(batch)

            for sample_idx in range(batch_size):
                difficulty = batch["difficulty"][sample_idx]
                patient_id = batch["patient_id"][sample_idx]
                trajectory_id = batch["trajectory_id"][sample_idx]
                saved_for_sample = []
                trajectory_record = None
                trajectory_pred_px = None
                trajectory_target_px = None

                if trajectory_pred_norm is not None:
                    trajectory_pred_norm_sample = trajectory_pred_norm[
                        sample_idx, :max_horizon
                    ].detach().cpu()
                    future_geometries = batch["frame_geometries"][
                        sample_idx,
                        args.context_frames:args.context_frames + max_horizon,
                    ]
                    trajectory_pred_px = normalized_coords_to_pixel(
                        trajectory_pred_norm_sample,
                        future_geometries,
                    )
                    trajectory_target_px = batch["anchor_coords_px"][
                        sample_idx,
                        args.context_frames:args.context_frames + max_horizon,
                    ].detach().cpu()
                    trajectory_target_norm = batch["anchor_coords_norm"][
                        sample_idx,
                        args.context_frames:args.context_frames + max_horizon,
                    ].detach().cpu()

                restored_cache = {}
                target_cache = {}
                for frame_idx in range(args.context_frames, args.context_frames + max_horizon):
                    geometry = batch["frame_geometries"][sample_idx, frame_idx]
                    restored = restore_letterboxed_frame(samples[sample_idx, :, frame_idx], geometry)
                    target = load_rgb_image(batch["frame_paths"][sample_idx][frame_idx])
                    restored_cache[frame_idx] = restored
                    target_cache[frame_idx] = target

                    should_save = (
                        args.max_save_samples < 0
                        or num_samples < args.max_save_samples
                    )
                    if should_save:
                        filename = (
                            f"{num_samples:06d}_{patient_id}_{trajectory_id}"
                            f"_anchor{frame_idx + 1:02d}.png"
                        )
                        pred_path = pred_dir / filename
                        imageio.imwrite(pred_path, tensor_to_uint8_hwc(restored))
                        saved_for_sample.append(str(pred_path))

                for horizon in args.horizons:
                    for frame_idx in range(args.context_frames, args.context_frames + horizon):
                        pred = restored_cache[frame_idx]
                        target = target_cache[frame_idx]
                        metrics = {
                            "psnr": compute_psnr(pred, target),
                            "ssim": compute_ssim(pred, target),
                        }
                        if lpips_model is not None:
                            lpips_pred = resize_for_lpips(pred, args.lpips_downsample)
                            lpips_target = resize_for_lpips(target, args.lpips_downsample)
                            metrics["lpips"] = compute_lpips(
                                lpips_pred,
                                lpips_target,
                                lpips_model,
                                device,
                            )

                        by_horizon[horizon].update(metrics)
                        by_difficulty[difficulty][horizon].update(metrics)
                        if horizon == max_horizon:
                            overall.update(metrics)

                    if trajectory_pred_px is not None:
                        trajectory_metrics = compute_trajectory_metrics(
                            trajectory_pred_px[:horizon],
                            trajectory_target_px[:horizon],
                        )
                        trajectory_by_horizon[horizon].update(trajectory_metrics)
                        trajectory_by_difficulty[difficulty][horizon].update(trajectory_metrics)
                        if horizon == max_horizon:
                            trajectory_overall.update(trajectory_metrics)

                if trajectory_pred_px is not None:
                    trajectory_record = {
                        "patient_id": patient_id,
                        "trajectory_id": trajectory_id,
                        "difficulty": difficulty,
                        "future_anchor_start": args.context_frames + 1,
                        "future_anchor_end": args.context_frames + max_horizon,
                        "pred_coords_norm": trajectory_pred_norm_sample.tolist(),
                        "pred_coords_px": trajectory_pred_px.tolist(),
                        "target_coords_norm": trajectory_target_norm.tolist(),
                        "target_coords_px": trajectory_target_px.tolist(),
                    }
                    predictions_file.write(json.dumps(trajectory_record) + "\n")

                if saved_for_sample:
                    artifact = {
                        "patient_id": patient_id,
                        "trajectory_id": trajectory_id,
                        "difficulty": difficulty,
                        "prediction_paths": saved_for_sample,
                    }
                    if trajectory_record is not None:
                        artifact["trajectory_prediction"] = trajectory_record
                    sample_artifacts.append(artifact)
                num_samples += 1
    if predictions_file is not None:
        predictions_file.close()

    metrics = build_metrics_json(
        args,
        model_args,
        overall,
        by_horizon,
        by_difficulty,
        trajectory_overall,
        trajectory_by_horizon,
        trajectory_by_difficulty,
        sample_artifacts,
        num_samples,
        predictions_path,
        has_trajectory_head,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
