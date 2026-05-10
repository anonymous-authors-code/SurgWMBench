"""Fast trajectory-only inference for v8 on SurgWMBench test set.

Skips the autoregressive image sampling loop (which dominates eval time) and
runs only the trajectory head, producing per-clip predictions in original
pixel coordinates plus the original frame paths for downstream visualization.
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from videogpt import VideoGPT
from videogpt.surgwmbench_data import SurgWMBenchDataModule
from videogpt.surgwmbench_metrics import normalized_coords_to_pixel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/mnt/data/SurgWMBench")
    parser.add_argument("--manifest", default="manifests/test.jsonl")
    parser.add_argument(
        "--ckpt",
        default="lightning_logs/version_8/checkpoints/epoch=19-step=960.ckpt",
    )
    parser.add_argument("--output", default="outputs/v8_trajectory_predictions.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-noise-std", type=float, default=0.0)
    parser.add_argument("--eval-mask-prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt = VideoGPT.load_from_checkpoint(args.ckpt, weights_only=False).to(device).eval()
    gpt.eval_traj_noise_std = float(args.eval_noise_std)
    gpt.eval_traj_mask_prob = float(args.eval_mask_prob)
    model_args = gpt.hparams["args"]
    sequence_length = int(getattr(model_args, "sequence_length", 20))
    context_frames = int(getattr(model_args, "n_cond_frames", 5))
    resolution = int(getattr(model_args, "resolution", 128))
    max_horizon = sequence_length - context_frames

    data_args = SimpleNamespace(
        dataset_root=args.dataset_root,
        train_manifest=args.manifest,
        val_manifest=args.manifest,
        test_manifest=args.manifest,
        sequence_length=sequence_length,
        resolution=resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_clips=None,
    )
    loader = SurgWMBenchDataModule(data_args).test_dataloader()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_clips = 0
    t0 = time.time()
    with out_path.open("w") as f, torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
            traj_pred_norm = gpt.predict_trajectory(batch)
            bsz = batch["video"].shape[0]
            for i in range(bsz):
                future_geom = batch["frame_geometries"][
                    i, context_frames : context_frames + max_horizon
                ]
                pred_norm = traj_pred_norm[i, :max_horizon].detach().cpu()
                pred_px = normalized_coords_to_pixel(pred_norm, future_geom)
                target_norm = batch["anchor_coords_norm"][
                    i, context_frames : context_frames + max_horizon
                ].detach().cpu()
                target_px = batch["anchor_coords_px"][
                    i, context_frames : context_frames + max_horizon
                ].detach().cpu()
                paths = batch["frame_paths"][i]
                ctx_paths = list(paths[:context_frames])
                future_paths = list(paths[context_frames : context_frames + max_horizon])
                ctx_coords_px = batch["anchor_coords_px"][i, :context_frames].detach().cpu()
                geom_first = batch["frame_geometries"][i, context_frames].detach().cpu().tolist()
                record = {
                    "patient_id": batch["patient_id"][i],
                    "trajectory_id": batch["trajectory_id"][i],
                    "difficulty": batch["difficulty"][i],
                    "context_frames": context_frames,
                    "future_anchor_start": context_frames + 1,
                    "future_anchor_end": context_frames + max_horizon,
                    "context_paths": ctx_paths,
                    "future_paths": future_paths,
                    "context_coords_px": ctx_coords_px.tolist(),
                    "pred_coords_norm": pred_norm.tolist(),
                    "pred_coords_px": pred_px.tolist(),
                    "target_coords_norm": target_norm.tolist(),
                    "target_coords_px": target_px.tolist(),
                    "frame_geometry_future0": geom_first,
                }
                f.write(json.dumps(record) + "\n")
            n_clips += bsz
    print(f"wrote {n_clips} clips to {out_path} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
