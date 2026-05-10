import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader

from finetune.dataset.surgwmbench_anchor_dataset import (
    SurgWMBenchAnchorDataset,
    surgwmbench_anchor_collate,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the SurgWMBench 20-anchor frame loader.")
    parser.add_argument("--dataset-root", required=True, help="Path to the SurgWMBench dataset root.")
    parser.add_argument("--manifest", default="manifests/train.jsonl", help="Manifest path relative to dataset root.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to inspect.")
    parser.add_argument("--height", type=int, default=64, help="Resize height used only for loader validation.")
    parser.add_argument("--width", type=int, default=64, help="Resize width used only for loader validation.")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    dataset = SurgWMBenchAnchorDataset(
        dataset_root=args.dataset_root,
        manifest=args.manifest,
        height=args.height,
        width=args.width,
        limit=args.num_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=surgwmbench_anchor_collate,
    )

    checked = 0
    first_summary = None
    for batch in loader:
        batch_size = batch["anchor_frames"].shape[0]
        assert batch["anchor_frames"].shape[1:] == (3, 20, args.height, args.width)
        assert batch["context_frames"].shape[1:] == (3, 5, args.height, args.width)
        assert batch["target_frames"].shape[1:] == (3, 15, args.height, args.width)
        assert batch["anchor_coords_norm"].shape[1:] == (20, 2)
        assert batch["context_coords_norm"].shape[1:] == (5, 2)
        assert batch["target_coords_norm"].shape[1:] == (15, 2)
        for paths, sampled_indices, original_size in zip(
            batch["anchor_frame_paths"], batch["sampled_indices"], batch["original_size"]
        ):
            assert len(paths) == 20
            assert len(sampled_indices) == 20
            assert original_size == (1080, 1920)
            for rel_path in paths:
                path = Path(args.dataset_root) / rel_path
                assert path.exists(), f"Missing frame path: {path}"
        if first_summary is None:
            first_summary = {
                "patient_id": batch["patient_id"][0],
                "trajectory_id": batch["trajectory_id"][0],
                "anchor_shape": list(batch["anchor_frames"].shape),
                "anchor_coords_shape": list(batch["anchor_coords_norm"].shape),
                "context_paths": batch["context_frame_paths"][0],
                "target_paths_first_last": [
                    batch["target_frame_paths"][0][0],
                    batch["target_frame_paths"][0][-1],
                ],
            }
        checked += batch_size

    print(
        json.dumps(
            {
                "status": "ok",
                "manifest": args.manifest,
                "checked_samples": checked,
                "first_sample": first_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
