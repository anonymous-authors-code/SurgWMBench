import json

import imageio.v2 as imageio
import numpy as np
import torch

from videogpt.surgwmbench_data import (
    SurgWMBenchAnchorDataset,
    letterbox_frame,
    restore_letterboxed_frame,
    surgwmbench_collate,
)
from videogpt.gpt import TrajectoryHead, augment_trajectory_context
from videogpt.surgwmbench_metrics import (
    compute_psnr,
    compute_ssim,
    compute_trajectory_metrics,
    normalized_coords_to_pixel,
)


def _write_toy_surgwmbench(root):
    clip_dir = root / "clips" / "video_01" / "traj_01"
    frames_dir = clip_dir / "frames"
    frames_dir.mkdir(parents=True)
    (root / "manifests").mkdir()

    frames = []
    anchors = []
    sampled_indices = list(range(20))
    for idx in sampled_indices:
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        image[..., 0] = idx
        image[..., 1] = 255 - idx
        frame_rel = f"clips/video_01/traj_01/frames/{idx:06d}.png"
        imageio.imwrite(root / frame_rel, image)
        frames.append({
            "local_frame_idx": idx,
            "source_frame_idx": idx + 100,
            "frame_path": frame_rel,
            "is_human_labeled": True,
            "anchor_idx": idx,
            "human_coord_px": [float(idx), float(idx + 1)],
            "human_coord_norm": [0.1, 0.2],
            "coord_source": "human",
        })
        anchors.append({
            "anchor_idx": idx,
            "old_frame_idx": idx,
            "local_frame_idx": idx,
            "source_frame_idx": idx + 100,
            "label_name": f"Label {idx + 1}",
            "value": idx + 1,
            "coord_px": [float(idx), float(idx + 1)],
            "coord_norm": [0.1, 0.2],
        })

    annotation = {
        "dataset_version": "SurgWMBench",
        "patient_id": "video_01",
        "source_video_id": "video_01",
        "source_video_path": "videos/video_01/video_left.avi",
        "trajectory_id": "traj_01",
        "difficulty": "low",
        "num_frames": 20,
        "image_size": {"width": 20, "height": 10},
        "coordinate_format": "pixel_xy",
        "coordinate_origin": "top_left",
        "num_human_anchors": 20,
        "sampled_indices": sampled_indices,
        "available_interpolation_methods": ["linear"],
        "default_interpolation_method": "linear",
        "frames": frames,
        "human_anchors": anchors,
        "interpolation_files": {"linear": "interpolations/video_01/traj_01.linear.json"},
    }
    annotation_path = clip_dir / "annotation.json"
    annotation_path.write_text(json.dumps(annotation))

    row = {
        "annotation_path": "clips/video_01/traj_01/annotation.json",
        "dataset_version": "SurgWMBench",
        "default_interpolation_method": "linear",
        "difficulty": "low",
        "frames_dir": "clips/video_01/traj_01/frames",
        "interpolation_files": {"linear": "interpolations/video_01/traj_01.linear.json"},
        "num_frames": 20,
        "num_human_anchors": 20,
        "patient_id": "video_01",
        "sampled_indices": sampled_indices,
        "source_video_id": "video_01",
        "source_video_path": "videos/video_01/video_left.avi",
        "trajectory_id": "traj_01",
    }
    manifest = root / "manifests" / "train.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    return manifest


def test_surgwmbench_anchor_dataset_loads_20_anchor_frames(tmp_path):
    manifest = _write_toy_surgwmbench(tmp_path)
    dataset = SurgWMBenchAnchorDataset(tmp_path, manifest, resolution=16)

    sample = dataset[0]

    assert sample["video"].shape == (3, 20, 16, 16)
    assert sample["anchor_local_frame_indices"].tolist() == list(range(20))
    assert sample["anchor_coords_px"].shape == (20, 2)
    assert sample["anchor_coords_norm"].shape == (20, 2)
    assert sample["anchor_coords_px"][19].tolist() == [19.0, 20.0]
    torch.testing.assert_close(sample["anchor_coords_norm"][0], torch.tensor([0.1, 0.2]))
    assert len(sample["frame_paths"]) == 20
    assert sample["difficulty"] == "low"


def test_surgwmbench_collate_preserves_metadata(tmp_path):
    manifest = _write_toy_surgwmbench(tmp_path)
    dataset = SurgWMBenchAnchorDataset(tmp_path, manifest, resolution=16)

    batch = surgwmbench_collate([dataset[0], dataset[0]])

    assert batch["video"].shape == (2, 3, 20, 16, 16)
    assert batch["anchor_coords_px"].shape == (2, 20, 2)
    assert batch["anchor_coords_norm"].shape == (2, 20, 2)
    assert batch["patient_id"] == ["video_01", "video_01"]
    assert len(batch["frame_paths"][0]) == 20


def test_letterbox_restore_returns_original_size():
    frame = torch.rand(3, 10, 20)
    letterboxed, geometry = letterbox_frame(frame, resolution=16)

    restored = restore_letterboxed_frame(letterboxed, geometry)

    assert letterboxed.shape == (3, 16, 16)
    assert restored.shape == (3, 10, 20)


def test_basic_image_metrics_identical_images():
    image = torch.ones(3, 12, 12)

    assert compute_psnr(image, image) == float("inf")
    assert compute_ssim(image, image) == 1.0


def test_trajectory_metrics_use_pixel_distances():
    pred = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    target = torch.zeros(2, 2)

    metrics = compute_trajectory_metrics(pred, target)

    assert metrics["ade_px"] == 2.5
    assert metrics["fde_px"] == 5.0


def test_normalized_coords_convert_to_original_pixels():
    coords_norm = torch.tensor([[0.5, 0.5], [1.0, 0.25]])
    frame_geometries = torch.tensor([
        [10, 20, 8, 16, 0, 0, 0, 0],
        [30, 40, 12, 16, 0, 0, 0, 0],
    ])

    coords_px = normalized_coords_to_pixel(coords_norm, frame_geometries)

    assert coords_px.tolist() == [[10.0, 5.0], [40.0, 7.5]]


def test_trajectory_head_predicts_future_anchor_shape():
    head = TrajectoryHead(input_dim=4, hidden_dim=8, n_future_frames=15)
    frame_cond = torch.randn(2, 5, 3, 3, 4)

    coords = head(frame_cond)

    assert coords.shape == (2, 15, 2)
    assert torch.all(coords >= 0.0)
    assert torch.all(coords <= 1.0)


def test_trajectory_head_accepts_context_trajectory_condition():
    head = TrajectoryHead(
        input_dim=4,
        hidden_dim=8,
        n_future_frames=15,
        use_context_condition=True,
    )
    frame_cond = torch.randn(2, 5, 3, 3, 4)
    context_coords = torch.rand(2, 5, 2)
    context_mask = torch.ones(2, 5)

    coords = head(frame_cond, context_coords, context_mask)

    assert coords.shape == (2, 15, 2)
    assert torch.all(coords >= 0.0)
    assert torch.all(coords <= 1.0)


def test_trajectory_context_augmentation_can_mask_all_conditions():
    context_coords = torch.full((2, 5, 2), 0.5)

    coords, mask = augment_trajectory_context(
        context_coords,
        noise_std=0.0,
        mask_prob=1.0,
        training=True,
    )

    assert torch.equal(coords, torch.zeros_like(coords))
    assert torch.equal(mask, torch.zeros_like(mask))


def test_trajectory_context_augmentation_disabled_for_eval():
    context_coords = torch.full((2, 5, 2), 0.5)

    coords, mask = augment_trajectory_context(
        context_coords,
        noise_std=10.0,
        mask_prob=1.0,
        training=False,
    )

    assert torch.equal(coords, context_coords)
    assert torch.equal(mask, torch.ones_like(mask))
