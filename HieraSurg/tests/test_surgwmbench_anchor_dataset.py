from pathlib import Path

import pytest
import torch

from finetune.dataset.surgwmbench_anchor_dataset import (
    SurgWMBenchAnchorDataset,
    latent_frame_count,
    pad_anchor_video,
    surgwmbench_anchor_collate,
)
from finetune.trajectory_head import SurgWMBenchTrajectoryHead


DATASET_ROOT = Path("/mnt/hdd1/neurips2026_dataset_track/SurgWMBench")


def test_latent_frame_count_matches_cogvideox_rule():
    assert latent_frame_count(5, 4) == 2
    assert latent_frame_count(20, 4) == 5
    assert latent_frame_count(33, 4) == 9


def test_pad_anchor_video_repeats_last_anchor():
    frames = torch.arange(1 * 3 * 20 * 2 * 2).view(1, 3, 20, 2, 2)
    padded = pad_anchor_video(frames, 33)
    assert padded.shape == (1, 3, 33, 2, 2)
    assert torch.equal(padded[:, :, :20], frames)
    assert torch.equal(padded[:, :, -1], frames[:, :, -1])


def test_trajectory_head_predicts_future_coords():
    head = SurgWMBenchTrajectoryHead(
        latent_channels=16,
        context_anchors=5,
        prediction_anchors=15,
        hidden_dim=32,
    )
    latents = torch.randn(2, 2, 16, 4, 4)
    context_coords = torch.rand(2, 5, 2)
    prediction = head(latents, context_coords)
    assert prediction.shape == (2, 15, 2)
    assert torch.all(prediction >= 0)
    assert torch.all(prediction <= 1)


def test_trajectory_head_accepts_context_coord_mask():
    head = SurgWMBenchTrajectoryHead(
        latent_channels=16,
        context_anchors=5,
        prediction_anchors=15,
        hidden_dim=32,
    )
    latents = torch.randn(2, 2, 16, 4, 4)
    context_coords = torch.rand(2, 5, 2)
    context_mask = torch.tensor([[1, 1, 0, 1, 0], [1, 0, 0, 1, 1]], dtype=torch.float32)
    prediction = head(latents, context_coords, context_mask)
    assert prediction.shape == (2, 15, 2)
    assert torch.all(prediction >= 0)
    assert torch.all(prediction <= 1)


def test_trajectory_head_loads_legacy_coord_condition_checkpoint(tmp_path):
    head = SurgWMBenchTrajectoryHead(latent_channels=8, hidden_dim=16, coord_condition_dim=10)
    head.save_checkpoint(tmp_path)
    payload_path = tmp_path / "trajectory_head.pt"
    payload = torch.load(payload_path, weights_only=False)
    payload["config"].pop("coord_condition_dim")
    torch.save(payload, payload_path)
    loaded = SurgWMBenchTrajectoryHead.from_checkpoint(tmp_path)
    assert loaded.coord_condition_dim == 10


def test_trajectory_head_checkpoint_round_trip(tmp_path):
    head = SurgWMBenchTrajectoryHead(latent_channels=8, hidden_dim=16)
    head.save_checkpoint(tmp_path)
    loaded = SurgWMBenchTrajectoryHead.from_checkpoint(tmp_path)
    assert loaded.config == head.config


@pytest.mark.skipif(not DATASET_ROOT.exists(), reason="SurgWMBench dataset root is not available")
def test_real_surgwmbench_anchor_sample_uses_twenty_human_anchors():
    dataset = SurgWMBenchAnchorDataset(
        dataset_root=str(DATASET_ROOT),
        manifest="manifests/train.jsonl",
        height=64,
        width=64,
        limit=1,
    )
    sample = dataset[0]
    assert sample["anchor_frames"].shape == (3, 20, 64, 64)
    assert sample["context_frames"].shape == (3, 5, 64, 64)
    assert sample["target_frames"].shape == (3, 15, 64, 64)
    assert sample["anchor_coords_norm"].shape == (20, 2)
    assert sample["context_coords_norm"].shape == (5, 2)
    assert sample["target_coords_norm"].shape == (15, 2)
    assert sample["anchor_coords_px"].shape == (20, 2)
    assert torch.all(sample["anchor_coords_norm"] >= 0)
    assert torch.all(sample["anchor_coords_norm"] <= 1)
    assert torch.equal(sample["context_coords_norm"], sample["anchor_coords_norm"][:5])
    assert torch.equal(sample["target_coords_norm"], sample["anchor_coords_norm"][5:])
    assert len(sample["sampled_indices"]) == 20
    assert sample["original_size"] == (1080, 1920)
    assert sample["anchor_frame_paths"][0].endswith(".png")

    batch = surgwmbench_anchor_collate([sample])
    assert batch["anchor_coords_norm"].shape == (1, 20, 2)
    assert batch["context_coords_norm"].shape == (1, 5, 2)
    assert batch["target_coords_norm"].shape == (1, 15, 2)
