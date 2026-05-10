import json

import numpy as np
import pytest
from PIL import Image

from ivideogpt.data import SurgWMBenchAnchorDataset


def _write_toy_surgwmbench(root, sampled_indices=None):
    sampled_indices = sampled_indices or list(range(20))
    clip_dir = root / "clips" / "video_01" / "toy_trajectory"
    frames_dir = clip_dir / "frames"
    frames_dir.mkdir(parents=True)
    for idx in range(20):
        image = np.full((6, 8, 3), idx, dtype=np.uint8)
        Image.fromarray(image).save(frames_dir / f"{idx:06d}.png")

    frames = []
    anchors = []
    for anchor_idx, local_frame_idx in enumerate(range(20)):
        frame_path = f"clips/video_01/toy_trajectory/frames/{local_frame_idx:06d}.png"
        frames.append({
            "local_frame_idx": local_frame_idx,
            "source_frame_idx": local_frame_idx,
            "frame_path": frame_path,
            "is_human_labeled": True,
            "anchor_idx": anchor_idx,
            "human_coord_px": [float(local_frame_idx), float(local_frame_idx + 1)],
            "human_coord_norm": [0.1, 0.2],
            "coord_source": "human",
        })
        anchors.append({
            "anchor_idx": anchor_idx,
            "old_frame_idx": anchor_idx,
            "local_frame_idx": local_frame_idx,
            "source_frame_idx": local_frame_idx,
            "label_name": f"Label {anchor_idx + 1}",
            "value": anchor_idx + 1,
            "coord_px": [float(local_frame_idx), float(local_frame_idx + 1)],
            "coord_norm": [0.1, 0.2],
        })

    annotation = {
        "dataset_version": "SurgWMBench",
        "patient_id": "video_01",
        "source_video_id": "video_01",
        "trajectory_id": "toy_trajectory",
        "difficulty": "low",
        "num_frames": 20,
        "image_size": {"width": 8, "height": 6},
        "num_human_anchors": 20,
        "sampled_indices": sampled_indices,
        "frames": frames,
        "human_anchors": anchors,
    }
    (clip_dir / "annotation.json").write_text(json.dumps(annotation))

    manifest_dir = root / "manifests"
    manifest_dir.mkdir()
    row = {
        "annotation_path": "clips/video_01/toy_trajectory/annotation.json",
        "dataset_version": "SurgWMBench",
        "default_interpolation_method": "linear",
        "difficulty": "low",
        "frames_dir": "clips/video_01/toy_trajectory/frames",
        "interpolation_files": {},
        "num_frames": 20,
        "num_human_anchors": 20,
        "patient_id": "video_01",
        "sampled_indices": sampled_indices,
        "source_video_id": "video_01",
        "source_video_path": "videos/video_01/video_left.avi",
        "trajectory_id": "toy_trajectory",
    }
    (manifest_dir / "train.jsonl").write_text(json.dumps(row) + "\n")


def test_surgwmbench_anchor_dataset_loads_20_anchor_frames(tmp_path):
    _write_toy_surgwmbench(tmp_path)

    dataset = SurgWMBenchAnchorDataset(
        dataset_root=tmp_path,
        manifest="manifests/train.jsonl",
        image_size=16,
        return_metadata=True,
        return_trajectory=True,
    )
    sample = dataset[0]

    assert sample["pixel_values"].shape == (20, 3, 16, 16)
    assert sample["trajectory_norm"].shape == (20, 2)
    assert sample["trajectory_px"].shape == (20, 2)
    assert sample["trajectory_norm"][0].tolist() == pytest.approx([0.1, 0.2])
    assert sample["trajectory_px"][3].tolist() == [3.0, 4.0]
    assert sample["metadata"]["sampled_indices"] == list(range(20))
    assert sample["metadata"]["anchor_frame_paths"][0].endswith("000000.png")


def test_surgwmbench_anchor_dataset_rejects_sampled_index_mismatch(tmp_path):
    _write_toy_surgwmbench(tmp_path, sampled_indices=list(range(1, 21)))

    dataset = SurgWMBenchAnchorDataset(dataset_root=tmp_path, manifest="manifests/train.jsonl", image_size=16)
    with pytest.raises(ValueError, match="sampled_indices"):
        dataset[0]
