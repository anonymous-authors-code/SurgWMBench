import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(dataset_root: Path, relative_path: str) -> Path:
    path = dataset_root / relative_path
    if path.exists():
        return path

    # Some transferred manifests may contain the private-use glyph where the
    # filesystem has a literal '*'. Keep this as a loader-level fallback.
    if "\uf021" in relative_path:
        fallback = dataset_root / relative_path.replace("\uf021", "*")
        if fallback.exists():
            return fallback

    raise FileNotFoundError(f"Missing SurgWMBench path: {path}")


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_image_transform(height: int, width: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class SurgWMBenchAnchorDataset(Dataset):
    """Manifest-backed SurgWMBench 20-anchor frame prediction dataset.

    Each item contains the 20 sparse human-anchor frames from one dense clip.
    The first `context_anchors` frames are conditioning frames and the next
    `prediction_anchors` frames are the future prediction target.
    """

    def __init__(
        self,
        dataset_root: str,
        manifest: str,
        height: int = 288,
        width: int = 512,
        context_anchors: int = 5,
        prediction_anchors: int = 15,
        limit: Optional[int] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.manifest_path = _resolve_path(self.dataset_root, manifest)
        self.rows = load_manifest(self.manifest_path)
        if limit is not None:
            self.rows = self.rows[:limit]

        self.height = height
        self.width = width
        self.context_anchors = context_anchors
        self.prediction_anchors = prediction_anchors
        self.required_anchors = context_anchors + prediction_anchors
        self.transform = make_image_transform(height, width)

        if self.required_anchors > 20:
            raise ValueError("SurgWMBench currently provides exactly 20 human anchors per clip.")

    def __len__(self) -> int:
        return len(self.rows)

    def _anchor_frame_records(self, row: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if row.get("num_human_anchors") != 20 or len(row.get("sampled_indices", [])) != 20:
            raise ValueError(f"Expected 20 anchors for {row.get('trajectory_id')}")

        annotation = _load_json(_resolve_path(self.dataset_root, row["annotation_path"]))
        frames = annotation["frames"]
        sampled_indices = row["sampled_indices"]
        records: List[Dict[str, Any]] = []

        for anchor_idx, local_frame_idx in enumerate(sampled_indices):
            frame = frames[local_frame_idx]
            if frame["local_frame_idx"] != local_frame_idx:
                raise ValueError(
                    f"Frame index mismatch for {row.get('trajectory_id')}: "
                    f"expected {local_frame_idx}, got {frame['local_frame_idx']}"
                )
            if frame.get("anchor_idx") != anchor_idx:
                raise ValueError(
                    f"Anchor index mismatch for {row.get('trajectory_id')}: "
                    f"expected {anchor_idx}, got {frame.get('anchor_idx')}"
                )
            if not frame.get("is_human_labeled", False):
                raise ValueError(f"Anchor frame {local_frame_idx} is not marked human-labeled.")
            records.append(frame)

        return annotation, records

    def _load_frame(self, relative_path: str) -> torch.Tensor:
        with Image.open(_resolve_path(self.dataset_root, relative_path)) as image:
            return self.transform(image.convert("RGB"))

    @staticmethod
    def _anchor_coords(records: Sequence[Dict[str, Any]], field: str) -> torch.Tensor:
        coords = []
        for frame in records:
            coord = frame.get(field)
            if coord is None:
                raise ValueError(f"Missing {field} for anchor frame {frame.get('local_frame_idx')}")
            coords.append(coord)
        return torch.tensor(coords, dtype=torch.float32)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        annotation, anchor_records = self._anchor_frame_records(row)
        selected_records = anchor_records[: self.required_anchors]
        anchor_frames = torch.stack([self._load_frame(frame["frame_path"]) for frame in selected_records], dim=1)
        anchor_coords_norm = self._anchor_coords(selected_records, "human_coord_norm")
        anchor_coords_px = self._anchor_coords(selected_records, "human_coord_px")

        context_frames = anchor_frames[:, : self.context_anchors]
        target_frames = anchor_frames[:, self.context_anchors : self.required_anchors]
        context_coords_norm = anchor_coords_norm[: self.context_anchors]
        target_coords_norm = anchor_coords_norm[self.context_anchors : self.required_anchors]
        context_coords_px = anchor_coords_px[: self.context_anchors]
        target_coords_px = anchor_coords_px[self.context_anchors : self.required_anchors]

        return {
            "anchor_frames": anchor_frames,
            "context_frames": context_frames,
            "target_frames": target_frames,
            "anchor_coords_norm": anchor_coords_norm,
            "context_coords_norm": context_coords_norm,
            "target_coords_norm": target_coords_norm,
            "anchor_coords_px": anchor_coords_px,
            "context_coords_px": context_coords_px,
            "target_coords_px": target_coords_px,
            "sampled_indices": list(row["sampled_indices"]),
            "anchor_frame_paths": [frame["frame_path"] for frame in selected_records],
            "context_frame_paths": [frame["frame_path"] for frame in selected_records[: self.context_anchors]],
            "target_frame_paths": [frame["frame_path"] for frame in selected_records[self.context_anchors :]],
            "original_size": (annotation["image_size"]["height"], annotation["image_size"]["width"]),
            "patient_id": row["patient_id"],
            "source_video_id": row["source_video_id"],
            "trajectory_id": row["trajectory_id"],
            "difficulty": row.get("difficulty"),
            "num_frames": row["num_frames"],
            "annotation_path": row["annotation_path"],
        }


def pad_anchor_video(anchor_frames: torch.Tensor, model_num_frames: int) -> torch.Tensor:
    """Pad anchor frames to the model frame count by repeating the last anchor."""

    if anchor_frames.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(anchor_frames.shape)}")
    current_frames = anchor_frames.shape[2]
    if current_frames > model_num_frames:
        raise ValueError(f"Anchor frames ({current_frames}) exceed model_num_frames ({model_num_frames}).")
    if current_frames == model_num_frames:
        return anchor_frames
    pad_count = model_num_frames - current_frames
    pad = anchor_frames[:, :, -1:].repeat(1, 1, pad_count, 1, 1)
    return torch.cat([anchor_frames, pad], dim=2)


def latent_frame_count(num_frames: int, temporal_compression_ratio: int) -> int:
    return (num_frames - 1) // temporal_compression_ratio + 1


def surgwmbench_anchor_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tensor_keys = (
        "anchor_frames",
        "context_frames",
        "target_frames",
        "anchor_coords_norm",
        "context_coords_norm",
        "target_coords_norm",
        "anchor_coords_px",
        "context_coords_px",
        "target_coords_px",
    )
    out: Dict[str, Any] = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}

    metadata_keys = [
        "sampled_indices",
        "anchor_frame_paths",
        "context_frame_paths",
        "target_frame_paths",
        "original_size",
        "patient_id",
        "source_video_id",
        "trajectory_id",
        "difficulty",
        "num_frames",
        "annotation_path",
    ]
    for key in metadata_keys:
        out[key] = [item[key] for item in batch]
    return out


def iter_anchor_samples(
    dataset_root: str,
    manifest: str,
    num_samples: Optional[int] = None,
) -> Iterable[Dict[str, Any]]:
    dataset = SurgWMBenchAnchorDataset(
        dataset_root=dataset_root,
        manifest=manifest,
        height=64,
        width=64,
        limit=num_samples,
    )
    for idx in range(len(dataset)):
        yield dataset[idx]
