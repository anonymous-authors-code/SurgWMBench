import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F


DEFAULT_SURGWMBENCH_ROOT = "/mnt/hdd1/neurips2026_dataset_track/SurgWMBench"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _resolve_dataset_path(dataset_root: Path, relative_path: Union[str, Path]) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        candidate = path
    else:
        candidate = dataset_root / path
    if candidate.exists():
        return candidate

    # Some transferred manifests used a private-use character where the
    # filesystem path contained a literal '*'. Keep this as a loader fallback.
    fixed = str(candidate).replace("\uf021", "*")
    if fixed != str(candidate):
        matches = sorted(Path(fixed).parent.glob(Path(fixed).name))
        if matches:
            return matches[0]
    return candidate


def _load_manifest(dataset_root: Path, manifest: Union[str, Path]) -> List[Dict[str, Any]]:
    manifest_path = _resolve_dataset_path(dataset_root, manifest)
    rows = []
    with manifest_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class SurgWMBenchAnchorDataset(data.Dataset):
    """Loads fixed 20-frame sparse-human-anchor clips from SurgWMBench."""

    def __init__(
        self,
        dataset_root: Union[str, Path] = DEFAULT_SURGWMBENCH_ROOT,
        manifest: Union[str, Path] = "manifests/train.jsonl",
        image_size: int = 256,
        max_samples: Optional[int] = None,
        return_metadata: bool = False,
        return_trajectory: bool = False,
    ):
        self.dataset_root = Path(dataset_root)
        self.manifest = manifest
        self.image_size = image_size
        self.return_metadata = return_metadata
        self.return_trajectory = return_trajectory
        self.rows = _load_manifest(self.dataset_root, manifest)
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if len(self.rows) == 0:
            raise ValueError(f"No SurgWMBench rows found in manifest {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def _annotation_for_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return _read_json(_resolve_dataset_path(self.dataset_root, row["annotation_path"]))

    def _anchor_frames(self, annotation: Dict[str, Any]) -> List[Dict[str, Any]]:
        sampled_indices = annotation.get("sampled_indices")
        anchors = sorted(annotation["human_anchors"], key=lambda item: item["anchor_idx"])
        if len(anchors) != 20:
            raise ValueError(
                f"Expected 20 human anchors for {annotation.get('patient_id')}/"
                f"{annotation.get('trajectory_id')}, got {len(anchors)}"
            )
        local_indices = [anchor["local_frame_idx"] for anchor in anchors]
        if sampled_indices is not None and local_indices != sampled_indices:
            raise ValueError(
                f"sampled_indices do not match human anchor local_frame_idx for "
                f"{annotation.get('patient_id')}/{annotation.get('trajectory_id')}"
            )

        frames_by_idx = {frame["local_frame_idx"]: frame for frame in annotation["frames"]}
        return [frames_by_idx[idx] for idx in local_indices]

    def _image_width_height(self, annotation: Dict[str, Any]) -> tuple[float, float]:
        image_size = annotation.get("image_size") or {}
        if isinstance(image_size, dict):
            width = image_size.get("width")
            height = image_size.get("height")
        else:
            width, height = image_size
        if width is None or height is None:
            raise ValueError(
                f"Missing image_size width/height for {annotation.get('patient_id')}/"
                f"{annotation.get('trajectory_id')}"
            )
        return float(width), float(height)

    def _anchor_trajectories(
        self,
        annotation: Dict[str, Any],
        anchor_frames: List[Dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchors = sorted(annotation["human_anchors"], key=lambda item: item["anchor_idx"])
        width, height = self._image_width_height(annotation)
        norm_coords = []
        px_coords = []
        for anchor, frame in zip(anchors, anchor_frames):
            norm = (
                anchor.get("human_coord_norm")
                or anchor.get("coord_norm")
                or frame.get("human_coord_norm")
                or frame.get("coord_norm")
            )
            px = (
                anchor.get("human_coord_px")
                or anchor.get("coord_px")
                or frame.get("human_coord_px")
                or frame.get("coord_px")
            )
            if norm is None and px is None:
                raise ValueError(
                    f"Missing human anchor coordinate for {annotation.get('patient_id')}/"
                    f"{annotation.get('trajectory_id')} anchor {anchor.get('anchor_idx')}"
                )
            if norm is None:
                norm = [float(px[0]) / width, float(px[1]) / height]
            if px is None:
                px = [float(norm[0]) * width, float(norm[1]) * height]
            norm_coords.append([float(norm[0]), float(norm[1])])
            px_coords.append([float(px[0]), float(px[1])])
        return (
            torch.tensor(norm_coords, dtype=torch.float32),
            torch.tensor(px_coords, dtype=torch.float32),
        )

    def _load_frame_tensor(self, frame_path: Union[str, Path]) -> torch.Tensor:
        path = _resolve_dataset_path(self.dataset_root, frame_path)
        if not path.exists():
            raise FileNotFoundError(f"SurgWMBench frame does not exist: {path}")
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = F.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            return F.to_tensor(image)

    def __getitem__(self, item: int):
        row = self.rows[item]
        annotation = self._annotation_for_row(row)
        anchor_frames = self._anchor_frames(annotation)
        pixel_values = torch.stack([self._load_frame_tensor(frame["frame_path"]) for frame in anchor_frames])

        if not self.return_metadata and not self.return_trajectory:
            return pixel_values

        sample = {"pixel_values": pixel_values}

        if self.return_trajectory:
            trajectory_norm, trajectory_px = self._anchor_trajectories(annotation, anchor_frames)
            sample.update({
                "trajectory_norm": trajectory_norm,
                "trajectory_px": trajectory_px,
            })

        if self.return_metadata:
            metadata = {
                "dataset_version": row.get("dataset_version"),
                "patient_id": row.get("patient_id"),
                "source_video_id": row.get("source_video_id"),
                "trajectory_id": row.get("trajectory_id"),
                "difficulty": row.get("difficulty"),
                "annotation_path": row.get("annotation_path"),
                "num_frames": row.get("num_frames"),
                "image_size": annotation.get("image_size"),
                "sampled_indices": [frame["local_frame_idx"] for frame in anchor_frames],
                "anchor_frame_paths": [frame["frame_path"] for frame in anchor_frames],
            }
            sample["metadata"] = metadata

        return sample


def make_surgwmbench_anchor_dataloader(
    dataset_root: Union[str, Path] = DEFAULT_SURGWMBENCH_ROOT,
    manifest: Union[str, Path] = "manifests/train.jsonl",
    batch_size: int = 1,
    num_workers: int = 0,
    image_size: int = 256,
    max_samples: Optional[int] = None,
    shuffle: bool = False,
    drop_last: bool = False,
    return_metadata: bool = False,
    return_trajectory: bool = False,
):
    dataset = SurgWMBenchAnchorDataset(
        dataset_root=dataset_root,
        manifest=manifest,
        image_size=image_size,
        max_samples=max_samples,
        return_metadata=return_metadata,
        return_trajectory=return_trajectory,
    )
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
