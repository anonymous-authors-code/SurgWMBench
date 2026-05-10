import json
import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data as data


@dataclass(frozen=True)
class LetterboxGeometry:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int

    def to_tensor(self):
        return torch.tensor([
            self.original_height,
            self.original_width,
            self.resized_height,
            self.resized_width,
            self.pad_top,
            self.pad_bottom,
            self.pad_left,
            self.pad_right,
        ], dtype=torch.long)


def resolve_dataset_path(dataset_root, relative_path):
    root = Path(dataset_root)
    path = root / relative_path
    if path.exists():
        return path

    fallback = root / str(relative_path).replace("\uf021", "*")
    matches = sorted(fallback.parent.glob(fallback.name))
    if matches:
        return matches[0]
    return path


def load_rgb_image(path):
    image = imageio.imread(path)
    if image.ndim == 2:
        image = image[..., None].repeat(3, axis=2)
    if image.shape[2] == 4:
        image = image[..., :3]
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return tensor.contiguous()


def letterbox_frame(frame, resolution):
    if frame.ndim != 3:
        raise ValueError(f"Expected CHW frame, got shape {tuple(frame.shape)}")

    _, height, width = frame.shape
    scale = float(resolution) / float(max(height, width))
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))

    resized = F.interpolate(
        frame.unsqueeze(0),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    pad_top = (resolution - resized_height) // 2
    pad_bottom = resolution - resized_height - pad_top
    pad_left = (resolution - resized_width) // 2
    pad_right = resolution - resized_width - pad_left
    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), value=0.5)

    geometry = LetterboxGeometry(
        original_height=height,
        original_width=width,
        resized_height=resized_height,
        resized_width=resized_width,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
    )
    return padded, geometry


def restore_letterboxed_frame(frame, geometry, input_range="zero_one"):
    if input_range == "minus_half_half":
        frame = frame + 0.5
    elif input_range != "zero_one":
        raise ValueError(f"Unsupported input_range: {input_range}")

    if isinstance(geometry, torch.Tensor):
        values = [int(v) for v in geometry.detach().cpu().tolist()]
        geometry = LetterboxGeometry(*values)

    frame = frame.clamp(0.0, 1.0)
    top = geometry.pad_top
    left = geometry.pad_left
    bottom = top + geometry.resized_height
    right = left + geometry.resized_width
    cropped = frame[:, top:bottom, left:right]
    restored = F.interpolate(
        cropped.unsqueeze(0),
        size=(geometry.original_height, geometry.original_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return restored.clamp(0.0, 1.0)


class SurgWMBenchAnchorDataset(data.Dataset):
    """SurgWMBench sparse 20-anchor frame dataset for VideoGPT.

    The returned video tensor is CTHW in [-0.5, 0.5]. The 20 frames are the
    human-labeled anchors identified by sampled_indices, not dense clip frames.
    """

    def __init__(
        self,
        dataset_root,
        manifest,
        resolution=128,
        sequence_length=20,
        max_clips=None,
        strict=True,
    ):
        self.dataset_root = Path(dataset_root)
        self.manifest = Path(manifest)
        if not self.manifest.is_absolute():
            self.manifest = self.dataset_root / self.manifest
        self.resolution = int(resolution)
        self.sequence_length = int(sequence_length)
        self.strict = strict

        with self.manifest.open("r") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        if max_clips is not None:
            self.rows = self.rows[: int(max_clips)]

        if self.sequence_length != 20:
            raise ValueError("SurgWMBench sparse-anchor VideoGPT uses sequence_length=20")

    def __len__(self):
        return len(self.rows)

    def _load_annotation(self, row):
        annotation_path = resolve_dataset_path(self.dataset_root, row["annotation_path"])
        with annotation_path.open("r") as f:
            return json.load(f), annotation_path

    def __getitem__(self, idx):
        row = self.rows[idx]
        annotation, annotation_path = self._load_annotation(row)

        sampled_indices = row.get("sampled_indices") or annotation["sampled_indices"]
        if len(sampled_indices) != self.sequence_length:
            raise ValueError(
                f"Expected {self.sequence_length} sampled_indices, got {len(sampled_indices)}"
            )
        if annotation.get("num_human_anchors") != self.sequence_length:
            raise ValueError(
                f"Expected {self.sequence_length} human anchors in {annotation_path}"
            )

        frames = annotation["frames"]
        human_anchors = annotation.get("human_anchors")
        if human_anchors is None or len(human_anchors) != self.sequence_length:
            raise ValueError(
                f"Expected {self.sequence_length} human_anchors in {annotation_path}"
            )

        tensors = []
        frame_paths = []
        geometries = []
        anchor_coords_px = []
        anchor_coords_norm = []
        for anchor_idx, local_frame_idx in enumerate(sampled_indices):
            frame_record = frames[int(local_frame_idx)]
            anchor_record = human_anchors[anchor_idx]
            if self.strict:
                if frame_record["local_frame_idx"] != int(local_frame_idx):
                    raise ValueError(f"Frame index mismatch in {annotation_path}")
                if not frame_record["is_human_labeled"]:
                    raise ValueError(f"sampled_indices entry is not human-labeled: {annotation_path}")
                if frame_record["anchor_idx"] != anchor_idx:
                    raise ValueError(f"Anchor index mismatch in {annotation_path}")
                if anchor_record["anchor_idx"] != anchor_idx:
                    raise ValueError(f"Human anchor index mismatch in {annotation_path}")
                if anchor_record["local_frame_idx"] != int(local_frame_idx):
                    raise ValueError(f"Human anchor frame mismatch in {annotation_path}")

            coord_px = anchor_record.get("coord_px", frame_record.get("human_coord_px"))
            coord_norm = anchor_record.get("coord_norm", frame_record.get("human_coord_norm"))
            if coord_px is None or coord_norm is None:
                raise ValueError(f"Missing human anchor coordinates in {annotation_path}")
            if len(coord_px) != 2 or len(coord_norm) != 2:
                raise ValueError(f"Expected xy coordinates in {annotation_path}")

            frame_path = resolve_dataset_path(self.dataset_root, frame_record["frame_path"])
            frame = load_rgb_image(frame_path)
            frame, geometry = letterbox_frame(frame, self.resolution)
            tensors.append(frame - 0.5)
            frame_paths.append(str(frame_path))
            geometries.append(geometry.to_tensor())
            anchor_coords_px.append(torch.tensor(coord_px, dtype=torch.float32))
            anchor_coords_norm.append(torch.tensor(coord_norm, dtype=torch.float32))

        return {
            "video": torch.stack(tensors, dim=1),
            "geometry": geometries[0],
            "frame_geometries": torch.stack(geometries, dim=0),
            "anchor_local_frame_indices": torch.tensor(sampled_indices, dtype=torch.long),
            "anchor_coords_px": torch.stack(anchor_coords_px, dim=0),
            "anchor_coords_norm": torch.stack(anchor_coords_norm, dim=0),
            "frame_paths": frame_paths,
            "patient_id": row["patient_id"],
            "source_video_id": row["source_video_id"],
            "trajectory_id": row["trajectory_id"],
            "difficulty": row.get("difficulty") or "unknown",
            "annotation_path": str(annotation_path),
        }


def surgwmbench_collate(batch):
    output = {
        "video": torch.stack([item["video"] for item in batch], dim=0),
        "geometry": torch.stack([item["geometry"] for item in batch], dim=0),
        "frame_geometries": torch.stack([item["frame_geometries"] for item in batch], dim=0),
        "anchor_local_frame_indices": torch.stack(
            [item["anchor_local_frame_indices"] for item in batch], dim=0
        ),
        "anchor_coords_px": torch.stack([item["anchor_coords_px"] for item in batch], dim=0),
        "anchor_coords_norm": torch.stack([item["anchor_coords_norm"] for item in batch], dim=0),
    }
    for key in [
        "frame_paths",
        "patient_id",
        "source_video_id",
        "trajectory_id",
        "difficulty",
        "annotation_path",
    ]:
        output[key] = [item[key] for item in batch]
    return output


class SurgWMBenchDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    @property
    def n_classes(self):
        return 0

    def _dataset(self, manifest):
        return SurgWMBenchAnchorDataset(
            dataset_root=self.args.dataset_root,
            manifest=manifest,
            resolution=self.args.resolution,
            sequence_length=self.args.sequence_length,
            max_clips=getattr(self.args, "max_clips", None),
        )

    def _dataloader(self, manifest, shuffle):
        dataset = self._dataset(manifest)
        return data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            collate_fn=surgwmbench_collate,
        )

    def train_dataloader(self):
        return self._dataloader(self.args.train_manifest, True)

    def val_dataloader(self):
        return self._dataloader(self.args.val_manifest, False)

    def test_dataloader(self):
        manifest = getattr(self.args, "test_manifest", self.args.val_manifest)
        return self._dataloader(manifest, False)


def add_surgwmbench_data_args(parser):
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/mnt/hdd1/neurips2026_dataset_track/SurgWMBench",
    )
    parser.add_argument("--train-manifest", type=str, default="manifests/train.jsonl")
    parser.add_argument("--val-manifest", type=str, default="manifests/val.jsonl")
    parser.add_argument("--test-manifest", type=str, default="manifests/test.jsonl")
    parser.add_argument("--sequence_length", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max-clips", type=int, default=None)
    return parser
