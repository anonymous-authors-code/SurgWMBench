from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn


def trajectory_checkpoint_file(path: Union[str, Path]) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.suffix == ".pt":
        return checkpoint_path
    return checkpoint_path / "trajectory_head.pt"


class SurgWMBenchTrajectoryHead(nn.Module):
    """Predict future normalized trajectory points from context video latents and coordinates."""

    def __init__(
        self,
        latent_channels: int,
        context_anchors: int = 5,
        prediction_anchors: int = 15,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        coord_condition_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if coord_condition_dim is None:
            coord_condition_dim = context_anchors * 3
        self.config = {
            "latent_channels": latent_channels,
            "context_anchors": context_anchors,
            "prediction_anchors": prediction_anchors,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "coord_condition_dim": coord_condition_dim,
        }
        self.context_anchors = context_anchors
        self.prediction_anchors = prediction_anchors
        self.coord_condition_dim = coord_condition_dim
        self.uses_coord_mask = coord_condition_dim == context_anchors * 3

        self.latent_norm = nn.LayerNorm(latent_channels)
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_proj = nn.Sequential(
            nn.Linear(coord_condition_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden_dim, prediction_anchors * 2)

    def forward(
        self,
        context_latents: torch.Tensor,
        context_coords_norm: torch.Tensor,
        context_coord_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context_latents.ndim != 5:
            raise ValueError(f"Expected context_latents [B,T,C,H,W], got {tuple(context_latents.shape)}")
        if context_coords_norm.shape[1:] != (self.context_anchors, 2):
            raise ValueError(
                f"Expected context_coords_norm [B,{self.context_anchors},2], "
                f"got {tuple(context_coords_norm.shape)}"
            )
        if context_coord_mask is not None and context_coord_mask.shape[1:] not in (
            (self.context_anchors,),
            (self.context_anchors, 1),
        ):
            raise ValueError(
                f"Expected context_coord_mask [B,{self.context_anchors}] or [B,{self.context_anchors},1], "
                f"got {tuple(context_coord_mask.shape)}"
            )

        latent_tokens = context_latents.mean(dim=(-1, -2))
        latent_tokens = self.latent_norm(latent_tokens)
        latent_features = self.latent_proj(latent_tokens).mean(dim=1)

        coords = context_coords_norm.to(device=context_latents.device, dtype=context_latents.dtype)
        if context_coord_mask is None:
            mask = torch.ones(coords.shape[:2], device=coords.device, dtype=coords.dtype)
        else:
            mask = context_coord_mask.to(device=coords.device, dtype=coords.dtype)
            if mask.ndim == 3:
                mask = mask.squeeze(-1)

        coords = coords * mask.unsqueeze(-1)
        if self.uses_coord_mask:
            coord_condition = torch.cat([coords, mask.unsqueeze(-1)], dim=-1)
        else:
            coord_condition = coords
        coord_features = self.coord_proj(coord_condition.reshape(coord_condition.shape[0], -1))
        fused = self.fusion(torch.cat([latent_features, coord_features], dim=-1))
        return self.out(fused).reshape(context_latents.shape[0], self.prediction_anchors, 2).sigmoid()

    def checkpoint_payload(self) -> Dict[str, Any]:
        return {"config": dict(self.config), "state_dict": self.state_dict()}

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        checkpoint_file = trajectory_checkpoint_file(path)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), checkpoint_file)

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path], map_location: str = "cpu") -> "SurgWMBenchTrajectoryHead":
        checkpoint_file = trajectory_checkpoint_file(path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Missing trajectory head checkpoint: {checkpoint_file}")

        payload = torch.load(checkpoint_file, map_location=map_location, weights_only=False)
        config = payload["config"]
        if "coord_condition_dim" not in config:
            input_dim = payload["state_dict"]["coord_proj.0.weight"].shape[1]
            config["coord_condition_dim"] = input_dim
        model = cls(**config)
        model.load_state_dict(payload["state_dict"])
        return model
