import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class SurgWMTrajectoryHeadConfig:
    hidden_size: int
    context_length: int = 5
    segment_length: int = 20
    context_token_grid: int = 16
    future_token_grid: int = 4
    dropout: float = 0.1


class SurgWMTrajectoryHead(nn.Module):
    """Small trajectory head for the SurgWMBench 20-anchor iVideoGPT task."""

    def __init__(
        self,
        hidden_size: int,
        context_length: int = 5,
        segment_length: int = 20,
        context_token_grid: int = 16,
        future_token_grid: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if segment_length <= context_length:
            raise ValueError("segment_length must be greater than context_length.")
        self.config = SurgWMTrajectoryHeadConfig(
            hidden_size=hidden_size,
            context_length=context_length,
            segment_length=segment_length,
            context_token_grid=context_token_grid,
            future_token_grid=future_token_grid,
            dropout=dropout,
        )
        self.context_proj = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.mask_embedding = nn.Parameter(torch.zeros(hidden_size))
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    @property
    def future_length(self) -> int:
        return self.config.segment_length - self.config.context_length

    @property
    def context_prefix_len(self) -> int:
        return self.config.context_length * (1 + self.config.context_token_grid ** 2) - 1

    @property
    def future_block_size(self) -> int:
        return 1 + self.config.future_token_grid ** 2

    def context_frame_token_slices(self) -> list[tuple[int, int]]:
        grid_tokens = self.config.context_token_grid ** 2
        slices = [(0, grid_tokens)]
        for frame_idx in range(1, self.config.context_length):
            start = grid_tokens + (frame_idx - 1) * (1 + grid_tokens)
            slices.append((start, start + 1 + grid_tokens))
        return slices

    def future_sdf_positions(self, sequence_length: Optional[int] = None) -> torch.Tensor:
        positions = self.context_prefix_len + torch.arange(self.future_length) * self.future_block_size
        if sequence_length is not None:
            positions = positions[positions < sequence_length]
        return positions

    def build_conditioned_inputs_embeds(
        self,
        base_model: nn.Module,
        input_ids: torch.Tensor,
        context_trajectory_norm: torch.Tensor,
        context_trajectory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context_trajectory_norm.shape[1] != self.config.context_length:
            raise ValueError(
                f"Expected {self.config.context_length} context trajectory points, "
                f"got {context_trajectory_norm.shape[1]}."
            )
        if context_trajectory_mask is not None and context_trajectory_mask.shape != context_trajectory_norm.shape[:2]:
            raise ValueError(
                f"context_trajectory_mask must have shape {context_trajectory_norm.shape[:2]}, "
                f"got {context_trajectory_mask.shape}."
            )
        embedding_owner = base_model.module if hasattr(base_model, "module") else base_model
        inputs_embeds = embedding_owner.get_input_embeddings()(input_ids).clone()
        trajectory_embeds = self.context_proj(context_trajectory_norm.to(inputs_embeds.dtype))
        mask_embedding = self.mask_embedding.to(inputs_embeds.dtype).view(1, 1, -1)
        trajectory_embeds = trajectory_embeds + mask_embedding * 0.0
        if context_trajectory_mask is not None:
            mask = context_trajectory_mask.to(device=inputs_embeds.device, dtype=torch.bool).unsqueeze(-1)
            trajectory_embeds = torch.where(mask, mask_embedding, trajectory_embeds)
        for frame_idx, (start, end) in enumerate(self.context_frame_token_slices()):
            clipped_end = min(end, inputs_embeds.shape[1])
            if start >= clipped_end:
                continue
            inputs_embeds[:, start:clipped_end] = (
                inputs_embeds[:, start:clipped_end] + trajectory_embeds[:, frame_idx:frame_idx + 1]
            )
        return inputs_embeds

    def _base_outputs(
        self,
        base_model: nn.Module,
        input_ids: torch.Tensor,
        context_trajectory_norm: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        context_trajectory_mask: Optional[torch.Tensor] = None,
    ):
        inputs_embeds = self.build_conditioned_inputs_embeds(
            base_model,
            input_ids,
            context_trajectory_norm,
            context_trajectory_mask=context_trajectory_mask,
        )
        return base_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

    def _trajectory_from_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        positions = self.future_sdf_positions(hidden_states.shape[1]).to(hidden_states.device)
        if positions.numel() != self.future_length:
            raise ValueError(
                f"Sequence length {hidden_states.shape[1]} exposes {positions.numel()} future SDF tokens; "
                f"expected {self.future_length}."
            )
        future_hidden = hidden_states.index_select(1, positions)
        return torch.sigmoid(self.prediction_head(future_hidden))

    def trajectory_losses(
        self,
        pred_norm: torch.Tensor,
        future_trajectory_norm: torch.Tensor,
        context_trajectory_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = future_trajectory_norm.to(pred_norm.dtype)
        trajectory_loss = F.smooth_l1_loss(pred_norm, target)
        pred_full = torch.cat([context_trajectory_norm[:, -1:].to(pred_norm.dtype), pred_norm], dim=1)
        target_full = torch.cat([context_trajectory_norm[:, -1:].to(pred_norm.dtype), target], dim=1)
        velocity_loss = F.smooth_l1_loss(
            pred_full[:, 1:] - pred_full[:, :-1],
            target_full[:, 1:] - target_full[:, :-1],
        )
        return trajectory_loss, velocity_loss

    def forward(
        self,
        base_model: nn.Module,
        input_ids: torch.Tensor,
        context_trajectory_norm: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        future_trajectory_norm: Optional[torch.Tensor] = None,
        context_trajectory_mask: Optional[torch.Tensor] = None,
        loss_context_trajectory_norm: Optional[torch.Tensor] = None,
    ) -> SimpleNamespace:
        base_outputs = self._base_outputs(
            base_model=base_model,
            input_ids=input_ids,
            context_trajectory_norm=context_trajectory_norm,
            labels=labels,
            context_trajectory_mask=context_trajectory_mask,
        )
        pred_norm = self._trajectory_from_hidden_states(base_outputs.hidden_states[-1])
        trajectory_loss = None
        velocity_loss = None
        if future_trajectory_norm is not None:
            trajectory_loss, velocity_loss = self.trajectory_losses(
                pred_norm=pred_norm,
                future_trajectory_norm=future_trajectory_norm,
                context_trajectory_norm=(
                    context_trajectory_norm
                    if loss_context_trajectory_norm is None
                    else loss_context_trajectory_norm
                ),
            )
        return SimpleNamespace(
            base_outputs=base_outputs,
            image_loss=base_outputs.loss,
            trajectory_pred_norm=pred_norm,
            trajectory_loss=trajectory_loss,
            velocity_loss=velocity_loss,
        )

    @torch.no_grad()
    def generate_tokens(
        self,
        base_model: nn.Module,
        input_ids: torch.Tensor,
        context_trajectory_norm: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        context_trajectory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        generated = input_ids
        for _ in range(max_new_tokens):
            outputs = self._base_outputs(
                base_model=base_model,
                input_ids=generated,
                context_trajectory_norm=context_trajectory_norm,
                labels=None,
                context_trajectory_mask=context_trajectory_mask,
            )
            logits = outputs.logits[:, -1, :]
            if temperature <= 0:
                raise ValueError("temperature must be positive.")
            logits = logits / temperature
            if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
                threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            if do_sample:
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    @torch.no_grad()
    def predict_trajectory(
        self,
        base_model: nn.Module,
        input_ids: torch.Tensor,
        context_trajectory_norm: torch.Tensor,
        context_trajectory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self._base_outputs(
            base_model=base_model,
            input_ids=input_ids,
            context_trajectory_norm=context_trajectory_norm,
            labels=None,
            context_trajectory_mask=context_trajectory_mask,
        )
        return self._trajectory_from_hidden_states(outputs.hidden_states[-1])


def _checkpoint_paths(path: Union[str, Path]) -> tuple[Path, Path]:
    path = Path(path)
    if path.suffix:
        return path.with_name("trajectory_head_config.json"), path
    return path / "trajectory_head_config.json", path / "trajectory_head.pt"


def save_trajectory_head(
    output_dir: Union[str, Path],
    trajectory_head: SurgWMTrajectoryHead,
    save_function=torch.save,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path, state_path = _checkpoint_paths(output_dir)
    config_path.write_text(json.dumps(asdict(trajectory_head.config), indent=2))
    save_function(trajectory_head.state_dict(), str(state_path))


def load_trajectory_head(
    path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = "cpu",
) -> SurgWMTrajectoryHead:
    config_path, state_path = _checkpoint_paths(path)
    config = SurgWMTrajectoryHeadConfig(**json.loads(config_path.read_text()))
    trajectory_head = SurgWMTrajectoryHead(**asdict(config))
    state_dict = torch.load(state_path, map_location=map_location)
    missing, unexpected = trajectory_head.load_state_dict(state_dict, strict=False)
    if unexpected or [key for key in missing if key != "mask_embedding"]:
        raise RuntimeError(
            f"Could not load trajectory head cleanly: missing={missing}, unexpected={unexpected}"
        )
    return trajectory_head
