import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXTransformer3DModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import cast_training_params
from diffusers.utils.torch_utils import is_compiled_module
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

import diffusers
from finetune.dataset.surgwmbench_anchor_dataset import (
    SurgWMBenchAnchorDataset,
    latent_frame_count,
    pad_anchor_video,
    surgwmbench_anchor_collate,
)
from finetune.trajectory_head import SurgWMBenchTrajectoryHead


logger = get_logger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HieraSurg/CogVideoX on SurgWMBench 20-anchor prediction.")
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-manifest", default="manifests/train.jsonl")
    parser.add_argument("--val-manifest", default="manifests/val.jsonl")
    parser.add_argument("--output_dir", default="outputs/surgwmbench_anchor_i2v")
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--context-anchors", type=int, default=5)
    parser.add_argument("--prediction-anchors", type=int, default=15)
    parser.add_argument("--model-num-frames", type=int, default=33)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--report_to", default=None)
    parser.add_argument("--logging_dir", default="logs")
    parser.add_argument("--tracker_name", default="surgwmbench-anchor-i2v")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--full_finetune", action="store_true")
    parser.add_argument("--train_limit", type=int, default=None, help="Optional manifest row limit for smoke runs.")
    parser.add_argument("--enable_slicing", action="store_true")
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument("--trajectory_hidden_dim", type=int, default=256)
    parser.add_argument("--trajectory_dropout", type=float, default=0.0)
    parser.add_argument("--trajectory_loss_weight", type=float, default=1.0)
    parser.add_argument("--disable_trajectory_head", action="store_true")
    parser.add_argument(
        "--trajectory_coord_noise_std",
        type=float,
        default=0.01,
        help="Gaussian noise std for normalized context trajectory coordinates during joint training.",
    )
    parser.add_argument(
        "--trajectory_coord_mask_prob",
        type=float,
        default=0.15,
        help="Per-context-point random masking probability for trajectory coordinates during joint training.",
    )
    return parser.parse_args()


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int,
    patch_size: int,
    attention_head_dim: int,
    device: torch.device,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)
    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )
    return freqs_cos.to(device=device), freqs_sin.to(device=device)


def encode_video_latents(vae: AutoencoderKLCogVideoX, video: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        video = video.to(device=vae.device, dtype=vae.dtype)
        encoded = vae.encode(video)
        latent_dist = encoded.latent_dist if hasattr(encoded, "latent_dist") else encoded[0]
        return latent_dist.sample() * vae.config.scaling_factor


def zero_prompt_embeds(batch_size: int, model_config, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(
        (batch_size, model_config.max_text_seq_length, getattr(model_config, "text_embed_dim", 4096)),
        device=device,
        dtype=dtype,
    )


def set_trainable_params(transformer: CogVideoXTransformer3DModel, full_finetune: bool) -> None:
    if full_finetune:
        transformer.requires_grad_(True)
        return

    transformer.requires_grad_(False)
    trainable_fragments = ("ff.net", "attn1", "attn2", "proj_out", "pos_embed", "timepositionalencoding")
    for name, param in transformer.named_parameters():
        if any(fragment in name for fragment in trainable_fragments):
            param.requires_grad = True


def unwrap_model(accelerator: Accelerator, model: torch.nn.Module) -> torch.nn.Module:
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def checkpoint_step(path: Path) -> Optional[int]:
    try:
        return int(path.name.split("-")[-1])
    except ValueError:
        return None


def save_checkpoint(
    accelerator: Accelerator,
    transformer: torch.nn.Module,
    trajectory_head: Optional[torch.nn.Module],
    args: argparse.Namespace,
    output_dir: str,
    global_step: int,
) -> None:
    if not accelerator.is_main_process:
        return

    if args.checkpoints_total_limit is not None:
        checkpoints = sorted(
            [p for p in Path(args.output_dir).glob("checkpoint-*") if p.is_dir() and checkpoint_step(p) is not None],
            key=lambda p: checkpoint_step(p) or -1,
        )
        if len(checkpoints) >= args.checkpoints_total_limit:
            for path in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                import shutil

                shutil.rmtree(path)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    save_path.mkdir(parents=True, exist_ok=True)
    unwrap_model(accelerator, transformer).save_pretrained(save_path / "transformer")
    if trajectory_head is not None:
        unwrap_model(accelerator, trajectory_head).save_checkpoint(save_path)
    with (save_path / "training_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    logger.info(f"Saved checkpoint to {save_path}")


def make_dataloader(args: argparse.Namespace, accelerator: Accelerator) -> DataLoader:
    dataset = SurgWMBenchAnchorDataset(
        dataset_root=args.dataset_root,
        manifest=args.train_manifest,
        height=args.height,
        width=args.width,
        context_anchors=args.context_anchors,
        prediction_anchors=args.prediction_anchors,
        limit=args.train_limit,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=args.seed,
    )
    return DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=False,
        drop_last=True,
        collate_fn=surgwmbench_anchor_collate,
        persistent_workers=args.dataloader_num_workers > 0,
    )


def load_checkpoint_weights(
    accelerator: Accelerator,
    transformer: torch.nn.Module,
    trajectory_head: Optional[torch.nn.Module],
    checkpoint: Path,
    torch_dtype: torch.dtype,
) -> None:
    transformer_path = checkpoint / "transformer"
    if transformer_path.is_dir():
        loaded_transformer = CogVideoXTransformer3DModel.from_pretrained(transformer_path, torch_dtype=torch_dtype)
        unwrap_model(accelerator, transformer).load_state_dict(loaded_transformer.state_dict())
        del loaded_transformer

    trajectory_path = checkpoint / "trajectory_head.pt"
    if trajectory_head is not None and trajectory_path.exists():
        loaded_head = SurgWMBenchTrajectoryHead.from_checkpoint(checkpoint)
        unwrap_model(accelerator, trajectory_head).load_state_dict(loaded_head.state_dict())


def augment_context_coords(
    context_coords_norm: torch.Tensor,
    noise_std: float,
    mask_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if noise_std < 0:
        raise ValueError("--trajectory_coord_noise_std must be non-negative.")
    if mask_prob < 0 or mask_prob > 1:
        raise ValueError("--trajectory_coord_mask_prob must be in [0, 1].")

    coords = context_coords_norm
    if noise_std > 0:
        coords = coords + torch.randn_like(coords) * noise_std
        coords = coords.clamp(0, 1)

    mask = torch.ones(coords.shape[:2], device=coords.device, dtype=coords.dtype)
    if mask_prob > 0:
        mask = (torch.rand(coords.shape[:2], device=coords.device) >= mask_prob).to(dtype=coords.dtype)
    coords = coords * mask.unsqueeze(-1)
    return coords, mask


def main(args: argparse.Namespace) -> None:
    if args.context_anchors + args.prediction_anchors != 20:
        raise ValueError("This task expects 5 context anchors plus 15 prediction anchors, totaling 20 anchors.")
    if args.model_num_frames < 20:
        raise ValueError("--model-num-frames must be at least 20.")
    if (args.model_num_frames - 1) % 4 != 0:
        raise ValueError("--model-num-frames should satisfy (N - 1) % 4 == 0 for CogVideoX temporal compression.")
    if args.trajectory_coord_noise_std < 0:
        raise ValueError("--trajectory_coord_noise_std must be non-negative.")
    if args.trajectory_coord_mask_prob < 0 or args.trajectory_coord_mask_prob > 1:
        raise ValueError("--trajectory_coord_mask_prob must be in [0, 1].")

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
        revision=args.revision,
        variant=args.variant,
    )
    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    vae.requires_grad_(False)
    set_trainable_params(transformer, args.full_finetune)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)

    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    trajectory_head: Optional[SurgWMBenchTrajectoryHead] = None
    if not args.disable_trajectory_head:
        latent_channels = getattr(transformer.config, "in_channels", getattr(vae.config, "latent_channels", 16))
        trajectory_head = SurgWMBenchTrajectoryHead(
            latent_channels=latent_channels,
            context_anchors=args.context_anchors,
            prediction_anchors=args.prediction_anchors,
            hidden_dim=args.trajectory_hidden_dim,
            dropout=args.trajectory_dropout,
        )
        trajectory_head.to(accelerator.device, dtype=weight_dtype)
        if args.mixed_precision == "fp16":
            cast_training_params([trajectory_head], dtype=torch.float32)

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    if trajectory_head is not None:
        trainable_params.extend(trajectory_head.parameters())
    num_trainable_params = sum(p.numel() for p in trainable_params)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = make_dataloader(args, accelerator)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    if trajectory_head is not None:
        transformer, trajectory_head, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, trajectory_head, optimizer, train_dataloader, lr_scheduler
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    if trajectory_head is not None:
        trainable_params.extend(trajectory_head.parameters())

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process and args.report_to:
        accelerator.init_trackers(args.tracker_name, config=vars(args))

    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    temporal_ratio = vae.config.temporal_compression_ratio
    context_latent_frames = latent_frame_count(args.context_anchors, temporal_ratio)
    valid_latent_frames = latent_frame_count(args.context_anchors + args.prediction_anchors, temporal_ratio)

    logger.info("***** Running SurgWMBench anchor training *****")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num trainable parameters = {num_trainable_params}")
    logger.info(f"  Context latent frames = {context_latent_frames}")
    logger.info(f"  Valid latent frames = {valid_latent_frames}")
    if trajectory_head is not None:
        logger.info(f"  Trajectory coord noise std = {args.trajectory_coord_noise_std}")
        logger.info(f"  Trajectory coord mask prob = {args.trajectory_coord_mask_prob}")

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        resume_path = args.resume_from_checkpoint
        if resume_path == "latest":
            checkpoints = sorted(
                [p for p in Path(args.output_dir).glob("checkpoint-*") if p.is_dir() and checkpoint_step(p) is not None],
                key=lambda p: checkpoint_step(p) or -1,
            )
            resume_path = str(checkpoints[-1]) if checkpoints else None
        if resume_path:
            resume_checkpoint = Path(resume_path)
            try:
                accelerator.load_state(resume_path)
            except Exception as exc:
                logger.warning(f"Could not load full accelerator state from {resume_path}: {exc}")
                load_checkpoint_weights(accelerator, transformer, trajectory_head, resume_checkpoint, weight_dtype)
            else:
                load_checkpoint_weights(accelerator, transformer, trajectory_head, resume_checkpoint, weight_dtype)
            global_step = checkpoint_step(Path(resume_path)) or 0
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        initial=global_step,
        total=args.max_train_steps,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        if trajectory_head is not None:
            trajectory_head.train()
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        for batch in train_dataloader:
            accumulate_models = (transformer, trajectory_head) if trajectory_head is not None else (transformer,)
            with accelerator.accumulate(*accumulate_models):
                video = pad_anchor_video(batch["anchor_frames"], args.model_num_frames)
                video = video.to(device=accelerator.device, dtype=torch.float32)
                latents = encode_video_latents(vae, video).to(dtype=weight_dtype).permute(0, 2, 1, 3, 4)

                batch_size, latent_frames, channels, latent_height, latent_width = latents.shape
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=accelerator.device,
                    dtype=torch.long,
                )
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                model_input = noisy_latents.clone()
                model_input[:, :context_latent_frames] = latents[:, :context_latent_frames]

                prompt_embeds = zero_prompt_embeds(batch_size, model_config, accelerator.device, weight_dtype)
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=args.height,
                        width=args.width,
                        num_frames=latent_frames,
                        vae_scale_factor_spatial=vae_scale_factor_spatial,
                        patch_size=model_config.patch_size,
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                )

                model_output = transformer(
                    hidden_states=model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
                model_pred = scheduler.get_velocity(model_output, noisy_latents, timesteps)

                alphas_cumprod = scheduler.alphas_cumprod.to(device=accelerator.device, dtype=weight_dtype)
                weights = 1 / (1 - alphas_cumprod[timesteps])
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                loss_mask = torch.zeros((1, latent_frames, 1, 1, 1), device=accelerator.device, dtype=weight_dtype)
                loss_mask[:, context_latent_frames:valid_latent_frames] = 1
                loss_values = weights * (model_pred - latents) ** 2 * loss_mask
                denom = loss_mask.sum() * batch_size * channels * latent_height * latent_width
                image_loss = loss_values.sum() / denom.clamp_min(1)

                trajectory_loss = None
                loss = image_loss
                if trajectory_head is not None:
                    trajectory_context_video = pad_anchor_video(batch["context_frames"], args.model_num_frames)
                    trajectory_context_video = trajectory_context_video.to(device=accelerator.device, dtype=torch.float32)
                    trajectory_context_latents = (
                        encode_video_latents(vae, trajectory_context_video)
                        .to(dtype=weight_dtype)
                        .permute(0, 2, 1, 3, 4)[:, :context_latent_frames]
                    )
                    context_coords_norm = batch["context_coords_norm"].to(device=accelerator.device, dtype=weight_dtype)
                    target_coords_norm = batch["target_coords_norm"].to(device=accelerator.device, dtype=weight_dtype)
                    context_coords_aug, context_coord_mask = augment_context_coords(
                        context_coords_norm,
                        args.trajectory_coord_noise_std,
                        args.trajectory_coord_mask_prob,
                    )
                    pred_coords_norm = trajectory_head(
                        trajectory_context_latents,
                        context_coords_aug,
                        context_coord_mask,
                    )
                    trajectory_loss = F.smooth_l1_loss(pred_coords_norm.float(), target_coords_norm.float())
                    loss = image_loss + args.trajectory_loss_weight * trajectory_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "image_loss": image_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if trajectory_loss is not None:
                    logs["trajectory_loss"] = trajectory_loss.detach().item()
                    logs["trajectory_coord_keep_ratio"] = context_coord_mask.detach().float().mean().item()
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    save_checkpoint(accelerator, transformer, trajectory_head, args, args.output_dir, global_step)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = Path(args.output_dir) / "checkpoint-final"
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrap_model(accelerator, transformer).save_pretrained(final_dir / "transformer")
        if trajectory_head is not None:
            unwrap_model(accelerator, trajectory_head).save_checkpoint(final_dir)
        with (final_dir / "training_args.json").open("w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
        logger.info(f"Saved final checkpoint to {final_dir}")


if __name__ == "__main__":
    main(get_args())
