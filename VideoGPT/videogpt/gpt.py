import os
import itertools
import numpy as np
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl

from .resnet import resnet34
from .attention import AttentionStack, LayerNorm, AddBroadcastPosEmbed
from .utils import shift_dim


def augment_trajectory_context(context_coords, noise_std=0.0, mask_prob=0.0, training=True):
    """Apply training-time noise and point masking to normalized context xy coords."""
    coords = context_coords.float()
    mask = torch.ones(coords.shape[:-1], dtype=coords.dtype, device=coords.device)
    if not training:
        return coords.clamp(0.0, 1.0), mask

    noise_std = float(noise_std)
    mask_prob = float(mask_prob)
    if noise_std > 0.0:
        coords = coords + torch.randn_like(coords) * noise_std
    coords = coords.clamp(0.0, 1.0)
    if mask_prob > 0.0:
        keep = (torch.rand(mask.shape, device=coords.device) >= mask_prob).type_as(mask)
        mask = mask * keep
        coords = coords * mask.unsqueeze(-1)
    return coords, mask


class TrajectoryHead(nn.Module):
    """Predict future normalized xy anchors from encoded context-frame features."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_future_frames,
        n_layers=1,
        use_context_condition=False,
    ):
        super().__init__()
        self.n_future_frames = int(n_future_frames)
        self.use_context_condition = bool(use_context_condition)
        if self.use_context_condition:
            self.context_embedding = nn.Sequential(
                nn.Linear(3, input_dim),
                nn.LayerNorm(input_dim),
                nn.GELU(),
            )
        else:
            self.context_embedding = None
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.n_future_frames * 2),
        )

    def forward(self, frame_cond, context_coords=None, context_mask=None):
        # frame_cond: BTHWC, where H/W are conditioning feature-grid axes.
        pooled = frame_cond.mean(dim=(2, 3))
        if self.use_context_condition:
            if context_coords is None or context_mask is None:
                raise ValueError("context_coords and context_mask are required")
            context = torch.cat(
                [context_coords.type_as(pooled), context_mask.type_as(pooled).unsqueeze(-1)],
                dim=-1,
            )
            pooled = pooled + self.context_embedding(context)
        _, hidden = self.encoder(pooled)
        coords = self.head(hidden[-1])
        coords = coords.view(-1, self.n_future_frames, 2)
        return torch.sigmoid(coords)


class VideoGPT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load VQ-VAE and set all parameters to no grad
        from .vqvae import VQVAE
        from .download import load_vqvae
        if not os.path.exists(args.vqvae):
            self.vqvae = load_vqvae(args.vqvae)
        else:
            self.vqvae = VQVAE.load_from_checkpoint(args.vqvae, weights_only=False)
        for p in self.vqvae.parameters():
            p.requires_grad = False
        self.vqvae.codebook._need_init = False
        self.vqvae.eval()

        # ResNet34 for frame conditioning
        self.use_frame_cond = args.n_cond_frames > 0
        if self.use_frame_cond:
            frame_cond_shape = (args.n_cond_frames,
                                args.resolution // 4,
                                args.resolution // 4,
                                240)
            self.resnet = resnet34(1, (1, 4, 4), resnet_dim=240)
            self.cond_pos_embd = AddBroadcastPosEmbed(
                shape=frame_cond_shape[:-1], embd_dim=frame_cond_shape[-1]
            )
        else:
            frame_cond_shape = None

        self.use_trajectory_head = bool(getattr(args, 'trajectory_head', False))
        self.use_trajectory_condition = bool(getattr(args, 'trajectory_condition', False))
        self.traj_condition_noise_std = float(getattr(args, 'traj_condition_noise_std', 0.0))
        self.traj_condition_mask_prob = float(getattr(args, 'traj_condition_mask_prob', 0.0))
        self.eval_traj_noise_std = 0.0
        self.eval_traj_mask_prob = 0.0
        if self.traj_condition_noise_std < 0.0:
            raise ValueError("traj_condition_noise_std must be non-negative")
        if not 0.0 <= self.traj_condition_mask_prob <= 1.0:
            raise ValueError("traj_condition_mask_prob must be in [0, 1]")
        if self.use_trajectory_head:
            if not self.use_frame_cond:
                raise ValueError("Trajectory head requires n_cond_frames > 0")
            n_future_frames = int(getattr(args, 'sequence_length', 20)) - int(args.n_cond_frames)
            if n_future_frames <= 0:
                raise ValueError("Trajectory head requires future frames after conditioning")
            self.trajectory_head = TrajectoryHead(
                input_dim=frame_cond_shape[-1],
                hidden_dim=int(getattr(args, 'traj_hidden_dim', frame_cond_shape[-1])),
                n_future_frames=n_future_frames,
                n_layers=int(getattr(args, 'traj_layers', 1)),
                use_context_condition=self.use_trajectory_condition,
            )
            self.traj_loss_weight = float(getattr(args, 'traj_loss_weight', 10.0))
        else:
            self.trajectory_head = None
            self.traj_loss_weight = 0.0

        # VideoGPT transformer
        self.shape = self.vqvae.latent_shape

        self.fc_in = nn.Linear(self.vqvae.embedding_dim, args.hidden_dim, bias=False)
        self.fc_in.weight.data.normal_(std=0.02)

        self.attn_stack = AttentionStack(
            self.shape, args.hidden_dim, args.heads, args.layers, args.dropout,
            args.attn_type, args.attn_dropout, args.class_cond_dim, frame_cond_shape
        )

        self.norm = LayerNorm(args.hidden_dim, args.class_cond_dim)

        self.fc_out = nn.Linear(args.hidden_dim, self.vqvae.n_codes, bias=False)
        self.fc_out.weight.data.copy_(torch.zeros(self.vqvae.n_codes, args.hidden_dim))

        # caches for faster decoding (if necessary)
        self.frame_cond_cache = None

        self.save_hyperparameters()

    def get_reconstruction(self, videos):
        return self.vqvae.decode(self.vqvae.encode(videos))

    def encode_frame_cond(self, frame_cond):
        return self.cond_pos_embd(self.resnet(frame_cond))

    def prepare_trajectory_context(self, batch, training):
        if 'anchor_coords_norm' not in batch:
            raise KeyError("Trajectory conditioning requires anchor_coords_norm in batch")
        context_coords = batch['anchor_coords_norm'][:, :self.args.n_cond_frames]
        if training:
            noise_std = self.traj_condition_noise_std
            mask_prob = self.traj_condition_mask_prob
            apply = True
        else:
            noise_std = float(self.eval_traj_noise_std)
            mask_prob = float(self.eval_traj_mask_prob)
            apply = noise_std > 0.0 or mask_prob > 0.0
        return augment_trajectory_context(
            context_coords,
            noise_std=noise_std,
            mask_prob=mask_prob,
            training=apply,
        )

    def predict_trajectory_from_frame_cond(self, frame_cond, context_coords=None, context_mask=None):
        if not self.use_trajectory_head:
            raise RuntimeError("This VideoGPT checkpoint does not have a trajectory head")
        return self.trajectory_head(frame_cond, context_coords, context_mask)

    def predict_trajectory(self, batch):
        if not self.use_frame_cond:
            raise RuntimeError("Trajectory prediction requires frame conditioning")
        video = batch['video']
        frame_cond = self.encode_frame_cond(video[:, :, :self.args.n_cond_frames])
        context_coords = None
        context_mask = None
        if self.use_trajectory_condition:
            context_coords, context_mask = self.prepare_trajectory_context(batch, training=False)
        return self.predict_trajectory_from_frame_cond(frame_cond, context_coords, context_mask)

    def sample(self, n, batch=None):
        device = self.fc_in.weight.device

        cond = dict()
        if self.use_frame_cond or self.args.class_cond:
            assert batch is not None
            video = batch['video']

            if self.args.class_cond:
                label = batch['label']
                cond['class_cond'] = F.one_hot(label, self.args.class_cond_dim).type_as(video)
            if self.use_frame_cond:
                cond['frame_cond'] = video[:, :, :self.args.n_cond_frames]

        samples = torch.zeros((n,) + self.shape).long().to(device)
        idxs = list(itertools.product(*[range(s) for s in self.shape]))

        with torch.no_grad():
            prev_idx = None
            for i, idx in enumerate(tqdm(idxs)):
                batch_idx_slice = (slice(None, None), *[slice(i, i + 1) for i in idx])
                batch_idx = (slice(None, None), *idx)
                embeddings = self.vqvae.codebook.dictionary_lookup(samples)

                if prev_idx is None:
                    # set arbitrary input values for the first token
                    # does not matter what value since it will be shifted anyways
                    embeddings_slice = embeddings[batch_idx_slice]
                    samples_slice = samples[batch_idx_slice]
                else:
                    embeddings_slice = embeddings[prev_idx]
                    samples_slice = samples[prev_idx]

                logits = self(embeddings_slice, samples_slice, cond,
                              decode_step=i, decode_idx=idx)[1]
                # squeeze all possible dim except batch dimension
                logits = logits.squeeze().unsqueeze(0) if logits.shape[0] == 1 else logits.squeeze()
                probs = F.softmax(logits, dim=-1)
                samples[batch_idx] = torch.multinomial(probs, 1).squeeze(-1)

                prev_idx = batch_idx_slice
            samples = self.vqvae.decode(samples)
            samples = torch.clamp(samples, -0.5, 0.5) + 0.5

        return samples # BCTHW in [0, 1]


    def forward(self, x, targets, cond, decode_step=None, decode_idx=None):
        if self.use_frame_cond:
            if decode_step is None:
                cond['frame_cond'] = self.encode_frame_cond(cond['frame_cond'])
            elif decode_step == 0:
                self.frame_cond_cache = self.encode_frame_cond(cond['frame_cond'])
                cond['frame_cond'] = self.frame_cond_cache
            else:
                cond['frame_cond'] = self.frame_cond_cache

        h = self.fc_in(x)
        h = self.attn_stack(h, cond, decode_step, decode_idx)
        h = self.norm(h, cond)
        logits = self.fc_out(h)

        loss = F.cross_entropy(shift_dim(logits, -1, 1), targets)

        return loss, logits

    def _compute_losses(self, batch):
        self.vqvae.eval()
        x = batch['video']

        cond = dict()
        if self.args.class_cond:
            label = batch['label']
            cond['class_cond'] = F.one_hot(label, self.args.class_cond_dim).type_as(x)
        if self.use_frame_cond:
            cond['frame_cond'] = x[:, :, :self.args.n_cond_frames]

        with torch.no_grad():
            targets, x = self.vqvae.encode(x, include_embeddings=True)
            x = shift_dim(x, 1, -1)

        image_loss, _ = self(x, targets, cond)
        total_loss = image_loss
        metrics = {'loss': total_loss, 'image_loss': image_loss}

        if self.use_trajectory_head:
            if 'anchor_coords_norm' not in batch:
                raise KeyError("Trajectory head training requires anchor_coords_norm in batch")
            context_coords = None
            context_mask = None
            if self.use_trajectory_condition:
                context_coords, context_mask = self.prepare_trajectory_context(
                    batch,
                    training=self.training,
                )
            trajectory_pred = self.predict_trajectory_from_frame_cond(
                cond['frame_cond'],
                context_coords,
                context_mask,
            )
            trajectory_target = batch['anchor_coords_norm'][
                :, self.args.n_cond_frames:self.args.n_cond_frames + trajectory_pred.shape[1]
            ].type_as(trajectory_pred)
            trajectory_loss = F.smooth_l1_loss(trajectory_pred, trajectory_target)
            total_loss = image_loss + self.traj_loss_weight * trajectory_loss
            metrics = {
                'loss': total_loss,
                'image_loss': image_loss,
                'traj_loss': trajectory_loss,
            }
            if context_mask is not None:
                metrics['traj_condition_mask_rate'] = 1.0 - context_mask.mean()

        return total_loss, metrics

    def _log_losses(self, stage, metrics, on_step):
        for name, value in metrics.items():
            self.log(
                f'{stage}/{name}',
                value,
                prog_bar=(name == 'loss'),
                on_step=on_step,
                on_epoch=True,
                sync_dist=True,
            )

    def training_step(self, batch, batch_idx):
        loss, metrics = self._compute_losses(batch)
        self._log_losses('train', metrics, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._compute_losses(batch)
        self._log_losses('val', metrics, on_step=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, betas=(0.9, 0.999))
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.args.max_steps)
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--vqvae', type=str, default='kinetics_stride4x4x4',
                            help='path to vqvae ckpt, or model name to download pretrained')
        parser.add_argument('--n_cond_frames', type=int, default=0)
        parser.add_argument('--class_cond', action='store_true')
        parser.add_argument('--trajectory_head', '--trajectory-head',
                            dest='trajectory_head', action='store_true', default=False,
                            help='train a future-anchor trajectory prediction head')
        parser.add_argument('--no_trajectory_head', '--no-trajectory-head',
                            dest='trajectory_head', action='store_false',
                            help='disable trajectory head training')
        parser.add_argument('--traj_loss_weight', '--traj-loss-weight',
                            dest='traj_loss_weight', type=float, default=10.0)
        parser.add_argument('--traj_hidden_dim', '--traj-hidden-dim',
                            dest='traj_hidden_dim', type=int, default=240)
        parser.add_argument('--traj_layers', '--traj-layers',
                            dest='traj_layers', type=int, default=1)
        parser.add_argument('--trajectory_condition', '--trajectory-condition',
                            dest='trajectory_condition', action='store_true', default=False,
                            help='condition trajectory head on the first n_cond_frames coords')
        parser.add_argument('--no_trajectory_condition', '--no-trajectory-condition',
                            dest='trajectory_condition', action='store_false',
                            help='disable context trajectory conditioning')
        parser.add_argument('--traj_condition_noise_std', '--traj-condition-noise-std',
                            dest='traj_condition_noise_std', type=float, default=0.0,
                            help='Gaussian noise std for normalized context trajectory coords')
        parser.add_argument('--traj_condition_mask_prob', '--traj-condition-mask-prob',
                            dest='traj_condition_mask_prob', type=float, default=0.0,
                            help='per-context-anchor random mask probability')

        # VideoGPT hyperparmeters
        parser.add_argument('--hidden_dim', type=int, default=576)
        parser.add_argument('--heads', type=int, default=4)
        parser.add_argument('--layers', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--attn_type', type=str, default='full',
                            choices=['full', 'sparse'])
        parser.add_argument('--attn_dropout', type=float, default=0.3)

        return parser
