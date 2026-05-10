import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from videogpt import VideoGPT
from videogpt.surgwmbench_data import (
    SurgWMBenchDataModule,
    add_surgwmbench_data_args,
)


def add_trainer_args(parser):
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Use ddp_find_unused_parameters_false for multi-device gradient checkpointing runs.",
    )
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--default_root_dir", type=str, default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--limit_train_batches", default=1.0)
    parser.add_argument("--limit_val_batches", default=1.0)
    parser.add_argument("--num_sanity_val_steps", type=int, default=2)
    return parser


def parse_limit(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return int(value)
    return int(parsed) if parsed > 1 and parsed.is_integer() else parsed


def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = add_trainer_args(parser)
    parser = VideoGPT.add_model_specific_args(parser)
    parser.set_defaults(
        vqvae=None,
        n_cond_frames=5,
        trajectory_head=True,
        trajectory_condition=True,
        traj_condition_noise_std=0.01,
        traj_condition_mask_prob=0.15,
    )
    parser = add_surgwmbench_data_args(parser)
    args = parser.parse_args()

    if args.vqvae is None:
        raise ValueError("Pass --vqvae with a SurgWMBench 20-frame VQ-VAE checkpoint")
    if args.sequence_length != 20:
        raise ValueError("This SurgWMBench task requires sequence_length=20")
    if args.n_cond_frames != 5:
        raise ValueError("This task is defined as 5-frame-conditioned prediction")
    if args.class_cond:
        raise ValueError("SurgWMBench image prediction does not use class conditioning")

    args.class_cond_dim = None
    data = SurgWMBenchDataModule(args)
    model = VideoGPT(args)

    callbacks = [
        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=-1, every_n_epochs=10)
    ]
    trainer = pl.Trainer(
        callbacks=callbacks,
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
        default_root_dir=args.default_root_dir,
        log_every_n_steps=args.log_every_n_steps,
        limit_train_batches=parse_limit(args.limit_train_batches),
        limit_val_batches=parse_limit(args.limit_val_batches),
        num_sanity_val_steps=args.num_sanity_val_steps,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
