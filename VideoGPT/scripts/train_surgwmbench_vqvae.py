import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from videogpt import VQVAE
from videogpt.surgwmbench_data import (
    SurgWMBenchDataModule,
    add_surgwmbench_data_args,
)


def add_trainer_args(parser):
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--strategy", type=str, default="auto")
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
    parser = VQVAE.add_model_specific_args(parser)
    parser = add_surgwmbench_data_args(parser)
    args = parser.parse_args()

    if args.sequence_length != 20:
        raise ValueError("This SurgWMBench task requires sequence_length=20")

    data = SurgWMBenchDataModule(args)
    model = VQVAE(args)

    callbacks = [
        ModelCheckpoint(monitor="val/recon_loss", mode="min", save_top_k=-1)
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
