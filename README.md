# SurgWMBench: A Benchmark for Surgical Video World Models

> Anonymous submission for double-blind review. Author and institutional
> information has been removed from this repository in accordance with the
> conference's anonymity guidelines.

This repository contains the code and evaluation harness for **SurgWMBench**, a
benchmark for evaluating world-model architectures on surgical video data.

## Repository layout

This benchmark vendors four baseline codebases as subdirectories. Each has been
locally adapted for SurgWMBench (data loaders, evaluation hooks, anchor-based
training scripts) while preserving the original methods and licenses.

| Directory | Baseline method | Original license |
|---|---|---|
| `HieraSurg/` | HieraSurg (hierarchy-aware diffusion model) | See `HieraSurg/` headers |
| `iVideoGPT/` | iVideoGPT (interactive VideoGPT) | MIT (see `iVideoGPT/LICENSE`) |
| `SurgSora/` | SurgSora (object-aware controllable diffusion) | See `SurgSora/` headers |
| `VideoGPT/` | VideoGPT (VQ-VAE + transformer) | MIT (see `VideoGPT/LICENSE`) |

Per-baseline setup, training, and evaluation instructions are in each
subdirectory's `README.md`.

## Pretrained checkpoints and benchmark splits

Model checkpoints, intermediate logs, and large preprocessed data are **not**
included in this repository. They are hosted externally; the anonymous URL
will be provided here for the camera-ready release. During the review period,
reviewers can reproduce results from scratch using the per-baseline training
scripts and the dataset preparation code under each `*/datasets/` or
equivalent path.

## Acknowledgements

This benchmark builds on the four baseline codebases listed above. Their
original authors and citations are preserved in each subdirectory's
`README.md` and `LICENSE`.
