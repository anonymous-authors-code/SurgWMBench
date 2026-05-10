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

## License

The SurgWMBench benchmark code, evaluation harness, and integration layer
provided at the top level of this repository are released under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The full legal text is included in the top-level `LICENSE` file.

The four baseline codebases vendored as subdirectories retain their original
licenses, which are unchanged by this release:

- `iVideoGPT/` — MIT (see `iVideoGPT/LICENSE`)
- `VideoGPT/` — MIT (see `VideoGPT/LICENSE`)
- `HieraSurg/` — See per-file copyright headers within `HieraSurg/`
- `SurgSora/` — See per-file copyright headers within `SurgSora/`

Use of any baseline subdirectory must comply with both its original license
and the CC BY-NC-SA 4.0 terms governing the SurgWMBench benchmark as a whole.
