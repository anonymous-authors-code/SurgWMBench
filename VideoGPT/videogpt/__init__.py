__all__ = [
    "VQVAE",
    "VideoGPT",
    "VideoData",
    "SurgWMBenchAnchorDataset",
    "SurgWMBenchDataModule",
    "load_vqvae",
    "load_videogpt",
    "load_i3d_pretrained",
    "download",
]


def __getattr__(name):
    if name == "VQVAE":
        from .vqvae import VQVAE
        return VQVAE
    if name == "VideoGPT":
        from .gpt import VideoGPT
        return VideoGPT
    if name == "VideoData":
        from .data import VideoData
        return VideoData
    if name in {"SurgWMBenchAnchorDataset", "SurgWMBenchDataModule"}:
        from .surgwmbench_data import SurgWMBenchAnchorDataset, SurgWMBenchDataModule
        return {
            "SurgWMBenchAnchorDataset": SurgWMBenchAnchorDataset,
            "SurgWMBenchDataModule": SurgWMBenchDataModule,
        }[name]
    if name in {"load_vqvae", "load_videogpt", "load_i3d_pretrained", "download"}:
        from .download import download, load_i3d_pretrained, load_videogpt, load_vqvae
        return {
            "download": download,
            "load_i3d_pretrained": load_i3d_pretrained,
            "load_videogpt": load_videogpt,
            "load_vqvae": load_vqvae,
        }[name]
    raise AttributeError(f"module 'videogpt' has no attribute {name!r}")
