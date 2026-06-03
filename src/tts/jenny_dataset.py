"""
jenny_dataset.py
----------------
A PyTorch Dataset that reads the .pt files produced by process_jenny.py.

Example
-------
    from jenny_dataset import JennyDataset
    from torch.utils.data import DataLoader

    ds     = JennyDataset("./jenny_processed")
    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=ds.collate_fn)

    for mel, text in loader:
        # mel:  (B, 1, 100, T_max)  — zero-padded to longest in batch
        # text: list[str] of length B
        ...
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


class JennyDataset(Dataset):
    """
    Loads pre-processed Jenny TTS samples from a directory of .pt files.

    Each .pt file is expected to contain a dict:
        {
            "mel":  Tensor of shape (1, 100, T) stored as float16,
            "text": str,
            "idx":  int,
        }

    Mels are cast to float32 on load, so your model always receives full
    precision tensors regardless of how they were stored.

    Parameters
    ----------
    root_dir : str | Path
        Directory that contains the ``samples/`` sub-folder.
    max_frames : int, optional
        If set, samples longer than this many mel frames are dropped.
        Useful to avoid very long sequences blowing up your GPU memory.
    """

    def __init__(self, root_dir: str | Path, max_frames: Optional[int] = None):
        self.samples_dir = Path(root_dir) / "samples"
        self.max_frames  = max_frames

        all_files = sorted(self.samples_dir.glob("*.pt"))
        if not all_files:
            raise FileNotFoundError(f"No .pt files found in {self.samples_dir}")

        # Optionally pre-filter by length (requires loading metadata once)
        # For large datasets keep max_frames=None and filter in collate_fn instead
        self.files = all_files
        print(f"[JennyDataset] {len(self.files)} samples found in {self.samples_dir}")

    # ── Core interface ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> tuple[Tensor, str]:
        sample = torch.load(self.files[i], weights_only=True)
        mel: Tensor = sample["mel"].float()  # stored as float16, cast to float32 for training
        text: str   = sample["text"]

        if self.max_frames is not None and mel.shape[-1] > self.max_frames:
            mel = mel[..., : self.max_frames]

        return mel, text

    # ── Collate (handles variable-length mel sequences) ───────────────────────

    @staticmethod
    def collate_fn(batch: list[tuple[Tensor, str]]) -> tuple[Tensor, list[str]]:
        """
        Pads mel spectrograms along the time axis to the longest in the batch.

        Returns
        -------
        mels  : Tensor  (B, 1, 100, T_max)
        texts : list[str]
        """
        mels, texts = zip(*batch)
        max_t = max(m.shape[-1] for m in mels)
        padded = torch.stack(
            [F.pad(m, (0, max_t - m.shape[-1])) for m in mels]
        )
        return padded, list(texts)
