"""
process_jenny.py
----------------
Streams the Jenny TTS dataset from HuggingFace row-by-row, computes mel
spectrograms via your custom vocoder, and saves each sample as a .pt file.

Output layout
-------------
output_dir/
    samples/
        000000.pt   # {"mel": Tensor(1,100,T) float16, "text": str, "idx": int}
        000001.pt
        ...
    progress.json   # tracks completed indices so restarts are safe

Usage
-----
    python process_jenny.py                        # process everything
    python process_jenny.py --limit 500            # first 500 rows only
    python process_jenny.py --output ./my_data     # custom output dir
    python process_jenny.py --start_from 200       # skip first 200 rows
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset, Audio
from tqdm import tqdm

from tts.vocoder import get_mel_vocos, load_model


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output",      type=str, default="./jenny_processed",
                   help="Root output directory")
    p.add_argument("--limit",       type=int, default=None,
                   help="Stop after this many rows (default: all)")
    p.add_argument("--start_from",  type=int, default=0,
                   help="Skip rows before this index (useful for manual range control)")
    p.add_argument("--skip_errors", action="store_true",
                   help="Log errors and continue instead of raising")
    return p.parse_args()


# ── Progress tracker ──────────────────────────────────────────────────────────

class ProgressTracker:
    """
    Persists a set of completed indices to disk so processing can be safely
    resumed after interruption.
    """
    def __init__(self, path: Path):
        self.path = path
        self.completed: set[int] = set()
        if path.exists():
            data = json.loads(path.read_text())
            self.completed = set(data["completed"])
            print(f"[resume] Found {len(self.completed)} already-completed samples")

    def mark_done(self, idx: int):
        self.completed.add(idx)
        # Write after every sample — cheap enough, guarantees no lost progress
        self.path.write_text(json.dumps({"completed": sorted(self.completed)}, indent=2))

    def is_done(self, idx: int) -> bool:
        return idx in self.completed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    output_dir  = Path(args.output)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    progress_file = output_dir / "progress.json"

    # Load vocoder once up front
    print("[setup] Loading vocoder model...")
    load_model()

    tracker = ProgressTracker(progress_file)

    # streaming=True means HuggingFace never downloads the full dataset at once;
    # it fetches shards lazily as you iterate — perfect for a laptop.
    print("[setup] Connecting to HuggingFace dataset (streaming)...")
    dataset = load_dataset(
        "reach-vb/jenny_tts_dataset",
        split="train",
        streaming=True,
    ).cast_column("audio", Audio(decode=False))  # <-- keep raw bytes

    errors = []

    for idx, row in enumerate(tqdm(dataset, desc="Processing")):

        # ── Range control ──────────────────────────────────────────────────
        if idx < args.start_from:
            continue
        if args.limit is not None and idx >= args.start_from + args.limit:
            break

        # ── Skip already-processed rows ────────────────────────────────────
        if tracker.is_done(idx):
            continue

        try:
            # row["audio"] is a dict: {"bytes": bytes, "path": str, ...}
            audio_bytes: bytes = row["audio"]["bytes"]
            text: str          = row["transcription"]

            mel: torch.Tensor  = get_mel_vocos(audio_bytes).half()  # (1, 100, T) float16 — ~half the disk space

            sample = {
                "mel":  mel,
                "text": text,
                "idx":  idx,
            }

            out_path = samples_dir / f"{idx:06d}.pt"
            torch.save(sample, out_path)
            tracker.mark_done(idx)

        except Exception as e:
            msg = f"[error] idx={idx}: {e}"
            errors.append(msg)
            if args.skip_errors:
                tqdm.write(msg)
            else:
                raise

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n✓ Done. {len(tracker.completed)} samples saved to {samples_dir}")
    if errors:
        error_log = output_dir / "errors.log"
        error_log.write_text("\n".join(errors))
        print(f"⚠  {len(errors)} errors logged to {error_log}")


if __name__ == "__main__":
    main()
