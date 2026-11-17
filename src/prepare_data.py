"""Build a balanced image subset from the Xiz9Dataset (Hugging Face).

The full dataset is ~150 GB, so we sample a fixed number of images from
selected parquet shards. Shards 0-16 contain only fake images and shards
17-33 only real ones. A few shards are reserved entirely for validation,
so the validation set comes from sources never seen during training.

Optionally, your own real photos can be added to the training set
(--own-real). They are never uploaded to this repository.

Usage:
    python src/prepare_data.py --out xiz_veri
    python src/prepare_data.py --out xiz_veri --own-real path/to/photos
"""
import argparse
import gc
import io
import os
import random
import shutil

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

REPO = "KarmaLeo/Xiz9Dataset"

# (shard number, class, split, number of images to sample)
# Ordered from small to large shards so a crash loses little work.
PLAN = [
    (26, "Real", "train", 850), (25, "Real", "val", 400), (24, "Real", "train", 850),
    (23, "Real", "train", 850), (27, "Real", "train", 850),
    (8, "Fake", "train", 400), (9, "Fake", "train", 400), (10, "Fake", "train", 400),
    (7, "Fake", "train", 400), (5, "Fake", "val", 400), (13, "Fake", "train", 400),
    (0, "Fake", "train", 400), (14, "Fake", "val", 400), (12, "Fake", "train", 400),
    (16, "Fake", "train", 400), (11, "Fake", "train", 400), (15, "Fake", "train", 400),
    (6, "Fake", "train", 400), (1, "Fake", "train", 400), (4, "Fake", "train", 400),
    (21, "Real", "train", 850), (22, "Real", "val", 400),
    (2, "Fake", "train", 400), (3, "Fake", "train", 400),
    (17, "Real", "train", 850), (19, "Real", "train", 850),
    (29, "Real", "train", 650), (33, "Real", "train", 650),
]

OWN_REAL_WEIGHT = 700  # approximate number of training images from own photos


def sample_shard(shard, cls, split, quota, out_dir):
    """Download one parquet shard, save a random sample, delete the shard."""
    marker = os.path.join(out_dir, f".done_{cls}_{shard}")
    if os.path.exists(marker):
        print(f"{cls} {shard:05d}: already done, skipping")
        return

    path = hf_hub_download(REPO, f"data/train-{shard:05d}.parquet", repo_type="dataset")
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    rng = random.Random(42_000 + shard)  # deterministic per shard, safe to resume
    chosen = set(rng.sample(range(total), min(quota, total)))

    target = os.path.join(out_dir, split, cls)
    written, idx = 0, 0
    for batch in pf.iter_batches(batch_size=50, columns=["image"]):
        for item in batch.column("image").to_pylist():
            if idx in chosen:
                data = item["bytes"]
                ext = "png" if Image.open(io.BytesIO(data)).format == "PNG" else "jpg"
                with open(os.path.join(target, f"{shard:05d}_{idx}.{ext}"), "wb") as f:
                    f.write(data)
                written += 1
            idx += 1
        del batch

    del pf
    gc.collect()
    real_path = os.path.realpath(path)  # remove cached shard to save disk space
    os.remove(path)
    if os.path.exists(real_path):
        os.remove(real_path)
    open(marker, "w").close()
    print(f"{cls} {shard:05d} -> {split}: {written} images written")


def add_own_real(own_dir, out_dir):
    """Add own real photos to the training set (repeated to reach ~OWN_REAL_WEIGHT)."""
    target = os.path.join(out_dir, "train", "Real")
    for name in os.listdir(target):  # remove earlier copies so we never duplicate
        if name.startswith("own_"):
            os.remove(os.path.join(target, name))

    good, bad = [], []
    for root, _, names in os.walk(own_dir):
        for name in names:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, name)
                try:
                    with Image.open(path) as im:
                        im.convert("RGB")
                    good.append(path)
                except Exception:
                    bad.append(name)

    repeat = max(1, round(OWN_REAL_WEIGHT / max(len(good), 1)))
    for i, path in enumerate(good):
        ext = os.path.splitext(path)[1].lower()
        for k in range(repeat):
            shutil.copy(path, os.path.join(target, f"own_{i}_{k}{ext}"))
    print(f"Own photos: {len(good)} usable, unreadable: {bad}, repeat x{repeat}")


def remove_corrupt(out_dir):
    """Delete images that cannot be opened (they would crash training)."""
    corrupt, total = [], 0
    for split in ("train", "val"):
        for cls in ("Fake", "Real"):
            folder = os.path.join(out_dir, split, cls)
            for name in os.listdir(folder):
                path = os.path.join(folder, name)
                total += 1
                try:
                    with Image.open(path) as im:
                        im.convert("RGB")
                except Exception:
                    corrupt.append(path)
    for path in corrupt:
        print("Removing corrupt file:", path)
        os.remove(path)
    print(f"Scanned {total} images, removed {len(corrupt)} corrupt")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="xiz_veri", help="output directory")
    parser.add_argument("--own-real", default=None,
                        help="folder with your own real photos (optional)")
    args = parser.parse_args()

    for split in ("train", "val"):
        for cls in ("Fake", "Real"):
            os.makedirs(os.path.join(args.out, split, cls), exist_ok=True)

    for shard, cls, split, quota in PLAN:
        try:
            sample_shard(shard, cls, split, quota, args.out)
        except Exception as e:  # network errors: re-run the script to retry
            print(f"{cls} {shard:05d}: ERROR -> {str(e)[:100]}")

    if args.own_real:
        add_own_real(args.own_real, args.out)

    remove_corrupt(args.out)

    print("\nFinal counts:")
    for split in ("train", "val"):
        for cls in ("Fake", "Real"):
            n = len(os.listdir(os.path.join(args.out, split, cls)))
            print(f"  {split} {cls}: {n}")


if __name__ == "__main__":
    main()