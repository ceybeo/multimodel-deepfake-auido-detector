"""Evaluate a trained detector on independent test folders.

Every image goes through the same preprocessing used in training validation
and in the app (short side 600 -> JPEG quality 90 -> center crop 380), so
differences in file size, format or resolution between folders cannot
act as a shortcut for the model.

Reported per group:
  * accuracy at the 0.5 decision threshold, with a 95% Wilson interval
    (test sets are small, so the interval matters)
  * mean fake probability
  * AUC against the real images (threshold-independent separation)
  * how many images fall in the "real", "uncertain" and "fake" zones
    used by the app

Usage:
    python src/evaluate.py --model checkpoints/efficientnet_b4_best.pth \\
        --real path/to/real_photos \\
        --fake gemini=path/to/gemini_images chatgpt=path/to/chatgpt_images \\
        --split-size 1000
"""
import argparse
import io
import math
import os
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

EXTENSIONS = (".jpg", ".jpeg", ".png", ".jfif")
CROP = 380

to_tensor = transforms.Compose([
    transforms.CenterCrop(CROP),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def prepare(img):
    """Short side 600, JPEG quality 90 (same as training validation and the app)."""
    w, h = img.size
    s = 600 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def load_model(path, device):
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def score_folder(model, folder, device):
    """Return a list of (fake_probability, short_side_px) for every image in a folder."""
    results = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(EXTENSIONS):
            continue
        try:
            img = Image.open(os.path.join(folder, name)).convert("RGB")
        except Exception as e:
            print(f"  skipping unreadable file {name}: {e}")
            continue
        x = to_tensor(prepare(img)).unsqueeze(0).to(device)
        with torch.no_grad():
            p = F.softmax(model(x), dim=1)[0]
        results.append((float(p[0]), min(img.size)))  # class 0 = Fake
    return results


def auc(fake_scores, real_scores):
    """Probability that a random fake scores higher than a random real image."""
    total = 0.0
    for f in fake_scores:
        for r in real_scores:
            total += 1.0 if f > r else (0.5 if f == r else 0.0)
    return total / (len(fake_scores) * len(real_scores))


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n, returned in %."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def parse_groups(items):
    groups = {}
    for item in items:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--fake expects name=folder, got: {item}")
        groups[name] = path
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="path to the .pth weights")
    parser.add_argument("--real", required=True, help="folder with real photos")
    parser.add_argument("--fake", nargs="+", required=True, metavar="NAME=FOLDER",
                        help="one or more groups of AI-generated images")
    parser.add_argument("--split-size", type=int, default=0,
                        help="if set, split each fake group by short side (px) into _large / _small")
    parser.add_argument("--low", type=float, default=0.30, help="below this: real zone")
    parser.add_argument("--high", type=float, default=0.70, help="above this: fake zone")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    print("Scoring real images...", flush=True)
    real = score_folder(model, args.real, device)
    groups = {"real": (real, False)}
    for name, folder in parse_groups(args.fake).items():
        print(f"Scoring {name}...", flush=True)
        scored = score_folder(model, folder, device)
        if args.split_size:
            groups[f"{name}_large"] = ([r for r in scored if r[1] >= args.split_size], True)
            groups[f"{name}_small"] = ([r for r in scored if r[1] < args.split_size], True)
        else:
            groups[name] = (scored, True)

    real_scores = [p for p, _ in real]
    print(f"\n{'group':18s}{'n':>4s}{'median px':>10s}{'acc %':>7s}{'95% CI':>14s}"
          f"{'mean fake':>10s}{'AUC':>7s}{'real/unc/fake zone':>22s}")
    print("-" * 92)
    for name, (items, is_fake) in groups.items():
        if not items:
            print(f"{name:18s}{0:4d}   (no images)")
            continue
        scores = [p for p, _ in items]
        n = len(scores)
        k = sum((p > 0.5) == is_fake for p in scores)
        lo, hi = wilson(k, n)
        auc_text = f"{auc(scores, real_scores):7.3f}" if is_fake else f"{'-':>7s}"
        z_real = sum(p <= args.low for p in scores)
        z_fake = sum(p >= args.high for p in scores)
        zones = f"{z_real}/{n - z_real - z_fake}/{z_fake}"
        median_px = int(statistics.median(px for _, px in items))
        print(f"{name:18s}{n:4d}{median_px:10d}{100 * k / n:7.1f}{lo:7.0f}-{hi:<5.0f}"
              f"{sum(scores) / n:10.3f}{auc_text}{zones:>22s}")


if __name__ == "__main__":
    main()