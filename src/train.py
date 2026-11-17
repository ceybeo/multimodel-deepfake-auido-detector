"""Fine-tune EfficientNet-B4 to separate real photos from AI-generated images.

Expects the folder layout produced by prepare_data.py:
    <data>/train/{Fake,Real}/*.jpg
    <data>/val/{Fake,Real}/*.jpg

Key design choices:
  * Random resize (short side 380-700 px) + random 380x380 crop for training,
    so image resolution cannot be used as a shortcut to tell the classes apart.
  * Random JPEG re-compression, so the model sees compressed images
    (e.g. WhatsApp photos), not only clean ones.
  * Class-weighted loss, because Real has more images than Fake.
  * Validation uses the same preprocessing as evaluation and the app
    (short side 600 -> JPEG quality 90 -> center crop 380).
  * Training state is saved every epoch, so an interrupted run resumes.

Usage:
    python src/train.py --data xiz_veri --out checkpoints
"""
import argparse
import io
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few truncated downloads should not stop training
Image.MAX_IMAGE_PIXELS = None

CROP = 380
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class RandomResize:
    """Resize so the short side is a random value in [CROP, 700]."""

    def __call__(self, img):
        target = random.randint(CROP, 700)
        w, h = img.size
        s = target / min(w, h)
        return img.resize((round(w * s), round(h * s)), Image.Resampling.BICUBIC)


class RandomJPEG:
    """Re-compress the image as JPEG with a random quality."""

    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(50, 95))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img


class FixedPrepare:
    """Deterministic preprocessing for validation: short side 600, JPEG quality 90."""

    def __call__(self, img):
        w, h = img.size
        s = 600 / min(w, h)
        img = img.resize((round(w * s), round(h * s)), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def build_model():
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model


def evaluate(model, loader, device):
    """Return overall, Fake and Real validation accuracy (in %)."""
    model.eval()
    correct, total = [0, 0], [0, 0]
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                pred = model(x).argmax(1)
            for c in (0, 1):
                mask = y == c
                total[c] += mask.sum().item()
                correct[c] += (pred[mask] == c).sum().item()
    overall = 100 * sum(correct) / sum(total)
    return overall, 100 * correct[0] / total[0], 100 * correct[1] / total[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="xiz_veri", help="dataset directory")
    parser.add_argument("--out", default="checkpoints", help="where to save models")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16, help="use 8 if CUDA runs out of memory")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU found, training will be extremely slow.")
    os.makedirs(args.out, exist_ok=True)
    best_path = os.path.join(args.out, "efficientnet_b4_best.pth")
    ckpt_path = os.path.join(args.out, "checkpoint.pt")

    train_tf = transforms.Compose([
        RandomResize(), transforms.RandomCrop(CROP), transforms.RandomHorizontalFlip(),
        RandomJPEG(0.7), transforms.ToTensor(), NORM,
    ])
    val_tf = transforms.Compose([FixedPrepare(), transforms.CenterCrop(CROP),
                                 transforms.ToTensor(), NORM])

    train_data = datasets.ImageFolder(os.path.join(args.data, "train"), transform=train_tf)
    val_data = datasets.ImageFolder(os.path.join(args.data, "val"), transform=val_tf)
    assert train_data.class_to_idx == {"Fake": 0, "Real": 1}, train_data.class_to_idx
    print("Classes:", train_data.class_to_idx)
    print("Train:", len(train_data), "| Val:", len(val_data))

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = build_model().to(device)

    counts = torch.bincount(torch.tensor(train_data.targets)).float()
    class_weights = counts.sum() / (2 * counts)  # rarer class -> larger weight
    print("Class counts:", counts.tolist(), "| weights:", [round(w, 3) for w in class_weights.tolist()])

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_acc, start_epoch = 0.0, 0
    if os.path.exists(ckpt_path):  # resume an interrupted run
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["sched"])
        start_epoch, best_acc = ck["epoch"] + 1, ck["best_acc"]
        print(f"Resuming from epoch {start_epoch}/{args.epochs} (best so far: {best_acc:.2f}%)")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        correct, total = 0, 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            if (i + 1) % 200 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} | step {i + 1}/{len(train_loader)} "
                      f"| loss {loss.item():.4f} | {(time.time() - t0) / 60:.1f} min")

        scheduler.step()
        overall, fake_acc, real_acc = evaluate(model, val_loader, device)
        print(f"\n=== Epoch {epoch + 1} done ({(time.time() - t0) / 60:.1f} min) ===")
        print(f"Train accuracy: {100 * correct / total:.2f}%")
        print(f"VALIDATION accuracy: {overall:.2f}%  (Fake: {fake_acc:.2f}% | Real: {real_acc:.2f}%)\n")

        if overall > best_acc:
            best_acc = overall
            torch.save(model.state_dict(), best_path)
            print("-> best model saved\n")
        torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
                    "sched":