"""
Track 1 - train.py (Category 1: pretrained allowed)

Trains a timm vision model on train.csv + train/images, evaluates on val.csv + val/images,
and saves:
  - model/weights.pt  (best checkpoint)
  - model/meta.json   (model_name + img_size for predict.py)

Run:
  python train.py

Optional args:
  python train.py --model_name convnext_tiny --img_size 384 --epochs 20
"""

import os
import json
import random
import argparse
from dataclasses import dataclass

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm


# ----------------------------
# Reproducibility
# ----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (may reduce speed slightly; safe for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------
# Label mapping (1..102 -> 0..101)
# ----------------------------
def load_label_list(path: str):
    labels = [line.strip() for line in open(path, "r") if line.strip()]
    labels_int = [int(x) for x in labels]  # expected: 1..102
    label_to_idx = {lab: i for i, lab in enumerate(labels_int)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}
    return labels_int, label_to_idx, idx_to_label


# ----------------------------
# Dataset
# ----------------------------
class FlowerDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, transform, label_to_idx, has_labels: bool = True):
        self.df = pd.read_csv(csv_path, dtype=str)
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = has_labels
        self.label_to_idx = label_to_idx

        if "filename" not in self.df.columns:
            raise ValueError(f"{csv_path} missing 'filename' column")
        if has_labels and "label" not in self.df.columns:
            raise ValueError(f"{csv_path} missing 'label' column")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row["filename"])
        img = Image.open(path).convert("RGB")
        x = self.transform(img)

        if self.has_labels:
            y = int(row["label"])       # 1..102
            y_idx = self.label_to_idx[y]  # 0..101
            return x, y_idx

        return x


# ----------------------------
# Metrics (macro F1 over present classes, like your notebook)
# ----------------------------
def macro_f1_present_only(conf: torch.Tensor):
    eps = 1e-9
    support = conf.sum(1)          # true samples per class
    present = support > 0

    tp = conf.diag()
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp

    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    if present.any():
        return f1[present].mean().item(), int(present.sum().item())
    return 0.0, 0


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, num_classes: int):
    model.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    correct, total = 0, 0
    total_loss, n = 0.0, 0

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

        for t, p in zip(y.detach().cpu(), pred.detach().cpu()):
            conf[t, p] += 1

    acc = correct / max(total, 1)
    f1p, k = macro_f1_present_only(conf)
    return total_loss / max(n, 1), acc, f1p, k


# ----------------------------
# Main training
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="train.csv")
    parser.add_argument("--val_csv", default="val.csv")
    parser.add_argument("--train_img_dir", default=os.path.join("train", "images"))
    parser.add_argument("--val_img_dir", default=os.path.join("val", "images"))
    parser.add_argument("--label_list", default="label_list.txt")

    parser.add_argument("--model_name", default="convnext_tiny")
    parser.add_argument("--img_size", type=int, default=384)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smooth", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_out", default="model")

    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print("DEVICE:", device)

    # Validate paths
    for p in [args.train_csv, args.val_csv, args.label_list, args.train_img_dir, args.val_img_dir]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required path: {p}")

    labels_int, label_to_idx, idx_to_label = load_label_list(args.label_list)
    num_classes = len(labels_int)
    print("NUM_CLASSES:", num_classes)

    # Transforms (matches your notebook)
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.25, 0.1)], p=0.5),
        transforms.RandomRotation(12),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.14)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_ds = FlowerDataset(args.train_csv, args.train_img_dir, train_tfms, label_to_idx, has_labels=True)
    val_ds = FlowerDataset(args.val_csv, args.val_img_dir, val_tfms, label_to_idx, has_labels=True)

    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    print("train batches:", len(train_loader), "val batches:", len(val_loader))

    # Model (Category 1 pretrained allowed)
    model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.model_out, exist_ok=True)
    best_f1p = -1.0
    best_path = os.path.join(args.model_out, "weights.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running += loss.item() * x.size(0)
            seen += x.size(0)

        scheduler.step()

        val_loss, val_acc, val_f1p, k = evaluate(model, val_loader, criterion, device, num_classes)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={running/max(seen,1):.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f} | "
            f"macroF1_present={val_f1p:.3f} (classes={k})"
        )

        if val_f1p > best_f1p:
            best_f1p = val_f1p
            torch.save(model.state_dict(), best_path)

    # Save meta for predict.py (predict.py reads model_name + img_size)
    meta = {
        "model_name": args.model_name,
        "img_size": args.img_size,
        "num_classes": num_classes,
        "idx_to_label": idx_to_label,  # not required by predict.py, but harmless
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smooth": args.label_smooth,
    }
    with open(os.path.join(args.model_out, "meta.json"), "w") as f:
        json.dump(meta, f)

    print("\nBest macroF1_present:", best_f1p)
    print("Saved model to:", os.path.abspath(args.model_out))
    print("Weights:", os.path.abspath(best_path))


if __name__ == "__main__":
    main()