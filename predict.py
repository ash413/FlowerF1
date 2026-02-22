"""
Track 1 – predict.py
=====================
Implement load_model() and predict() only.
DO NOT modify anything below the marked line.

Self-evaluate on val set:
    INPUT_CSV  = "val.csv"
    IMAGE_DIR  = "val/images/"

Final submission (paths must be set to test before submitting):
    INPUT_CSV  = "test.csv"
    IMAGE_DIR  = "test/images/"
"""

import os
import pandas as pd
from PIL import Image

# ==============================================================================
# CHANGE THESE PATHS IF NEEDED
# ==============================================================================

INPUT_CSV   = "val.csv"
IMAGE_DIR   = "val/images/"
OUTPUT_PATH = "val_predictions.csv"
MODEL_PATH  = "model/"

# ==============================================================================
# YOUR CODE — IMPLEMENT THESE TWO FUNCTIONS
# ==============================================================================

def load_model():
    """
    Loads a model from MODEL_PATH.

    Supports:
      1) TorchScript model (recommended): model/model.pt or model/model.ts, etc.
      2) PyTorch state_dict checkpoint + timm backbone (reads optional model/meta.json)

    Expected files (any one works):
      - model/model.pt OR model/model.ts OR model/torchscript.pt   (TorchScript)
      - model/weights.pth OR model/weights.pt OR model/checkpoint.pth OR model/model.pth (state_dict)
      - model/meta.json (optional): {"model_name": "...", "img_size": 224}
    """
    import os
    import json
    import torch

    # 1) Try TorchScript first (best: no architecture guess needed)
    ts_candidates = [
        os.path.join(MODEL_PATH, "model.ts"),
        os.path.join(MODEL_PATH, "model.pt"),
        os.path.join(MODEL_PATH, "torchscript.pt"),
        os.path.join(MODEL_PATH, "torchscript.ts"),
    ]
    for p in ts_candidates:
        if os.path.exists(p):
            m = torch.jit.load(p, map_location="cpu")
            m.eval()
            # optionally store img_size if provided
            meta_path = os.path.join(MODEL_PATH, "meta.json")
            if os.path.exists(meta_path):
                try:
                    meta = json.load(open(meta_path, "r"))
                    setattr(m, "_img_size", int(meta.get("img_size", 224)))
                except Exception:
                    setattr(m, "_img_size", 224)
            else:
                setattr(m, "_img_size", 224)
            return m

    # 2) Otherwise load a normal PyTorch checkpoint (state_dict)
    weights_candidates = [
        os.path.join(MODEL_PATH, "weights.pth"),
        os.path.join(MODEL_PATH, "weights.pt"),
        os.path.join(MODEL_PATH, "checkpoint.pth"),
        os.path.join(MODEL_PATH, "model.pth"),
        os.path.join(MODEL_PATH, "best.pth"),
    ]
    weight_path = None
    for p in weights_candidates:
        if os.path.exists(p):
            weight_path = p
            break
    if weight_path is None:
        raise FileNotFoundError(
            f"No model file found in {MODEL_PATH}. "
            f"Tried TorchScript: {ts_candidates} and weights: {weights_candidates}"
        )

    # Optional meta to know which timm backbone + input size you trained with
    model_name = "convnext_tiny"   # safe default for Category 1
    img_size = 224
    meta_path = os.path.join(MODEL_PATH, "meta.json")
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path, "r"))
            model_name = meta.get("model_name", model_name)
            img_size = int(meta.get("img_size", img_size))
        except Exception:
            pass

    # Create model + load weights
    import timm

    model = timm.create_model(model_name, pretrained=False, num_classes=102)

    ckpt = torch.load(weight_path, map_location="cpu")

    # Accept multiple checkpoint formats
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        else:
            # might already be a state_dict-like mapping
            sd = ckpt
    else:
        sd = ckpt

    # Strip 'module.' prefix if saved from DataParallel
    cleaned = {}
    for k, v in sd.items():
        nk = k.replace("module.", "") if isinstance(k, str) else k
        cleaned[nk] = v

    # strict=False helps if your checkpoint includes extra keys (EMA, etc.)
    model.load_state_dict(cleaned, strict=False)
    model.eval()

    setattr(model, "_img_size", img_size)
    return model


def predict(model, images: list) -> list[int]:
    """
    Batched inference. Returns labels in [1..102].
    """
    import torch
    from PIL import Image
    import torchvision.transforms as T

    # Device selection (organizers likely CPU; this also supports CUDA/MPS if present)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Some TorchScript models may not support .to() in rare cases; try safely
    try:
        model = model.to(device)
    except Exception:
        pass

    img_size = int(getattr(model, "_img_size", 224))

    # Standard ImageNet normalization (good for pretrained Category 1 backbones)
    transform = T.Compose([
        T.Resize(int(img_size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    # Batch inference
    batch_size = 64  # safe on CPU; bump to 128 if you want and it fits RAM
    preds_out: list[int] = []

    model_was_in_eval = True
    try:
        model_was_in_eval = not model.training
    except Exception:
        pass

    try:
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                chunk = images[i:i + batch_size]

                tensors = []
                for im in chunk:
                    if im is None:
                        # missing image fallback (should be rare)
                        im = Image.new("RGB", (img_size, img_size), (0, 0, 0))
                    tensors.append(transform(im))
                x = torch.stack(tensors, dim=0).to(device)

                logits = model(x)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]

                # Argmax gives class index in [0..101] for 102-way head
                idx = torch.argmax(logits, dim=1).tolist()

                # Convert to required label space [1..102]
                labs = [int(j) + 1 for j in idx]

                # Final safety clamp to valid range
                labs = [1 if l < 1 else 102 if l > 102 else l for l in labs]
                preds_out.extend(labs)
    finally:
        # keep model in eval (should already be)
        try:
            if not model_was_in_eval:
                model.train()
        except Exception:
            pass

    return preds_out

# ==============================================================================
# DO NOT MODIFY ANYTHING BELOW THIS LINE
# ==============================================================================

def _load_images(df):
    images, missing = [], []
    for _, row in df.iterrows():
        path = os.path.join(IMAGE_DIR, row["filename"])
        if os.path.exists(path):
            images.append(Image.open(path).convert("RGB"))
        else:
            missing.append(row["filename"])
            images.append(None)
    if missing:
        print(f"WARNING: {len(missing)} image(s) not found. First few: {missing[:5]}")
    return images

def main():
    df = pd.read_csv(INPUT_CSV, dtype=str)
    missing_cols = {"image_id", "filename"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input CSV missing columns: {missing_cols}")
    print(f"Loaded {len(df):,} images from {INPUT_CSV}")

    images = _load_images(df)
    model  = load_model()
    preds  = predict(model, images)

    if len(preds) != len(df):
        raise ValueError(f"predict() returned {len(preds)} predictions for {len(df)} images.")

    out = df[["image_id"]].copy()
    out["label"] = [int(p) for p in preds]
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()