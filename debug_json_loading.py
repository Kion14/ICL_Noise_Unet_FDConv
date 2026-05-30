import os
import json
from pathlib import Path
from collections import Counter

import numpy as np
import csv
from PIL import Image
import matplotlib.pyplot as plt


BASE_DATA_DIR = Path(os.environ["DATA_DIR"])
SPLIT_JSON = "datasplits_he_lizard_cellbindb_with_GOODGOOD2context_FIXED.json"

TEST_KEY = "all_stains_without_he_without_mif"
TRAIN_KEY = "he_lizard_plus_half_cellbindb_he"

IMAGE_SIZE = 192
OUT_DIR = Path("debug_json_loading")
OUT_DIR.mkdir(exist_ok=True)


def preprocess_grayscale_percentile(img_pil):
    img = np.array(img_pil, dtype=np.float32)

    if img.ndim == 3:
        gray = img.mean(axis=2)
    else:
        gray = img

    p1, p99 = np.percentile(gray, (1, 99))

    if p99 - p1 < 1e-6:
        gray_norm = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)
    else:
        gray_norm = (gray - p1) / (p99 - p1 + 1e-6)

    gray_norm = np.clip(gray_norm, 0, 1)
    gray_rgb = np.stack([gray_norm, gray_norm, gray_norm], axis=-1)

    return gray_rgb.astype(np.float32), gray, gray_norm


def load_and_debug_item(item, split_name, idx):
    img_path = BASE_DATA_DIR / item["image"]
    mask_path = BASE_DATA_DIR / item["mask"]
    stain = item["stain"]
    sample_id = item.get("sample_id", img_path.stem)

    img_pil_raw = Image.open(img_path)
    img_raw_arr = np.array(img_pil_raw)

    img_pil = img_pil_raw.convert("RGB")
    mask_pil = Image.open(mask_path).convert("L")

    img_pil = img_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    mask_pil = mask_pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

    img_proc, gray_raw, gray_norm = preprocess_grayscale_percentile(img_pil)

    mask_raw = np.array(mask_pil, dtype=np.float32)
    mask = (mask_raw > 0).astype(np.float32)

    row = {
        "split": split_name,
        "idx": idx,
        "stain": stain,
        "sample_id": sample_id,
        "image_path": str(img_path),
        "mask_path": str(mask_path),

        "raw_shape": str(img_raw_arr.shape),
        "raw_dtype": str(img_raw_arr.dtype),
        "raw_min": float(np.min(img_raw_arr)),
        "raw_max": float(np.max(img_raw_arr)),
        "raw_mean": float(np.mean(img_raw_arr)),
        "raw_std": float(np.std(img_raw_arr)),

        "gray_min": float(gray_raw.min()),
        "gray_max": float(gray_raw.max()),
        "gray_mean": float(gray_raw.mean()),
        "gray_std": float(gray_raw.std()),

        "proc_min": float(img_proc.min()),
        "proc_max": float(img_proc.max()),
        "proc_mean": float(img_proc.mean()),
        "proc_std": float(img_proc.std()),

        "mask_mean": float(mask.mean()),
        "mask_sum": float(mask.sum()),

        "is_low_contrast": bool(img_proc.std() < 0.01),
        "is_flat": bool((img_proc.max() - img_proc.min()) < 0.05),
    }

    # Sla verdachte beelden en eerste paar per stain op
    save_debug = row["is_low_contrast"] or row["is_flat"] or idx < 10

    if save_debug:
        safe_stain = stain.replace("/", "_").replace("×", "x")
        safe_sample = sample_id.replace("/", "_")

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(img_pil)
        axes[0].set_title("RGB loaded")

        axes[1].imshow(gray_raw, cmap="gray")
        axes[1].set_title("Gray raw")

        axes[2].imshow(gray_norm, cmap="gray")
        axes[2].set_title("Gray normalized")

        axes[3].imshow(mask, cmap="gray")
        axes[3].set_title("Mask")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            OUT_DIR / f"{split_name}_{idx}_{safe_stain}_{safe_sample}.png",
            dpi=150,
            bbox_inches="tight"
        )
        plt.close(fig)

    return row


with open(SPLIT_JSON, "r") as f:
    splits = json.load(f)

items = []

# Alleen testset debuggen
test_items = splits["test"][TEST_KEY]

print("BASE_DATA_DIR:", BASE_DATA_DIR)
print("Test samples:", len(test_items))
print("Stain counts:", Counter([x["stain"] for x in test_items]))

for idx, item in enumerate(test_items):
    row = load_and_debug_item(item, "test", idx)
    items.append(row)

csv_path = OUT_DIR / "debug_loading_stats.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=items[0].keys())
    writer.writeheader()
    writer.writerows(items)

