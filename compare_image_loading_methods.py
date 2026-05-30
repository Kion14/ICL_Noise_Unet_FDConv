import os, json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

BASE_DATA_DIR = Path(os.environ.get("DATA_DIR", "/home/kkramer/data"))
SPLIT_JSON = "datasplits_he_lizard_cellbindb_with_GOODGOOD2context_FIXED.json"
TEST_KEY = "all_stains_without_he_without_mif"
OUT_DIR = Path("debug_loader_compare")
OUT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 192

def norm_minmax(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)

def norm_percentile(x):
    x = x.astype(np.float32)
    p1, p99 = np.percentile(x, (1, 99))
    return np.clip((x - p1) / (p99 - p1 + 1e-6), 0, 1)

def to_gray(x):
    if x.ndim == 3:
        return x.mean(axis=2)
    return x

with open(SPLIT_JSON) as f:
    splits = json.load(f)

items = splits["test"][TEST_KEY]

# Pak bijvoorbeeld eerste 30 samples
for idx, item in enumerate(items[:30]):
    img_path = BASE_DATA_DIR / item["image"]
    mask_path = BASE_DATA_DIR / item["mask"]
    stain = item["stain"]
    sample_id = item.get("sample_id", img_path.stem)

    # Methode 1: PIL raw, geen RGB convert
    pil_raw = Image.open(img_path)
    pil_raw_resized = pil_raw.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr_pil_raw = np.array(pil_raw_resized)
    gray_pil_raw = to_gray(arr_pil_raw)

    # Methode 2: PIL convert RGB
    pil_rgb = Image.open(img_path).convert("RGB")
    pil_rgb = pil_rgb.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr_pil_rgb = np.array(pil_rgb)
    gray_pil_rgb = to_gray(arr_pil_rgb)

    # Methode 3: OpenCV unchanged
    cv_raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if cv_raw is not None:
        cv_raw = cv2.resize(cv_raw, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        if cv_raw.ndim == 3:
            cv_raw = cv2.cvtColor(cv_raw, cv2.COLOR_BGR2RGB)
        gray_cv_raw = to_gray(cv_raw)
    else:
        gray_cv_raw = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

    # Methode 4: OpenCV grayscale
    cv_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if cv_gray is not None:
        cv_gray = cv2.resize(cv_gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    else:
        cv_gray = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

    # Mask
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
    mask = np.array(mask) > 0

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0,0].imshow(norm_minmax(gray_pil_raw), cmap="gray")
    axes[0,0].set_title("PIL raw minmax")

    axes[0,1].imshow(norm_percentile(gray_pil_raw), cmap="gray")
    axes[0,1].set_title("PIL raw percentile")

    axes[0,2].imshow(norm_minmax(gray_pil_rgb), cmap="gray")
    axes[0,2].set_title("PIL RGB minmax")

    axes[0,3].imshow(norm_percentile(gray_pil_rgb), cmap="gray")
    axes[0,3].set_title("PIL RGB percentile")

    axes[1,0].imshow(norm_minmax(gray_cv_raw), cmap="gray")
    axes[1,0].set_title("CV unchanged minmax")

    axes[1,1].imshow(norm_percentile(gray_cv_raw), cmap="gray")
    axes[1,1].set_title("CV unchanged percentile")

    axes[1,2].imshow(norm_minmax(cv_gray), cmap="gray")
    axes[1,2].set_title("CV grayscale minmax")

    axes[1,3].imshow(mask, cmap="gray")
    axes[1,3].set_title("Mask")

    for ax in axes.ravel():
        ax.axis("off")

    plt.suptitle(f"{idx} | {stain} | {sample_id}")
    plt.tight_layout()
    save_name = f"{idx}_{stain}_{sample_id}".replace("/", "_").replace("×", "x")
    plt.savefig(OUT_DIR / f"{save_name}.png", dpi=150)
    plt.close()

    print(
        f"{idx} {stain} {sample_id} | "
        f"PILraw shape={arr_pil_raw.shape} dtype={arr_pil_raw.dtype} "
        f"min={arr_pil_raw.min()} max={arr_pil_raw.max()} std={arr_pil_raw.std():.2f} | "
        f"CVraw shape={None if cv_raw is None else cv_raw.shape} dtype={None if cv_raw is None else cv_raw.dtype}",
        flush=True
    )

print("Saved comparisons to:", OUT_DIR)