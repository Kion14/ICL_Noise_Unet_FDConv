from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

EXPERIMENT_NAME = "28mei_TrainHEliz_db_TestHEbindb_NMBENCODER_ctx16"

METRICS_PATH = Path("results") / EXPERIMENT_NAME / f"{EXPERIMENT_NAME}_metrics.txt"
OUT_DIR = Path("test_plots") / EXPERIMENT_NAME.replace("/", "_")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pattern = re.compile(
    r"Stain:\s*(\S+)\s*Sample:\s*(\S+)\s*Dice:\s*([0-9.]+),\s*IoU:\s*([0-9.]+)"
)

rows = []

with open(METRICS_PATH, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            stain, sample, dice, iou = match.groups()
            rows.append({
                "stain": stain,
                "sample": sample,
                "dice": float(dice),
                "iou": float(iou)
            })

df = pd.DataFrame(rows)

print(df.head())
print("\nPer stain:")
print(df.groupby("stain")[["dice", "iou"]].mean())

# Dice per sample
plt.figure(figsize=(10, 5))
plt.plot(range(len(df)), df["dice"], marker="o", linestyle="none")
plt.axhline(df["dice"].mean(), linestyle="--", label=f"Mean Dice = {df['dice'].mean():.3f}")
plt.xlabel("Test sample")
plt.ylabel("Dice")
plt.title("Dice distribution over test samples")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_dice_per_sample.png", dpi=300)
plt.close()

# IoU per sample
plt.figure(figsize=(10, 5))
plt.plot(range(len(df)), df["iou"], marker="o", linestyle="none")
plt.axhline(df["iou"].mean(), linestyle="--", label=f"Mean IoU = {df['iou'].mean():.3f}")
plt.xlabel("Test sample")
plt.ylabel("IoU")
plt.title("IoU distribution over test samples")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUT_DIR / "02_iou_per_sample.png", dpi=300)
plt.close()

# Mean Dice per stain
stain_df = df.groupby("stain")[["dice", "iou"]].mean().reset_index()

plt.figure(figsize=(8, 5))
plt.bar(stain_df["stain"], stain_df["dice"])
plt.xlabel("Stain")
plt.ylabel("Mean Dice")
plt.title("Mean Dice per stain")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_mean_dice_per_stain.png", dpi=300)
plt.close()

# Mean IoU per stain
plt.figure(figsize=(8, 5))
plt.bar(stain_df["stain"], stain_df["iou"])
plt.xlabel("Stain")
plt.ylabel("Mean IoU")
plt.title("Mean IoU per stain")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "04_mean_iou_per_stain.png", dpi=300)
plt.close()

# Dropout count per stain
df["dropout"] = df["dice"] < 0.3
dropout_df = df.groupby("stain")["dropout"].sum().reset_index()

plt.figure(figsize=(8, 5))
plt.bar(dropout_df["stain"], dropout_df["dropout"])
plt.xlabel("Stain")
plt.ylabel("Number of dropouts")
plt.title("Dropouts per stain, Dice < 0.3")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "05_dropouts_per_stain.png", dpi=300)
plt.close()

df.to_csv(OUT_DIR / "test_metrics_parsed.csv", index=False)

print(f"Saved test plots to: {OUT_DIR.resolve()}")