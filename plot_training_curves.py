from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Pas dit aan naar jouw experimentnaam
EXPERIMENT_NAME = "28mei_TrainHEliz_db_TestHEbindb_NMBENCODER_ctx16"

CSV_PATH = Path("iclnoise_csv") / EXPERIMENT_NAME / "version_0" / "metrics.csv"
OUT_DIR = Path("training_plots") / EXPERIMENT_NAME.replace("/", "_")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

print("Available columns:")
print(df.columns.tolist())

epoch_df = df.groupby("epoch").mean(numeric_only=True).reset_index()


def plot_curve(y_keys, labels, title, ylabel, filename):
    plt.figure(figsize=(8, 5))

    for y_key, label in zip(y_keys, labels):
        if y_key in epoch_df.columns:
            plt.plot(epoch_df["epoch"], epoch_df[y_key], label=label)
        else:
            print(f"Skipping missing column: {y_key}")

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300)
    plt.close()


plot_curve(
    ["train_loss", "val_loss"],
    ["Train loss", "Validation loss"],
    "Training convergence",
    "Loss",
    "01_convergence_loss.png"
)

plot_curve(
    ["train_dice", "val_dice"],
    ["Train Dice", "Validation Dice"],
    "Dice score over training",
    "Dice",
    "02_dice_curve.png"
)

plot_curve(
    ["train_iou", "val_iou"],
    ["Train IoU", "Validation IoU"],
    "IoU over training",
    "IoU",
    "03_iou_curve.png"
)

plot_curve(
    ["train_precision", "val_precision", "train_recall", "val_recall"],
    ["Train precision", "Validation precision", "Train recall", "Validation recall"],
    "Precision and recall over training",
    "Score",
    "04_precision_recall_curve.png"
)

print(f"Saved training plots to: {OUT_DIR.resolve()}")