import os
import json
import random
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from PIL import Image
from torch import optim
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from DataAugmentation import random_he_augmentation, enhance_bright_nuclei
from models.UNet import UNet
import cv2
from dataloaders import preprocess_histology_grayscale


# ============================================================
# Experiment settings
# ============================================================
EXPERIMENT_NAME = "1juni_IMPROVE_HEINVERTAUGMENT_TrainHEliz_TestALLSTAINSOOKmIFbin_UNET"

# This should point to the folder that contains both CellBinDB/ and Lizard/
# In your Slurm job: export DATA_DIR=$TMPDIR
BASE_DATA_DIR = Path(os.environ["DATA_DIR"])

SPLIT_JSON = "datasplits_he_lizard_cellbindb_with_GOODGOOD2context_FIXED.json"
TRAIN_KEY = "he_lizard_plus_half_cellbindb_he"

# Choose one:
# TEST_KEY = "he_only"
TEST_KEY = "all_stains_without_he"
# TEST_KEY = "all_stains_without_he_without_mif"
# TEST_KEY = "mif_only"

IMAGE_SIZE = 192
THRESHOLD = 0.45
BATCH_SIZE = 4
NUM_WORKERS = 8
MAX_EPOCHS = 150
SEED = 42


# ============================================================
# Loss
# ============================================================
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


# ============================================================
# Data loading from JSON
# ============================================================

# def preprocess_grayscale_percentile(img_pil):
#     img = np.array(img_pil, dtype=np.float32)

#     # RGB -> grayscale
#     if img.ndim == 3:
#         gray = img.mean(axis=2)
#     else:
#         gray = img

#     # Percentile normalization
#     p1, p99 = np.percentile(gray, (1, 99))
#     gray = (gray - p1) / (p99 - p1 + 1e-6)
#     gray = np.clip(gray, 0, 1)

#     # Terug naar 3 kanalen zodat modelinput [3,H,W] blijft
#     gray_rgb = np.stack([gray, gray, gray], axis=-1)

#     return gray_rgb.astype(np.float32)

def preprocess_grayscale_percentile(img):
    img = img.astype(np.float32)

    if img.ndim == 3:
        gray_raw = img.mean(axis=2)
    else:
        gray_raw = img

    p1, p99 = np.percentile(gray_raw, (1, 99))

    if p99 - p1 < 1e-6:
        gray_norm = (gray_raw - gray_raw.min()) / (gray_raw.max() - gray_raw.min() + 1e-6)
    else:
        gray_norm = (gray_raw - p1) / (p99 - p1 + 1e-6)

    gray_norm = np.clip(gray_norm, 0, 1)

    gray_rgb = np.stack([gray_norm, gray_norm, gray_norm], axis=-1)

    return gray_rgb.astype(np.float32)



def load_sample_from_json_item(item, image_size=192):
    img_path = BASE_DATA_DIR / item["image"]
    mask_path = BASE_DATA_DIR / item["mask"]
    stain = item["stain"]
    sample_id = item.get("sample_id", Path(img_path).stem)

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    # img = Image.open(img_path).convert("RGB")
    # mask = Image.open(mask_path).convert("L")

    # img = img.resize((image_size, image_size), Image.BILINEAR)
    # mask = mask.resize((image_size, image_size), Image.NEAREST)

    # img = np.array(img, dtype=np.float32) / 255.0
    # mask_raw = np.array(mask, dtype=np.float32)


    # img_pil = Image.open(img_path).convert("RGB")
    # img_pil = Image.open(img_path)

    # img_raw = np.array(img_pil).astype(np.float32)

    # img_raw = np.array(img_pil, dtype=np.float32)




    # mask = Image.open(mask_path).convert("L")

    # img_pil = img_pil.resize((image_size, image_size), Image.BILINEAR)
    # mask = mask.resize((image_size, image_size), Image.NEAREST)

    # img = preprocess_grayscale_percentile(img_pil)


    img_raw = cv2.imread(
        str(img_path),
        cv2.IMREAD_UNCHANGED
    )

    if img_raw is None:
        raise RuntimeError(f"Could not load image: {img_path}")

    img_raw = cv2.resize(
        img_raw,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR
    )

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((image_size, image_size), Image.NEAREST)

    # img = preprocess_grayscale_percentile(img_raw)
    img = preprocess_histology_grayscale(img_raw, stain)







    if img.std() < 0.01:
        print(
            f"WARNING LOW CONTRAST | "
            f"stain={stain} | sample={sample_id}",
            flush=True
        )

    if img.max() - img.min() < 0.05:
        print(
            f"WARNING FLAT IMAGE | "
            f"stain={stain} | sample={sample_id}",
            flush=True
        )


    mask_raw = np.array(mask, dtype=np.float32)

    if stain == "mIF":
        mask = (mask_raw < 128).astype(np.float32)
    else:
        mask = (mask_raw > 0).astype(np.float32)

    return img, mask, stain, sample_id


def load_json_split(json_path, train_key, test_key, image_size=192, val_ratio=0.2, seed=42):
    with open(json_path, "r") as f:
        splits = json.load(f)

    train_items = splits["train"][train_key]
    test_items = splits["test"][test_key]

    train_data = [
        load_sample_from_json_item(item, image_size=image_size)
        for item in train_items
    ]

    test_data = [
        load_sample_from_json_item(item, image_size=image_size)
        for item in test_items
    ]

    random.seed(seed)
    random.shuffle(train_data)

    val_len = int(len(train_data) * val_ratio)
    V = train_data[:val_len]
    X = train_data[val_len:]
    Y = test_data

    print("\n========== JSON SPLIT LOADED ==========")
    print("BASE_DATA_DIR:", BASE_DATA_DIR)
    print("Train:", len(X), Counter([x[2] for x in X]))
    print("Val:", len(V), Counter([x[2] for x in V]))
    print("Test:", len(Y), Counter([x[2] for x in Y]))

    return X, V, Y


# ============================================================
# Dataset classes
# ============================================================
class UNetTrainDataset(Dataset):
    def __init__(self, data, augment=True):
        self.data = data
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, mask, stain, sample_id = self.data[idx]

        if self.augment:
            img = random_he_augmentation(img)
            img = enhance_bright_nuclei(img, p=0.5)

        img = torch.tensor(np.ascontiguousarray(img), dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(np.ascontiguousarray(mask), dtype=torch.float32).unsqueeze(0)

        return img, mask


class UNetEvalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, mask, stain, sample_id = self.data[idx]

        img = torch.tensor(np.ascontiguousarray(img), dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(np.ascontiguousarray(mask), dtype=torch.float32).unsqueeze(0)

        return img, mask, stain, sample_id


class UNetDataModule(LightningDataModule):
    def __init__(self, X_train, X_val, X_test, batch_size=4, num_workers=0):
        super().__init__()
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = UNetTrainDataset(self.X_train, augment=True)
        self.val_dataset = UNetEvalDataset(self.X_val)
        self.test_dataset = UNetEvalDataset(self.X_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )


# ============================================================
# Lightning model
# ============================================================
class LightningUNetBaseline(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.save_dir = f"results/{EXPERIMENT_NAME}"
        os.makedirs(self.save_dir, exist_ok=True)

        self.net = UNet()
        self.dice_loss = SoftDiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.test_dices = []
        self.test_ious = []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.dice_loss(logits, masks) + self.bce_loss(logits, masks)

        metrics = self._calculate_metrics(logits, masks)
        self.log_dict({f"train_{k}": v for k, v in metrics.items()}, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks, stains, sample_ids = batch
        logits = self(images)
        loss = self.dice_loss(logits, masks) + self.bce_loss(logits, masks)

        metrics = self._calculate_metrics(logits, masks)
        self.save_validation_visuals(images, masks, logits, batch_idx)

        self.log_dict({f"val_{k}": v for k, v in metrics.items()}, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, masks, stains, sample_ids = batch
        logits = self(images)
        loss = self.dice_loss(logits, masks) + self.bce_loss(logits, masks)

        metrics = self._calculate_metrics(logits, masks)
        self.test_dices.append(metrics["dice"])
        self.test_ious.append(metrics["iou"])

        print(
            f"Batch {batch_idx} | Stain: {stains[0]} | Sample: {sample_ids[0]} | "
            f"Dice: {metrics['dice']:.4f} | IoU: {metrics['iou']:.4f}",
            flush=True,
        )

        self.log_dict({f"test_{k}": v for k, v in metrics.items()})
        self.log("test_loss", loss)

        self.save_test_visual(images, masks, logits, batch_idx, stains[0], sample_ids[0], metrics)
        self.write_test_metrics(batch_idx, stains[0], sample_ids[0], metrics)

        return loss

    def on_test_epoch_end(self):
        dices = np.array(self.test_dices)
        ious = np.array(self.test_ious)

        summary_path = os.path.join(self.save_dir, f"{EXPERIMENT_NAME}_summary_metrics.txt")

        with open(summary_path, "w") as f:
            f.write(f"Mean Dice: {dices.mean():.4f}\n")
            f.write(f"Median Dice: {np.median(dices):.4f}\n")
            f.write(f"Std Dice: {dices.std():.4f}\n")
            f.write(f"Highest Dice: {dices.max():.4f}\n")
            f.write(f"Lowest Dice: {dices.min():.4f}\n")
            f.write(f"Dropouts Dice < 0.3: {(dices < 0.3).sum()}\n\n")

            f.write(f"Mean IoU: {ious.mean():.4f}\n")
            f.write(f"Median IoU: {np.median(ious):.4f}\n")
            f.write(f"Std IoU: {ious.std():.4f}\n")
            f.write(f"Highest IoU: {ious.max():.4f}\n")
            f.write(f"Lowest IoU: {ious.min():.4f}\n")
            f.write(f"Dropouts IoU < 0.2: {(ious < 0.2).sum()}\n")

        print(f"Saved summary metrics to: {summary_path}", flush=True)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-7)

    def _calculate_metrics(self, pred_logits, target):
        target = target.squeeze(1)
        pred_logits = pred_logits.squeeze(1)

        pred = torch.sigmoid(pred_logits)
        preds = pred > THRESHOLD
        targets = target > 0.5
        smooth = 1e-6

        tp = (preds & targets).float().sum()
        fp = (preds & ~targets).float().sum()
        fn = (~preds & targets).float().sum()
        tn = (~preds & ~targets).float().sum()

        accuracy = (tp + tn) / (tp + fp + fn + tn + smooth)
        precision = tp / (tp + fp + smooth)
        recall = tp / (tp + fn + smooth)
        specificity = tn / (tn + fp + smooth)
        iou = tp / (tp + fp + fn + smooth)
        dice = (2.0 * tp + smooth) / (preds.float().sum() + targets.float().sum() + smooth)

        return {
            "accuracy": accuracy.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "specificity": specificity.item(),
            "iou": iou.item(),
            "dice": dice.item(),
        }

    def save_validation_visuals(self, images, masks, logits, batch_idx, max_batches=3):
        if batch_idx >= max_batches:
            return
        if self.current_epoch % 10 != 0:
            return

        vis_dir = os.path.join(self.save_dir, "validation_visuals")
        os.makedirs(vis_dir, exist_ok=True)

        self._save_visual_grid(
            images,
            masks,
            logits,
            os.path.join(vis_dir, f"epoch_{self.current_epoch}_batch_{batch_idx}.png"),
            title_prefix="Validation",
        )

    def save_test_visual(self, images, masks, logits, batch_idx, stain, sample_id, metrics):
        save_path = os.path.join(
            self.save_dir,
            f"{EXPERIMENT_NAME}_batch_{batch_idx}_sample_{sample_id}_stain_{stain}_dice_{metrics['dice']:.3f}.png",
        )
        self._save_visual_grid(images, masks, logits, save_path, title_prefix="Test")

    def _save_visual_grid(self, images, masks, logits, save_path, title_prefix=""):
        img = images[0].detach().cpu()
        gt_mask = masks[0, 0].detach().cpu()
        pred_mask = torch.sigmoid(logits[0, 0]).detach().cpu()
        pred_binary = (pred_mask > THRESHOLD).float()

        img_np = img.permute(1, 2, 0).numpy()
        gt_np = gt_mask.numpy()
        pred_np = pred_binary.numpy()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img_np)
        axes[0].set_title("Input")
        axes[1].imshow(gt_np, cmap="gray")
        axes[1].set_title("Ground truth")
        axes[2].imshow(pred_np, cmap="gray")
        axes[2].set_title("Prediction")
        axes[3].imshow(img_np)
        axes[3].imshow(pred_np, alpha=0.4, cmap="jet")
        axes[3].set_title("Overlay")

        for ax in axes:
            ax.axis("off")

        if title_prefix:
            plt.suptitle(title_prefix)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def write_test_metrics(self, batch_idx, stain, sample_id, metrics):
        metrics_path = os.path.join(self.save_dir, f"{EXPERIMENT_NAME}_metrics.txt")
        with open(metrics_path, "a") as f:
            f.write(
                f"Batch {batch_idx} "
                f"Stain: {stain} "
                f"Sample: {sample_id} "
                f"Dice: {metrics['dice']:.4f}, "
                f"IoU: {metrics['iou']:.4f}\n"
            )


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    pl.seed_everything(SEED, workers=True)

    X, V, Y = load_json_split(
        json_path=SPLIT_JSON,
        train_key=TRAIN_KEY,
        test_key=TEST_KEY,
        image_size=IMAGE_SIZE,
        val_ratio=0.2,
        seed=SEED,
    )

    data_module = UNetDataModule(
        X_train=X,
        X_val=V,
        X_test=Y,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    data_module.setup()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training_unet_baseline.log"),
            logging.StreamHandler(),
        ],
    )

    tb_logger = TensorBoardLogger("unet_logs", name=EXPERIMENT_NAME)
    csv_logger = CSVLogger("unet_csv", name=EXPERIMENT_NAME)
    logger = [tb_logger, csv_logger]

    hparams = {"learning_rate": 1e-4}
    model = LightningUNetBaseline(hparams)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=f"{EXPERIMENT_NAME}-{{epoch}}-{{val_loss:.4f}}",
        every_n_epochs=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=8,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
    )

    logging.info(f"Training samples: {len(data_module.train_dataset)}")
    logging.info(f"Validation samples: {len(data_module.val_dataset)}")
    logging.info(f"Test samples: {len(data_module.test_dataset)}")

    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

    best_model_path = checkpoint_callback.best_model_path
    logging.info(f"Best model saved at: {best_model_path}")
    print(f"Loading best checkpoint: {best_model_path}")

    model = LightningUNetBaseline.load_from_checkpoint(best_model_path, hparams=hparams)

    test_results = trainer.test(model, data_module.test_dataloader())
    logging.info(f"Test results: {test_results}")
