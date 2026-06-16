"""
Train SegFormer on the same histopathology JSON split as the ICL script,
but without context images, context masks, NoiseUNet, FDConv, or unused imports.

Expected files/environment:
- DATA_DIR environment variable points to the root folder used by the JSON split.
- SPLIT_JSON exists in the working directory, unless you change the path below.
- dataloaders.py contains preprocess_histology_grayscale.
- DataAugmentation.py contains random_he_augmentation and enhance_bright_nuclei.
- transformers is installed: pip install transformers
"""

import json
import logging
import os
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from DataAugmentation import enhance_bright_nuclei, random_he_augmentation
from dataloaders import preprocess_histology_grayscale


# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_NAME = "16juni_5_segformer_specificECHTsize512"

BASE_DATA_DIR = Path(os.environ["DATA_DIR"])
SPLIT_JSON = "datasplits_he_lizard_cellbindb_with_GOODGOOD2context_FIXED.json"

TRAIN_KEY = "he_lizard_plus_half_cellbindb_he"

# Kies hier je testmodus:
TEST_KEY = "he_only"
# TEST_KEY = "all_stains_without_he"
# TEST_KEY = "all_stains_without_he_without_mif"
# TEST_KEY = "mif_only"

IMAGE_SIZE = 512
BATCH_SIZE = 4
NUM_WORKERS = 8
MAX_EPOCHS = 150
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-7
THRESHOLD = 0.45
VAL_RATIO = 0.2
SEED = 42


# =============================================================================
# Reproducibility
# =============================================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


# =============================================================================
# Loss
# =============================================================================

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
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


# =============================================================================
# SegFormer model
# =============================================================================

class SegFormerBinarySegmentation(nn.Module):
    """Small SegFormer configured for binary segmentation on 3-channel input."""

    def __init__(self, num_channels: int = 3):
        super().__init__()

        config = SegformerConfig(
            num_channels=num_channels,
            num_labels=1,
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            hidden_sizes=[32, 64, 160, 256],
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            num_attention_heads=[1, 2, 5, 8],
            decoder_hidden_size=256,
        )

        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits

        # Hugging Face SegFormer returns lower-resolution logits.
        # Resize back to mask resolution.
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits


# =============================================================================
# Data loading
# =============================================================================

def load_sample_from_json_item(item, image_size=192):
    img_path = BASE_DATA_DIR / item["image"]
    mask_path = BASE_DATA_DIR / item["mask"]
    stain = item["stain"]
    sample_id = item.get("sample_id", Path(img_path).stem)

    img_raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise RuntimeError(f"Could not load image: {img_path}")

    img_raw = cv2.resize(
        img_raw,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )

    mask_pil = Image.open(mask_path).convert("L")
    mask_pil = mask_pil.resize((image_size, image_size), Image.NEAREST)

    # This follows your current ICL preprocessing and returns [H, W, 3].
    img = preprocess_histology_grayscale(img_raw, stain)

    mask_raw = np.array(mask_pil, dtype=np.float32)
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
    val_data = train_data[:val_len]
    train_data = train_data[val_len:]

    print("\n========== JSON SPLIT LOADED ==========")
    print("Train:", len(train_data), Counter([x[2] for x in train_data]))
    print("Val:", len(val_data), Counter([x[2] for x in val_data]))
    print("Test:", len(test_data), Counter([x[2] for x in test_data]))

    return train_data, val_data, test_data


class HistologyTrainDataset(Dataset):
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

        img_tensor = torch.tensor(
            np.ascontiguousarray(img),
            dtype=torch.float32,
        ).permute(2, 0, 1)  # [3, H, W]

        mask_tensor = torch.tensor(
            np.ascontiguousarray(mask),
            dtype=torch.float32,
        ).unsqueeze(0)  # [1, H, W]

        return img_tensor, mask_tensor


class HistologyEvalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, mask, stain, sample_id = self.data[idx]

        img_tensor = torch.tensor(
            np.ascontiguousarray(img),
            dtype=torch.float32,
        ).permute(2, 0, 1)  # [3, H, W]

        mask_tensor = torch.tensor(
            np.ascontiguousarray(mask),
            dtype=torch.float32,
        ).unsqueeze(0)  # [1, H, W]

        return img_tensor, mask_tensor, stain, sample_id


class HistologyDataModule(LightningDataModule):
    def __init__(self, train_data, val_data, test_data, batch_size=4, num_workers=8):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = HistologyTrainDataset(self.train_data, augment=True)
        self.val_dataset = HistologyEvalDataset(self.val_data)
        self.test_dataset = HistologyEvalDataset(self.test_data)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


# =============================================================================
# Lightning module
# =============================================================================

class LightningSegFormer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.save_dir = f"results/{EXPERIMENT_NAME}"
        os.makedirs(self.save_dir, exist_ok=True)

        self.net = SegFormerBinarySegmentation(num_channels=3)
        self.dice_loss = SoftDiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.test_dices = []
        self.test_ious = []

    def forward(self, images):
        return self.net(images)

    def _loss(self, logits, masks):
        return self.dice_loss(logits, masks) + self.bce_loss(logits, masks)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self._loss(logits, masks)

        metrics = self._calculate_metrics(logits, masks)
        self.log_dict({f"train_{k}": v for k, v in metrics.items()}, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks, stains, sample_ids = batch
        logits = self(images)
        loss = self._loss(logits, masks)

        metrics = self._calculate_metrics(logits, masks)
        self.log_dict({f"val_{k}": v for k, v in metrics.items()}, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, masks, stains, sample_ids = batch

        logits = self(images)
        loss = self._loss(logits, masks)
        metrics = self._calculate_metrics(logits, masks)

        self.test_dices.append(metrics["dice"])
        self.test_ious.append(metrics["iou"])

        print(
            f"Batch {batch_idx} | "
            f"Stain: {stains[0]} | "
            f"Sample: {sample_ids[0]} | "
            f"Dice: {metrics['dice']:.4f} | "
            f"IoU: {metrics['iou']:.4f}",
            flush=True,
        )

        self._save_test_metrics_line(batch_idx, stains[0], sample_ids[0], metrics)
        self._save_test_visual(batch_idx, images, masks, logits, stains[0], sample_ids[0], metrics)
        self._save_dropout_debug(batch_idx, masks, logits, stains[0], sample_ids[0], metrics)

        self.log_dict({f"test_{k}": v for k, v in metrics.items()})
        self.log("test_loss", loss)
        return loss

    def on_test_epoch_end(self):
        dices = np.array(self.test_dices)
        ious = np.array(self.test_ious)

        summary_path = os.path.join(
            self.save_dir,
            f"{EXPERIMENT_NAME}_summary_metrics.txt",
        )

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
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )

    def _calculate_metrics(self, pred_logits, target):
        pred_logits = pred_logits.squeeze(1)
        target = target.squeeze(1)

        probs = torch.sigmoid(pred_logits)
        preds = probs > THRESHOLD
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

    def _save_test_metrics_line(self, batch_idx, stain, sample_id, metrics):
        metrics_path = os.path.join(self.save_dir, f"{EXPERIMENT_NAME}_metrics.txt")
        with open(metrics_path, "a") as f:
            f.write(
                f"Batch {batch_idx} "
                f"Stain: {stain} "
                f"Sample: {sample_id} "
                f"Dice: {metrics['dice']:.4f}, "
                f"IoU: {metrics['iou']:.4f}\n"
            )

    def _save_dropout_debug(self, batch_idx, masks, logits, stain, sample_id, metrics):
        probs = torch.sigmoid(logits)
        preds = (probs > THRESHOLD).float()

        if metrics["dice"] >= 0.3:
            return

        print(
            f"DROPOUT | Batch {batch_idx} | "
            f"Stain: {stain} | "
            f"Sample: {sample_id} | "
            f"Dice: {metrics['dice']:.4f} | "
            f"mask_mean: {masks.mean().item():.4f} | "
            f"pred_mean: {preds.mean().item():.4f} | "
            f"prob_max: {probs.max().item():.4f} | "
            f"prob_mean: {probs.mean().item():.4f}",
            flush=True,
        )

        dropout_dir = os.path.join(self.save_dir, "dropout_debug")
        os.makedirs(dropout_dir, exist_ok=True)

        prob_np = probs[0, 0].detach().cpu().numpy()
        plt.imsave(
            os.path.join(dropout_dir, f"batch{batch_idx}_{sample_id}_prob.png"),
            prob_np,
            cmap="gray",
        )

    def _save_test_visual(self, batch_idx, images, masks, logits, stain, sample_id, metrics):
        img = images[0].detach().cpu()
        img_np = img.permute(1, 2, 0).numpy()

        gt_np = masks[0, 0].detach().cpu().numpy()
        prob_np = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()
        pred_np = (prob_np > THRESHOLD).astype(float)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img_np, cmap="gray")
        axes[0].set_title("Input Image")

        axes[1].imshow(gt_np, cmap="gray")
        axes[1].set_title("GT Mask")

        axes[2].imshow(pred_np, cmap="gray")
        axes[2].set_title("Predicted Mask")

        axes[3].imshow(img_np)
        axes[3].imshow(pred_np, alpha=0.4, cmap="jet")
        axes[3].set_title("Prediction Overlay")

        for ax in axes:
            ax.axis("off")

        save_path = os.path.join(
            self.save_dir,
            f"{EXPERIMENT_NAME}_batch_{batch_idx}_sample_{sample_id}_"
            f"stain_{stain}_dice_{metrics['dice']:.3f}.png",
        )

        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# Plotting
# =============================================================================

def plot_convergence(csv_path, save_dir, model_name):
    df = pd.read_csv(csv_path)
    os.makedirs(save_dir, exist_ok=True)

    def epoch_mean(metric):
        return df[["epoch", metric]].dropna().groupby("epoch").mean().reset_index()

    train_loss = epoch_mean("train_loss")
    val_loss = epoch_mean("val_loss")
    train_dice = epoch_mean("train_dice")
    val_dice = epoch_mean("val_dice")

    best_epoch_loss = val_loss.loc[val_loss["val_loss"].idxmin(), "epoch"]
    best_epoch_dice = val_dice.loc[val_dice["val_dice"].idxmax(), "epoch"]

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss["epoch"], train_loss["train_loss"], label="Train Loss")
    plt.plot(val_loss["epoch"], val_loss["val_loss"], label="Val Loss")
    plt.axvline(best_epoch_loss, linestyle="--", label=f"Best val loss epoch {int(best_epoch_loss)}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss Convergence")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "loss_convergence.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_dice["epoch"], train_dice["train_dice"], label="Train Dice")
    plt.plot(val_dice["epoch"], val_dice["val_dice"], label="Val Dice")
    plt.axvline(best_epoch_dice, linestyle="--", label=f"Best val dice epoch {int(best_epoch_dice)}")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title(f"{model_name} - Dice Convergence")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "dice_convergence.png"), dpi=300, bbox_inches="tight")
    plt.close()

    val_dice["dice_delta"] = val_dice["val_dice"].diff()
    plt.figure(figsize=(8, 5))
    plt.plot(val_dice["epoch"], val_dice["dice_delta"], label="Val Dice improvement")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Δ Dice")
    plt.title(f"{model_name} - Validation Dice Improvement per Epoch")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "dice_delta.png"), dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    seed_everything(SEED)

    train_data, val_data, test_data = load_json_split(
        json_path=SPLIT_JSON,
        train_key=TRAIN_KEY,
        test_key=TEST_KEY,
        image_size=IMAGE_SIZE,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    data_module = HistologyDataModule(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    data_module.setup()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training_segformer.log"),
            logging.StreamHandler(),
        ],
    )

    tb_logger = TensorBoardLogger("segformer_logs", name=EXPERIMENT_NAME)
    csv_logger = CSVLogger("segformer_csv", name=EXPERIMENT_NAME)
    logger = [tb_logger, csv_logger]

    hparams = {
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    }

    model = LightningSegFormer(hparams)

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
    logging.info("Training complete")

    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best checkpoint: {best_model_path}")

    model = LightningSegFormer.load_from_checkpoint(
        best_model_path,
        hparams=hparams,
    )

    logging.info("Starting test phase...")
    test_results = trainer.test(model, data_module.test_dataloader())
    logging.info(f"Test results: {test_results}")
    logging.info(f"Best model saved at: {best_model_path}")

    csv_path = f"segformer_csv/{EXPERIMENT_NAME}/version_0/metrics.csv"
    if os.path.exists(csv_path):
        print("Saved training curves.")
        plot_convergence(
            csv_path=csv_path,
            save_dir=f"results/{EXPERIMENT_NAME}/convergence_plots",
            model_name="SegFormer",
        )
    else:
        print(f"CSV metrics file not found: {csv_path}")
