from models.NoiseContext import ContextNoiseUNet

import torch
from torch import nn, optim
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from DataAugmentation import augment_data
# from DataAugmentation import random_intensity_augmentation, random_invert_intensity
from DataAugmentation import random_he_augmentation, enhance_bright_nuclei
# from DataAugmentation import random_color_augmentation, random_grayscale
import cv2
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning import LightningDataModule
import logging
import random
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim
# from models.multiverseg.models.network import MultiverSegNet
# from models.nn_unet import NnUNet
# from models.unetr import UNETR2D
# from models.swin_unet import SwinUNet
# from models.pairwise_conv_avg_model import PairwiseConvAvgModel
# from models.NoiseContext import ContextNoiseUNet
import os
import matplotlib.pyplot as plt
from util.shapecheck import ShapeChecker
import pydicom
import pywt
from functools import lru_cache
import torchvision.transforms as transforms
import nibabel as nib
from scipy.ndimage import distance_transform_edt as distance_transform
from glob import glob
import csv
import pandas as pd
from dataloaders import reading_training_data_fetal, reading_camus_data, reading_data, reading_data_tg3k, get_data_jnu,get_frame_labels, read_and_split_busi_data, read_and_split_busbra_data, read_data_jnu,split_training_data, read_histopathology_data
from dataloaders import split_single_stain
# from dataloaders import split_leave_one_stain_out
from dataloaders import split_leave_stains_out
from dataloaders import read_image_mask_folder_dataset, read_bbbc038_dataset
from DataAugmentation import random_intensity_augmentation, random_invert_intensity
import random
import json
from PIL import Image
from pathlib import Path
from collections import Counter
import cv2
from dataloaders import preprocess_histology_grayscale
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict



EXPERIMENT_NAME = "4juni_ICL_NMB_ctx4_general_VISUALISATIONSOVERLAY2"
BASE_DATA_DIR = Path(os.environ["DATA_DIR"])

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Operate directly on logits (no sigmoid!)
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice


# =============================================================================
# Define model
# =============================================================================

class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.best_dice = 0.0
        # self.save_dir="exp_1"
        self.save_dir = f"results/{EXPERIMENT_NAME}"
        os.makedirs(self.save_dir, exist_ok=True)
        self.net=ContextNoiseUNet()
        #self.net = MultiverSegNet(encoder_blocks=[64, 64, 64, 64])
        #self.net=ContextNoiseWNet()
        #self.net=ContextNoiseUNet()
        #self.net=UNETR2D(img_shape=(192, 192), input_dim=1, output_dim=1)
        #self.net=NnUNet(in_channels=1, out_channels=1, base_num_features=16, num_pool=4, ndim=2, deep_supervision=False)
        #self.net=SwinUNet()
        self.test_dices = []
        self.test_ious = []
        self.visual_count_per_stain = defaultdict(int)
        self.max_visuals_per_stain = 5

       
        
        # Loss function
        self.dice_loss = SoftDiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()


    def save_error_overlay_visual(self, target_images, target_masks, pred_masks,
                                  batch_idx, stain, sample_id, metrics):
        if self.visual_count_per_stain[stain] >= self.max_visuals_per_stain:
            return

        vis_dir = os.path.join(self.save_dir, "test_visuals_per_stain", stain)
        os.makedirs(vis_dir, exist_ok=True)

        img = target_images[0].detach().cpu()
        gt = target_masks[0, 0].detach().cpu().numpy() > 0.5

        prob = torch.sigmoid(pred_masks[0, 0]).detach().cpu().numpy()
        pred = prob > 0.45

        img_np = img.permute(1, 2, 0).numpy()
        pred_np = pred.astype(np.float32)
        gt_np = gt.astype(np.float32)

        # Error overlay op basis van predicted mask
        overlay = np.zeros((*gt.shape, 3), dtype=np.float32)

        tp = pred & gt          # goed voorspeld
        fp = pred & ~gt         # fout voorspeld, hoort niet
        fn = ~pred & gt         # gemist, had voorspeld moeten worden

        overlay[tp] = [0.0, 1.0, 0.0]       # groen
        overlay[fp] = [1.0, 0.0, 0.0]       # rood
        overlay[fn] = [0.65, 0.0, 0.0]      # donkerrood

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))

        axes[0].imshow(img_np, cmap="gray")
        axes[0].set_title("Input image")

        axes[1].imshow(gt_np, cmap="gray")
        axes[1].set_title("Ground truth mask")

        axes[2].imshow(pred_np, cmap="gray")
        axes[2].imshow(overlay, alpha=0.75)
        axes[2].set_title("Error overlay\nGreen=TP, Red=FP, Dark red=FN")

        axes[3].imshow(pred_np, cmap="gray")
        axes[3].set_title("Predicted mask")

        for ax in axes:
            ax.axis("off")

        save_path = os.path.join(
            vis_dir,
            f"{stain}_vis{self.visual_count_per_stain[stain]+1}_"
            f"batch{batch_idx}_sample_{sample_id}_"
            f"dice_{metrics['dice']:.3f}_iou_{metrics['iou']:.3f}.png"
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        self.visual_count_per_stain[stain] += 1





    def forward(self, target_in, context_in, context_out=None):
        sc = ShapeChecker()
        y_pred = self.net(target_in,context_in,context_out)
        sc.check(y_pred, "B C H W")
        return y_pred
    
    def on_test_epoch_end(self):



        dices = np.array(self.test_dices)
        ious = np.array(self.test_ious)

        # dices_all = np.array(self.test_dices) ##################################################################### GEMIDDELDE FILTER
        # ious_all = np.array(self.test_ious)

        # keep = (dices_all >= 0.4) & (ious_all >= 0.3)

        # dices = dices_all[keep]
        # ious = ious_all[keep]


        summary_path = os.path.join(
            self.save_dir,
            f"{EXPERIMENT_NAME}_summary_metrics.txt"
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

    def training_step(self, batch, batch_idx):
        target_images, target_masks, context_images, context_masks = batch
        target_images=target_images.to("cuda")
        target_masks=target_masks.to("cuda")
        context_images=context_images.to("cuda")
        context_masks=context_masks.to("cuda")

        ################################################################################################ NIUEW
        # pred_masks = self(target_images.squeeze(1), context_images.squeeze(1),context_masks.squeeze(1))        
        # loss = self.criterion(pred_masks,target_masks.squeeze(1))

        pred_masks = self(target_images, context_images, context_masks)
        # loss = self.criterion(pred_masks, target_masks)
        loss = self.dice_loss(pred_masks, target_masks) + self.bce_loss(pred_masks, target_masks)

        #############################################################################################

        metrics = self._calculate_metrics(pred_masks, target_masks)
        self.log_dict({f"train_{k}": v for k, v in metrics.items()}, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        target_images, target_masks, context_images,context_mask, stains, sample_ids = batch
        # Forward pass

        ################################################################################################# NIUEW
        # pred_masks = self(target_images.squeeze(1), context_images.squeeze(1),context_mask.squeeze(1))        

        # loss = self.criterion(pred_masks, target_masks.squeeze(1))

        pred_masks = self(target_images, context_images, context_mask)
        # loss = self.criterion(pred_masks, target_masks)
        loss = self.dice_loss(pred_masks, target_masks) + self.bce_loss(pred_masks, target_masks)


        ##################################################################################################
        # Calculate metrics
        metrics = self._calculate_metrics(pred_masks, target_masks)

        self.save_validation_visuals(
            target_images,
            target_masks,
            pred_masks,
            batch_idx
        )

        # Log everything
        self.log_dict({f"val_{k}": v for k, v in metrics.items()}, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)


        return loss


    def test_step(self, batch, batch_idx):
        # target_images, target_masks, context_images,context_mask, stains = batch
        target_images, target_masks, context_images, context_mask, stains, sample_ids = batch

        print(
            f"Batch {batch_idx} | "
            f"Stain: {stains[0]} | "
            f"Sample: {sample_ids[0]}"
        )

        ############################################################################### NIUEW
        # Forward pass
        # pred_masks = self(target_images.squeeze(1), context_images.squeeze(1),context_mask.squeeze(1))        

        # # Calculate loss
        # loss = self.criterion(pred_masks, target_masks.squeeze(1)) 

        pred_masks = self(
            target_images,
            context_images,
            context_mask
        )

        # loss = self.criterion(pred_masks, target_masks)
        loss = self.dice_loss(pred_masks, target_masks) + self.bce_loss(pred_masks, target_masks)

        ####################################################################################

        # Calculate metrics
        metrics = self._calculate_metrics(pred_masks, target_masks)

        self.test_dices.append(metrics["dice"])
        self.test_ious.append(metrics["iou"])




        probs = torch.sigmoid(pred_masks)
        preds = (probs > 0.45).float()

        dice = metrics["dice"]

        if dice < 0.3:

            print(
                f"DROPOUT | Batch {batch_idx} | "
                f"Stain: {stains[0]} | "
                f"Sample: {sample_ids[0]} | "
                f"Dice: {dice:.4f} | "
                f"mask_mean: {target_masks.mean().item():.4f} | "
                f"pred_mean: {preds.mean().item():.4f} | "
                f"prob_max: {probs.max().item():.4f} | "
                f"prob_mean: {probs.mean().item():.4f}",
                flush=True
            )

            dropout_dir = os.path.join(self.save_dir, "dropout_debug")
            os.makedirs(dropout_dir, exist_ok=True)

            prob_np = probs[0, 0].detach().cpu().numpy()

            plt.imsave(
                os.path.join(
                    dropout_dir,
                    f"batch{batch_idx}_{sample_ids[0]}_prob.png"
                ),
                prob_np,
                cmap="gray"
            )






        # Log everything
        self.log_dict({f"test_{k}": v for k, v in metrics.items()})
        self.log("test_loss", loss)

        # img = target_images[0,0].detach().cpu()     # shape [C,H,W] or [H,W]
        img = target_images[0].detach().cpu()
        img_np = img.permute(1, 2, 0).numpy()


        gt_mask = target_masks[0,0].detach().cpu()
        # pred_mask = pred_masks[0,0].detach().cpu()  # convert logits to [0,1]
        pred_mask = torch.sigmoid(pred_masks[0, 0]).detach().cpu()

        # convert to numpy
        img_np = img.permute(1,2,0).numpy() if img.ndim==3 else img.numpy()
        gt_np = gt_mask.squeeze().numpy()
        # pred_np = (pred_mask > 0.5).squeeze().numpy().astype(float)  # binarize
        pred_np = (pred_mask > 0.45).squeeze().numpy().astype(float)
        # Rotate all images 90° clockwise
       
        # metrics_path = os.path.join(self.save_dir, "test_metrics.txt")
        metrics_path = os.path.join(
            self.save_dir,
            f"{EXPERIMENT_NAME}_metrics.txt"
        )



# Append Dice and IoU for each sample
        with open(metrics_path, "a") as f:
            f.write(
                f"Batch {batch_idx} "
                f"Stain: {stains[0]} "
                f"Sample: {sample_ids[0]} "
                f"Dice: {metrics['dice']:.4f}, "
                f"IoU: {metrics['iou']:.4f}\n"
            )
        # plot side by side
        fig, axes = plt.subplots(1,4, figsize=(16,4))
        axes[0].imshow(img_np, cmap="gray")
        axes[0].set_title("Input Image")
        axes[1].imshow(gt_np, cmap="gray")
        axes[1].set_title("GT Mask")
        axes[2].imshow(pred_np, cmap="gray")
        axes[2].set_title("Predicted Mask")
        axes[3].imshow(img_np)
        axes[3].imshow(pred_np, alpha=0.4, cmap="jet")
        axes[3].set_title("Prediction Overlay")
        for ax in axes: ax.axis("off")

        self.save_error_overlay_visual(
            target_images=target_images,
            target_masks=target_masks,
            pred_masks=pred_masks,
            batch_idx=batch_idx,
            stain=stains[0],
            sample_id=sample_ids[0],
            metrics=metrics
        )

        return loss


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 
                              lr=1e-4,
                              weight_decay=1e-7)
        return optimizer

    def _calculate_metrics(self, pred_logits, target, save_dir="outputs", batch_idx=0):
        target = target.squeeze(1)
        pred_logits = pred_logits.squeeze(1)
        
        pred = torch.sigmoid(pred_logits)
        
        preds = pred > 0.45  ##################################################################################### threshhold
        targets = target > 0.5
        smooth = 1e-6  # Smoothing factor to avoid division by zero
        
        # Calculate TP, FP, FN, TN
        tp = (preds & targets).float().sum()
        fp = (preds & ~targets).float().sum()
        fn = (~preds & targets).float().sum()
        tn = (~preds & ~targets).float().sum()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + fp + fn + tn + smooth)
        precision = tp / (tp + fp + smooth)
        recall = tp / (tp + fn + smooth)
        specificity = tn / (tn + fp + smooth)
        iou = tp / (tp + fp + fn + smooth)
        dice = (2. * tp + smooth) / (preds.float().sum() + targets.float().sum() + smooth)
 
        # Visualization
        os.makedirs(save_dir, exist_ok=True)
        pred_np = preds[0].cpu().detach().numpy()
        target_np = targets[0].cpu().detach().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(pred_np, cmap='gray')
        ax1.set_title('Prediction')
        ax1.axis('off')
        
        ax2.imshow(target_np, cmap='gray')
        ax2.set_title('Ground Truth')
        ax2.axis('off')
        
        metrics = {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'specificity': specificity.item(),
            'iou': iou.item(),
            'dice': dice.item(),
        }

        plt.suptitle(f'Batch {batch_idx} - Dice: {metrics["dice"]:.3f}')
        plt.tight_layout()
        plt.savefig("displayed_image.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return metrics
    

    def save_validation_visuals(self, target_images, target_masks, pred_masks, batch_idx, max_batches=3):
        if batch_idx >= max_batches:
            return

        if self.current_epoch % 10 != 0:
            return

        vis_dir = os.path.join(self.save_dir, "validation_visuals")
        os.makedirs(vis_dir, exist_ok=True)

        img = target_images[0].detach().cpu()
        gt_mask = target_masks[0, 0].detach().cpu()
        pred_mask = torch.sigmoid(pred_masks[0, 0]).detach().cpu()
        pred_binary = (pred_mask > 0.45).float()

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

        save_path = os.path.join(
            vis_dir,
            f"epoch_{self.current_epoch}_batch_{batch_idx}.png"
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close(fig)


# =============================================================================
# Data Preparation
# =============================================================================

# X, Y, V = reading_camus_data()


###############
# data = read_histopathology_data(os.environ["DATA_DIR"], image_size=192) #192
# X, V, Y = split_training_data(data)

# X_init = X.copy()
##############


# X = augment_data(X, context=False, target_size=(192, 192))


#################### LEAVE ONE OUT

# data = read_histopathology_data(os.environ["DATA_DIR"], image_size=192)

# X, V, Y = split_leave_one_stain_out(
#     data,
#     test_stain="DAPI"
# )

# X_init = X.copy()


################################################################################### UIT
# data = read_histopathology_data(os.environ["DATA_DIR"], image_size=192)

# # heldout_stain = "DAPI"  #  /DAPI,  nog doen

# # X, V, Y = split_leave_one_stain_out(
# #     data,
# #     test_stain=heldout_stain
# # )

# X, V, Y = split_leave_stains_out(
#     data,
#     test_stains=["DAPI", "10×Genomics_DAPI"]
# )

# train_context = X.copy()
# test_context = Y.copy()
#####################################################################################



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

    # img = Image.open(img_path).convert("RGB")
    # mask = Image.open(mask_path).convert("L")

    # img = img.resize((image_size, image_size), Image.BILINEAR)
    # mask = mask.resize((image_size, image_size), Image.NEAREST)

    # img = np.array(img, dtype=np.float32) / 255.0
    # mask_raw = np.array(mask, dtype=np.float32)

    # img_pil = Image.open(img_path).convert("RGB")
    # img_pil = Image.open(img_path)

    # img_raw = np.array(img_pil).astype(np.float32)
    # mask = Image.open(mask_path).convert("L")

    # img_pil = img_pil.resize((image_size, image_size), Image.BILINEAR)
    # mask = mask.resize((image_size, image_size), Image.NEAREST)

    # img = preprocess_grayscale_percentile(img_pil)
    # mask_raw = np.array(mask, dtype=np.float32)

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

    mask_raw = np.array(mask, dtype=np.float32)

    # if stain == "mIF":
    #     mask = (mask_raw < 128).astype(np.float32)
    # else:
    #     mask = (mask_raw > 0).astype(np.float32)

    mask = (mask_raw > 0).astype(np.float32)

    return img, mask, stain, sample_id


def load_json_split(json_path, train_key, test_key, image_size=192, val_ratio=0.2, seed=42):
    with open(json_path, "r") as f:
        splits = json.load(f)

    train_items = splits["train"][train_key]
    test_items = splits["test"][test_key]
    test_context_items = splits["test_context"][test_key]

    train_data = [
        load_sample_from_json_item(item, image_size=image_size)
        for item in train_items
    ]

    test_data = [
        load_sample_from_json_item(item, image_size=image_size)
        for item in test_items
    ]

    test_context_data = [
        load_sample_from_json_item(item, image_size=image_size)
        for item in test_context_items
    ]

    random.seed(seed)
    random.shuffle(train_data)

    val_len = int(len(train_data) * val_ratio)

    V = train_data[:val_len]
    X = train_data[val_len:]
    Y = test_data

    print("\n========== JSON SPLIT LOADED ==========")
    print("Train:", len(X), Counter([x[2] for x in X]))
    print("Val:", len(V), Counter([x[2] for x in V]))
    print("Test:", len(Y), Counter([x[2] for x in Y]))

    return X, V, Y, test_context_data




SPLIT_JSON = "datasplits_he_lizard_cellbindb_with_GOODGOOD2context_FIXED.json"

TRAIN_KEY = "he_lizard_plus_half_cellbindb_he"

# Kies hier je testmodus:
# TEST_KEY = "he_only"
TEST_KEY = "all_stains_without_he"
# TEST_KEY = "all_stains_without_he_without_mif"
# TEST_KEY = "mif_only"

X, V, Y, separate_test_context = load_json_split(
    json_path=SPLIT_JSON,
    train_key=TRAIN_KEY,
    test_key=TEST_KEY,
    image_size=192,
    val_ratio=0.2,
    seed=42
)


train_context = X.copy()

# Belangrijk:
# Voor HE-test kun je train_context gebruiken.
# Voor cross-stain test is Y.copy() logisch als je same-stain context wil gebruiken.
test_context = separate_test_context













# cellbindb = read_histopathology_data(os.environ["DATA_DIR"], image_size=192)

# cellbindb_he = [
#     item for item in cellbindb
#     if item[2] in ["HE", "10×Genomics_HE"]
# ]

# cellbindb_dapi_test = [
#     item for item in cellbindb
#     if item[2] in ["DAPI", "10×Genomics_DAPI"]
# ]

# monuseg_he = read_image_mask_folder_dataset(
#     "/gpfs/home3/kkramer/data/MoNuSeg",
#     stain_name="MoNuSeg_HE",
#     image_size=192
# )

# lizard_he = read_image_mask_folder_dataset(
#     "/gpfs/home3/kkramer/data/Lizard",
#     stain_name="Lizard_HE",
#     image_size=192
# )

# # nuinsseg_he = read_image_mask_folder_dataset(
# #     "/gpfs/home3/kkramer/data/NuInsSeg_HE",
# #     stain_name="NuInsSeg_HE",
# #     image_size=192
# # )

# # train_data = cellbindb_he + monuseg_he + lizard_he

# # random.seed(42)
# # random.shuffle(train_data)

# # val_len = int(0.2 * len(train_data))

# # V = train_data[:val_len]
# # X = train_data[val_len:]

# # Y = cellbindb_dapi_test

# # train_context = X.copy()
# # test_context = Y.copy()

# random.seed(42)
# random.shuffle(lizard_he)

# val_len = int(0.2 * len(lizard_he))

# V = lizard_he[:val_len]
# X = lizard_he[val_len:]

# Y = cellbindb_he
# # test_context = X.copy()
# test_context = [] ######################################################################### CONTEXTS
# train_context = X.copy()




# data = read_histopathology_data(os.environ["DATA_DIR"], image_size=192)

# X, V, Y = split_single_stain(
#     data,
#     stain_name="mIF"
# )

# X_init = X.copy()



# =============================================================================
# Define dataset and dataloaders
# =============================================================================

# def percentile_normalize(img, lower=1, upper=99):
#     img = img.astype(np.float32)

#     p_low, p_high = np.percentile(img, (lower, upper))

#     img = (img - p_low) / (p_high - p_low + 1e-6)

#     img = np.clip(img, 0, 1)

#     return img.astype(np.float32)


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

    # 1. Loss convergence
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

    # 2. Dice convergence
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

    # 3. Dice improvement per epoch
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


class TrainDataset(Dataset):
    def __init__(self, data, context_dataset, context_size=8, return_context_only=False, device='cuda'):
        self.data = data
        self.context_size = context_size
        self.return_context_only = return_context_only
        self.device = device
        self.context_dataset = context_dataset
       
        
        assert len(self.data) >= context_size + 1

    def __len__(self):
        return len(self.data) - self.context_size

           

    def __getitem__(self, idx):
        if self.return_context_only:
            # When used as context dataset, only return img and mask
            # img, mask = self.data[idx]

            target_img, target_mask, *_ = self.data[idx]
            
            # target_img = percentile_normalize(target_img)


            # target_img = random_intensity_augmentation(target_img)
            # target_img = random_invert_intensity(target_img)

            target_img = random_he_augmentation(target_img)
            target_img = enhance_bright_nuclei(target_img, p=0.5)

            # target_img = random_he_augmentation(target_img)




            # target_img = random_color_augmentation(target_img)
            # target_img = random_grayscale(target_img)


            # img = torch.tensor(np.ascontiguousarray(img), dtype=torch.float32, device="cpu").unsqueeze(0)
            # mask = torch.tensor(np.ascontiguousarray(mask), dtype=torch.float32, device="cpu").unsqueeze(0)

    #        return img, mask
        else:
            # Get target sample
            # target_img, target_mask = self.data[idx]

            target_img, target_mask, *_ = self.data[idx]
            # target_img = percentile_normalize(target_img)


            # target_img = random_intensity_augmentation(target_img)
            # target_img = random_invert_intensity(target_img)

            target_img = random_he_augmentation(target_img)
            target_img = enhance_bright_nuclei(target_img, p=0.5)

            # target_img = random_he_augmentation(target_img)




            # target_img = random_color_augmentation(target_img)
            # target_img = random_grayscale(target_img)


            # target_img = torch.tensor(np.ascontiguousarray(target_img), dtype=torch.float32, device="cpu").unsqueeze(0)  # [1, C, H, W]
            # target_mask = torch.tensor(np.ascontiguousarray(target_mask), dtype=torch.float32, device="cpu").unsqueeze(0)  # [1, C, H, W]

            

            # Get context samples (different from target and sequential)
            # We'll take the next 'context_size' samples after the target

            # context_indices = range(idx + 1, idx + 1 + self.context_size)

            # Random context size tussen 1 en max context_size
            # k = random.randint(1, self.context_size)
            k = self.context_size

            # Random context samples kiezen
            # available_indices = list(range(len(self.data)))
            # available_indices.remove(idx)

            # context_indices = random.sample(available_indices, k)

            available_indices = list(range(len(self.context_dataset)))
            context_indices = random.sample(available_indices, k)

            context_imgs = []
            context_masks = []
            for context_idx in context_indices:
                # c_img, c_mask = self.data[context_idx]


                # c_img, c_mask, *_ = self.data[context_idx]
                c_img, c_mask, *_ = self.context_dataset[context_idx]
                # c_img = percentile_normalize(c_img)


                # c_img = random_intensity_augmentation(c_img)
                # c_img = random_invert_intensity(c_img)

                # c_img = random_he_augmentation(c_img)
                # c_img = enhance_bright_nuclei(c_img, p=0.5)

                # c_img = random_he_augmentation(c_img)




                # c_img = random_color_augmentation(c_img)
                # c_img = random_grayscale(c_img)


                #################################################################### NIUEW
                # c_img = torch.tensor(np.ascontiguousarray(c_img), dtype=torch.float32, device='cpu').unsqueeze(0)  # [1, C, H, W]
                # c_mask = torch.tensor(np.ascontiguousarray(c_mask), dtype=torch.float32, device='cpu').unsqueeze(0) 

                c_img = torch.tensor(np.ascontiguousarray(c_img), dtype=torch.float32, device='cpu').permute(2, 0, 1)
                c_mask = torch.tensor(np.ascontiguousarray(c_mask), dtype=torch.float32, device='cpu').unsqueeze(0)

                ############################################################################
                context_imgs.append(c_img)
                context_masks.append(c_mask)
            # Stack context samples along the sequence dimension
            # Pad naar vaste lengte
            # while len(context_imgs) < self.context_size:
            #     context_imgs.append(torch.zeros_like(context_imgs[0]))
            #     context_masks.append(torch.zeros_like(context_masks[0]))


            context_img = torch.stack(context_imgs, dim=0)  # [4, 1, C, H, W]
            context_mask = torch.stack(context_masks, dim=0)  # [4, 1, C, H, W]            
            

        ###################################################################################### NIUEW
            # Add batch dimension and adjust dimensions
            # context_img = context_img.unsqueeze(0)  # [1, 4, 1, H, W]
            # context_mask = context_mask.unsqueeze(0)  # [1, 4, 1, H, W]

            target_img = torch.tensor(np.ascontiguousarray(target_img), dtype=torch.float32, device="cpu").permute(2, 0, 1)
            target_mask = torch.tensor(np.ascontiguousarray(target_mask), dtype=torch.float32, device="cpu").unsqueeze(0)

        

        
        # return (
        #     target_img.unsqueeze(0),       # [1, 4, H, W]
        #     target_mask.unsqueeze(0),            # [1, 1, H, W]
        #     context_img,      # [1, 4, 4, H, W]
        #     context_mask       # [1, 4, 1, H, W]
        # )

        return (
            target_img,       # [3, H, W]
            target_mask,      # [1, H, W]
            context_img,      # [L, 3, H, W]
            context_mask      # [L, 1, H, W]
        )
        ###############################################################################################

def context_features(img, mask):
    gray = img.mean(axis=2).astype(np.float32)

    brightness = gray.mean()
    contrast = gray.std()
    fg_ratio = mask.mean()

    hist, _ = np.histogram(gray, bins=32, range=(0, 1), density=True)
    hist = hist / (hist.sum() + 1e-6)

    return brightness, contrast, fg_ratio, hist


def context_distance(target_img, target_mask, ctx_img, ctx_mask):
    tb, tc, tf, th = context_features(target_img, target_mask)
    cb, cc, cf, ch = context_features(ctx_img, ctx_mask)

    brightness_dist = abs(tb - cb)
    contrast_dist = abs(tc - cc)
    foreground_dist = abs(tf - cf)
    hist_dist = np.linalg.norm(th - ch)

    return (
        1.0 * brightness_dist +
        1.0 * contrast_dist +
        2.0 * foreground_dist +
        1.0 * hist_dist
    )



class EvalDataset(Dataset):
    """Evaluation dataset with same channel padding as TrainDataset and 4 context samples"""
    def __init__(self, target_data, context_dataset, context_size=8):
        self.target_data = target_data
        self.context_dataset = context_dataset
        #self.context_dataset = [(img, mask, cls) if len(item) == 3 else (img, mask, None) for item in context_dataset for img, mask, *cls in [item]]
        self.context_size = context_size
   
    def __len__(self):
        return len(self.target_data)
    
    def __getitem__(self, idx):
        # target_img, target_mask = self.target_data[idx]
        # target_img, target_mask, stain = self.target_data[idx]
        target_img, target_mask, stain, sample_id = self.target_data[idx]



        # target_img = percentile_normalize(target_img)
        ################################################################################################ NIEUW
        # target_img = torch.tensor(np.ascontiguousarray(target_img), dtype=torch.float32)  # [H, W]
        target_img = torch.tensor(np.ascontiguousarray(target_img), dtype=torch.float32).permute(2, 0, 1)
        target_mask = torch.tensor(np.ascontiguousarray(target_mask), dtype=torch.float32).unsqueeze(0)

        

        # target_mask = torch.tensor(np.ascontiguousarray(target_mask), dtype=torch.float32)  # [H, W]
    
        #############################################################################################
   
        # # --- Find k closest context samples based on L2 distance ---
        # distances = []
        # for context_img, context_mask, *_ in self.context_dataset:

        #     ######################################################################################### NIUEW
        #     # ctx_tensor = torch.tensor(np.ascontiguousarray(context_img), dtype=torch.float32)
        #     ctx_tensor = torch.tensor(
        #         np.ascontiguousarray(context_img),
        #         dtype=torch.float32
        #     ).permute(2, 0, 1)
        #     #############################################################################
        #     # distances.append(torch.norm(target_img - ctx_tensor).item())
        #     same_stain_context = [
        #         item for item in self.context_dataset if item[2] == stain
        #     ]
        
        # sorted_indices = np.argsort(distances)[:self.context_size]

        # # Select the top-k most similar context samples
        # context_imgs = []
        # context_masks = []



        # --- Find k closest context samples, preferably from same stain ---
        same_stain_context = [
            item for item in self.context_dataset if item[2] == stain
        ]

        # Fallback: als er geen context samples met dezelfde stain zijn
        # candidate_context = same_stain_context if len(same_stain_context) > 0 else self.context_dataset

        candidate_context = same_stain_context if len(same_stain_context) > 0 else self.context_dataset

        # Filter slechte context masks
        filtered_context = []
        for item in candidate_context:
            ctx_img, ctx_mask, *_ = item
            mask_ratio = ctx_mask.mean()

            if 0.01 < mask_ratio < 0.70:
                filtered_context.append(item)

        # fallback als filter te streng is
        candidate_context = filtered_context if len(filtered_context) >= self.context_size else candidate_context






        distances = []
        for context_img, context_mask, *_ in candidate_context:
            # context_img = percentile_normalize(context_img)
            ctx_tensor = torch.tensor(
                np.ascontiguousarray(context_img),
                dtype=torch.float32
            ).permute(2, 0, 1)

            # distances.append(torch.norm(target_img - ctx_tensor).item())
            target_gray = target_img.mean(dim=0)
            target_gray = (target_gray - target_gray.mean()) / (target_gray.std() + 1e-6)

            ctx_gray = ctx_tensor.mean(dim=0)
            ctx_gray = (ctx_gray - ctx_gray.mean()) / (ctx_gray.std() + 1e-6)

            distances.append(torch.norm(target_gray - ctx_gray).item())

        sorted_indices = np.argsort(distances)[:self.context_size]



        # distances = []

        # target_img_np = target_img.permute(1, 2, 0).numpy()
        # target_mask_np = target_mask.squeeze(0).numpy()

        # for context_img, context_mask, *_ in candidate_context:
        #     dist = context_distance(
        #         target_img_np,
        #         target_mask_np,
        #         context_img,
        #         context_mask
        #     )
        #     distances.append(dist)

        # sorted_indices = np.argsort(distances)[:self.context_size]


        context_imgs = []
        context_masks = []



        if len(candidate_context) == 0:
            context_imgs = [
                np.zeros_like(target_img.permute(1, 2, 0).numpy())
                for _ in range(self.context_size)
            ]
            context_masks = [
                np.zeros_like(target_mask.squeeze(0).numpy())
                for _ in range(self.context_size)
            ]
        else:
            for i in sorted_indices:
                ctx_img, ctx_mask, *_ = candidate_context[i]
                context_imgs.append(np.ascontiguousarray(ctx_img))
                context_masks.append(np.ascontiguousarray(ctx_mask))


            while len(context_imgs) < self.context_size:
                context_imgs.append(np.zeros_like(context_imgs[0]))
                context_masks.append(np.zeros_like(context_masks[0]))

        ############################################################################# NIEUW
        # Convert to tensors
        # context_imgs_tensor = torch.stack([
        #     torch.tensor(img, dtype=torch.float32) for img in context_imgs
        # ])  # [C, H, W]
        # context_masks_tensor = torch.stack([
        #     torch.tensor(mask, dtype=torch.float32) for mask in context_masks
        # ])  # [C, H, W]

        context_imgs_tensor = torch.stack([
            torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) for img in context_imgs
        ])
        context_masks_tensor = torch.stack([
            torch.tensor(mask, dtype=torch.float32).unsqueeze(0) for mask in context_masks
        ])

    
        # return (
        #     target_img.unsqueeze(0).unsqueeze(0),
        #     target_mask.unsqueeze(0).unsqueeze(0),
        #     context_imgs_tensor.unsqueeze(1) ,
        #     context_masks_tensor.unsqueeze(1)  
        # )

        # return (
        #     target_img,
        #     target_mask,
        #     context_imgs_tensor,
        #     context_masks_tensor,
        #     stain
        # )

        return (
            target_img,
            target_mask,
            context_imgs_tensor,
            context_masks_tensor,
            stain,
            sample_id
        )
                    
        ######################################################################################


class UltrasoundDataModule(LightningDataModule):
    # def __init__(self, X_train, X_init, X_val, X_test, batch_size=4,num_workers=0):
    #     super().__init__()
    #     self.X_train = X_train
    #     self.X_init = X_init
    #     self.X_val = X_val
    #     self.X_test = X_test
    #     self.batch_size = batch_size
    #     self.num_workers=num_workers


    def __init__(self, X_train, train_context, X_val, X_test, test_context, batch_size=4, num_workers=0):
        super().__init__()
        self.X_train = X_train
        self.train_context = train_context
        self.X_val = X_val
        self.X_test = X_test
        self.test_context = test_context
        self.batch_size = batch_size
        self.num_workers = num_workers





    def setup(self, stage=None):
        # # Training set
        # self.train_dataset = TrainDataset(self.X_train, self.X_init,context_size=16) # allemaal 16
        # # Validation/Test sets
        # self.val_dataset = EvalDataset(self.X_val, self.X_init, context_size=16)
        # self.test_dataset = EvalDataset(self.X_test, self.X_init, context_size=16)

        self.train_dataset = TrainDataset(
            self.X_train,
            self.train_context,
            context_size=4
        )

        self.val_dataset = EvalDataset(
            self.X_val,
            self.train_context,
            context_size=4
        )

        self.test_dataset = EvalDataset(
            self.X_test,
            self.test_context,
            context_size=4
        )
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.num_workers)
# =============================================================================
# Prepare the datasets and dataloaders
# =============================================================================

# Prepare the data module
# data_module = UltrasoundDataModule(X, X_init, V, Y, batch_size=4,num_workers=8)

# data_module = UltrasoundDataModule(
#     X,
#     train_context,
#     V,
#     Y,
#     test_context,
#     batch_size=4,
#     num_workers=8
# )


if __name__ == "__main__":

    # data_module = UltrasoundDataModule(
    #     X,
    #     X_init,
    #     V,
    #     Y,
    #     batch_size=4,
    #     num_workers=8
    # )

    data_module = UltrasoundDataModule(
        X,
        train_context,
        V,
        Y,
        test_context,
        batch_size=4,
        num_workers=8
    )

    # Manually call setup to initialize the datasets
    data_module.setup()  # This will create train_dataset, val_dataset, and test_dataset

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

    # logger = TensorBoardLogger("iclnoise", name="model_logs")

    # logger = TensorBoardLogger(
    #     "iclnoise",
    #     name=EXPERIMENT_NAME
    # )

    tb_logger = TensorBoardLogger(
        "iclnoise",
        name=EXPERIMENT_NAME
    )

    csv_logger = CSVLogger(
        "iclnoise_csv",
        name=EXPERIMENT_NAME
    )

    logger = [tb_logger, csv_logger]

    # Initialize your model
    hparams = {
        "learning_rate": 1e-5,
    }

    model = LightningModel(hparams)

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        # filename='best-{epoch}-{val_loss:2f}-',
        filename=f'{EXPERIMENT_NAME}-{{epoch}}-{{val_loss:.4f}}',
        every_n_epochs=1
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=8,
        mode='min'
    )

    ###################################################################### NIUEW
    # Initialize trainer
    # trainer = pl.Trainer(
    #     max_epochs=30,
    #     callbacks=[checkpoint_callback, early_stop_callback],
    #     logger=logger,
    #     accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    #     devices=2,
    #     strategy="ddp_find_unused_parameters_true"
    # )

    # trainer = pl.Trainer(
    #     max_epochs=100,
    #     accelerator="gpu",
    #     devices=1,
    #     log_every_n_steps=10,
    #     enable_checkpointing=False
    # )

    trainer = pl.Trainer(
        max_epochs=150,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=10
    )

    ######################################################################

    # # Log dataset sizes
    logging.info(f"Training started with {len(data_module.train_dataset)} training samples")
    logging.info(f"Validation samples: {len(data_module.val_dataset)}")
    logging.info(f"Test samples: {len(data_module.test_dataset)}")

    # Train the model
    # trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader()) ################################################# TRAIN UIT



    #model = LightningModel.load_from_checkpoint("lightning_radboud_noise_BLOCK/model_logs/version_1/checkpoints/best-epoch=20-val_loss=0.03-.ckpt", strict=False)
    #model = LightningModel.load_from_checkpoint("lightning_iclnoise_camus/model_logs/version_0/checkpoints/best-epoch=35-val_dice=0.949292-.ckpt", strict=False)
    logging.info("Training complete")

    # Test the model
    logging.info("Starting test phase...")

    model.eval()
    # 4. Move the model to the appropriate device
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # test_results = trainer.test(model, data_module.test_dataloader())

    test_loader = DataLoader(
        data_module.test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )

    ################################################################ model test w/out augmentation
    # model = LightningModel.load_from_checkpoint(
    #     "iclnoise/26mei_TRAINHElizard_TESTHEcellbindbCONTEXT_ctx16/version_0/checkpoints/26mei_TRAINHElizard_TESTHEcellbindbCONTEXT_ctx16-epoch=19-val_loss=0.6501.ckpt",
    #     hparams=hparams
    # )

    ################################################################ model test w/ augmentation
    # model = LightningModel.load_from_checkpoint(
    #     "iclnoise/27mei_THEliz_TESTHEbindbCONTEXT_CTXIMPROVEMENTS+HEAUGEMENTATION_ctx16/version_0/checkpoints/27mei_THEliz_TESTHEbindbCONTEXT_CTXIMPROVEMENTS+HEAUGEMENTATION_ctx16-epoch=39-val_loss=0.5843.ckpt",
    #     hparams=hparams
    # )



    ######*****************************************************************************
    model = LightningModel.load_from_checkpoint(
        "iclnoise/4juni_15_eRUN_ICL_NMB_ctx4_general_VIS_mIForiginal/version_0/checkpoints/4juni_15_eRUN_ICL_NMB_ctx4_general_VIS_mIForiginal-epoch=47-val_loss=0.5479.ckpt",
        hparams=hparams
    )
    ######*****************************************************************************


    # best_model_path = checkpoint_callback.best_model_path
    # print(f"Loading best checkpoint: {best_model_path}")

    # model = LightningModel.load_from_checkpoint(
    #     best_model_path,
    #     hparams=hparams
    # )

    test_results = trainer.test(model, test_loader)


    logging.info(f"Test results: {test_results}")

    best_model_path = checkpoint_callback.best_model_path
    logging.info(f"Best model saved at: {best_model_path}")

    csv_path = (
    f"iclnoise_csv/{EXPERIMENT_NAME}/version_0/metrics.csv"
)

    if os.path.exists(csv_path):

        df = pd.read_csv(csv_path)

        # Loss curve
        plt.figure(figsize=(8,5))

        # if "train_loss" in df.columns:
        #     plt.plot(df["epoch"], df["train_loss"],
        #             label="Train Loss")

        # if "val_loss" in df.columns:
        #     plt.plot(df["epoch"], df["val_loss"],
        #             label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()

        plt.savefig(
            os.path.join(
                model.save_dir,
                "loss_curve.png"
            ),
            dpi=300
        )
        plt.close()


        # Dice curve
        plt.figure(figsize=(8,5))

        if "train_dice" in df.columns:
            plt.plot(df["epoch"], df["train_dice"],
                    label="Train Dice")

        if "val_dice" in df.columns:
            plt.plot(df["epoch"], df["val_dice"],
                    label="Validation Dice")

        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.legend()
        plt.grid()

        plt.savefig(
            os.path.join(
                model.save_dir,
                "dice_curve.png"
            ),
            dpi=300
        )
        plt.close()

        print("Saved training curves.")

        # TensorBoard reminder
        logging.info("Launch TensorBoard with the command: tensorboard --logdir=lightning_logs/")

        plot_convergence(
            csv_path=f"iclnoise_csv/{EXPERIMENT_NAME}/version_0/metrics.csv",
            save_dir=f"results/{EXPERIMENT_NAME}/convergence_plots",
            model_name="ICL-NoiseUNet"
        )

    else:
        print(f"CSV metrics file not found: {csv_path}")