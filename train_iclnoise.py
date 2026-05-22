from models.NoiseContext import ContextNoiseUNet

import torch
from torch import nn, optim
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from DataAugmentation import augment_data
import cv2
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
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
from dataloaders import reading_training_data_fetal, reading_camus_data, reading_data, reading_data_tg3k, get_data_jnu,get_frame_labels, read_and_split_busi_data, read_and_split_busbra_data, read_data_jnu,split_training_data, read_histopathology_data, split_leave_one_stain_out






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
        self.save_dir="exp_1"
        os.makedirs(self.save_dir, exist_ok=True)
        self.net=ContextNoiseUNet()
        #self.net = MultiverSegNet(encoder_blocks=[64, 64, 64, 64])
        #self.net=ContextNoiseWNet()
        #self.net=ContextNoiseUNet()
        #self.net=UNETR2D(img_shape=(192, 192), input_dim=1, output_dim=1)
        #self.net=NnUNet(in_channels=1, out_channels=1, base_num_features=16, num_pool=4, ndim=2, deep_supervision=False)
        #self.net=SwinUNet()

       
        
        # Loss function
        self.dice_loss = SoftDiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, target_in, context_in, context_out=None):
        sc = ShapeChecker()
        y_pred = self.net(target_in,context_in,context_out)
        sc.check(y_pred, "B C H W")
        return y_pred 

    def training_step(self, batch, batch_idx):
        target_images, target_masks, context_images,context_masks = batch
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
        target_images, target_masks, context_images,context_mask, stains = batch
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

        # Log everything
        self.log_dict({f"val_{k}": v for k, v in metrics.items()}, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)


        return loss


    def test_step(self, batch, batch_idx):
        target_images, target_masks, context_images,context_mask, stains = batch

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

        # Log everything
        self.log_dict({f"test_{k}": v for k, v in metrics.items()})
        self.log("test_loss", loss)
        img = target_images[0,0].detach().cpu()     # shape [C,H,W] or [H,W]
        gt_mask = target_masks[0,0].detach().cpu()
        pred_mask = pred_masks[0,0].detach().cpu()  # convert logits to [0,1]

        # convert to numpy
        img_np = img.permute(1,2,0).numpy() if img.ndim==3 else img.numpy()
        gt_np = gt_mask.squeeze().numpy()
        pred_np = (pred_mask > 0.5).squeeze().numpy().astype(float)  # binarize
        # Rotate all images 90° clockwise
       
        metrics_path = os.path.join(self.save_dir, "test_metrics.txt")

# Append Dice and IoU for each sample
        with open(metrics_path, "a") as f:
            f.write(
                f"Batch {batch_idx} Stain: {stains[0]} "
                f"Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}\n"
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

        # save
        # save_path = os.path.join(self.save_dir, f"sample_{self.current_epoch}_{batch_idx}.png")


        save_path = os.path.join(
            self.save_dir,
            f"test_batch_{batch_idx}_dice_{metrics['dice']:.3f}_iou_{metrics['iou']:.3f}.png"
        )
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

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
        
        preds = pred > 0.4
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


# =============================================================================
# Data Preparation
# =============================================================================

# X, Y, V = reading_camus_data()

data = read_histopathology_data(os.environ["DATA_DIR"], image_size=192) #192
X, V, Y = split_training_data(data)

X_init = X.copy()
# X = augment_data(X, context=False, target_size=(192, 192))


#################### LEAVE ONE OUT

# data = read_histopathology_data(os.environ["DATA_DIR"], image_size=192)

# X, V, Y = split_leave_one_stain_out(
#     data,
#     test_stain="DAPI"
# )

# X_init = X.copy()






# =============================================================================
# Define dataset and dataloaders
# =============================================================================


class TrainDataset(Dataset):
    def __init__(self, data, context_dataset, context_size=4, return_context_only=False, device='cuda'):
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

            # img = torch.tensor(np.ascontiguousarray(img), dtype=torch.float32, device="cpu").unsqueeze(0)
            # mask = torch.tensor(np.ascontiguousarray(mask), dtype=torch.float32, device="cpu").unsqueeze(0)

    #        return img, mask
        else:
            # Get target sample
            # target_img, target_mask = self.data[idx]

            target_img, target_mask, *_ = self.data[idx]
            # target_img = torch.tensor(np.ascontiguousarray(target_img), dtype=torch.float32, device="cpu").unsqueeze(0)  # [1, C, H, W]
            # target_mask = torch.tensor(np.ascontiguousarray(target_mask), dtype=torch.float32, device="cpu").unsqueeze(0)  # [1, C, H, W]

            

            # Get context samples (different from target and sequential)
            # We'll take the next 'context_size' samples after the target
            context_indices = range(idx + 1, idx + 1 + self.context_size)

            context_imgs = []
            context_masks = []
            for context_idx in context_indices:
                # c_img, c_mask = self.data[context_idx]
                c_img, c_mask, *_ = self.data[context_idx]


                #################################################################### NIUEW
                # c_img = torch.tensor(np.ascontiguousarray(c_img), dtype=torch.float32, device='cpu').unsqueeze(0)  # [1, C, H, W]
                # c_mask = torch.tensor(np.ascontiguousarray(c_mask), dtype=torch.float32, device='cpu').unsqueeze(0) 

                c_img = torch.tensor(np.ascontiguousarray(c_img), dtype=torch.float32, device='cpu').permute(2, 0, 1)
                c_mask = torch.tensor(np.ascontiguousarray(c_mask), dtype=torch.float32, device='cpu').unsqueeze(0)

                ############################################################################
                context_imgs.append(c_img)
                context_masks.append(c_mask)
            # Stack context samples along the sequence dimension
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

class EvalDataset(Dataset):
    """Evaluation dataset with same channel padding as TrainDataset and 4 context samples"""
    def __init__(self, target_data, context_dataset, context_size=4):
        self.target_data = target_data
        self.context_dataset = context_dataset
        #self.context_dataset = [(img, mask, cls) if len(item) == 3 else (img, mask, None) for item in context_dataset for img, mask, *cls in [item]]
        self.context_size = context_size
   
    def __len__(self):
        return len(self.target_data)
    
    def __getitem__(self, idx):
        # target_img, target_mask = self.target_data[idx]
        target_img, target_mask, stain = self.target_data[idx]

        ################################################################################################ NIEUW
        # target_img = torch.tensor(np.ascontiguousarray(target_img), dtype=torch.float32)  # [H, W]
        target_img = torch.tensor(np.ascontiguousarray(target_img), dtype=torch.float32).permute(2, 0, 1)
        target_mask = torch.tensor(np.ascontiguousarray(target_mask), dtype=torch.float32).unsqueeze(0)

        

        # target_mask = torch.tensor(np.ascontiguousarray(target_mask), dtype=torch.float32)  # [H, W]
    
        #############################################################################################
   
        # --- Find k closest context samples based on L2 distance ---
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
        candidate_context = same_stain_context if len(same_stain_context) > 0 else self.context_dataset

        distances = []
        for context_img, context_mask, *_ in candidate_context:
            ctx_tensor = torch.tensor(
                np.ascontiguousarray(context_img),
                dtype=torch.float32
            ).permute(2, 0, 1)

            distances.append(torch.norm(target_img - ctx_tensor).item())

        sorted_indices = np.argsort(distances)[:self.context_size]

        context_imgs = []
        context_masks = []
        for i in sorted_indices:
            ctx_img, ctx_mask, *_ = candidate_context[i]
            context_imgs.append(np.ascontiguousarray(ctx_img))
            context_masks.append(np.ascontiguousarray(ctx_mask))





        for i in sorted_indices:
            ctx_img, ctx_mask, *_ = self.context_dataset[i]
            context_imgs.append(np.ascontiguousarray(ctx_img))
            context_masks.append(np.ascontiguousarray(ctx_mask))

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

        return (
            target_img,
            target_mask,
            context_imgs_tensor,
            context_masks_tensor,
            stain
        )
            
        ######################################################################################


class UltrasoundDataModule(LightningDataModule):
    def __init__(self, X_train, X_init, X_val, X_test, batch_size=4,num_workers=0):
        super().__init__()
        self.X_train = X_train
        self.X_init = X_init
        self.X_val = X_val
        self.X_test = X_test
        self.batch_size = batch_size
        self.num_workers=num_workers
    def setup(self, stage=None):
        # Training set
        self.train_dataset = TrainDataset(self.X_train, self.X_init,context_size=16) # allemaal 16
        # Validation/Test sets
        self.val_dataset = EvalDataset(self.X_val, self.X_init, context_size=16)
        self.test_dataset = EvalDataset(self.X_test, self.X_init, context_size=16)
        
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
data_module = UltrasoundDataModule(X, X_init, V, Y, batch_size=4,num_workers=8)


if __name__ == "__main__":

    data_module = UltrasoundDataModule(
        X,
        X_init,
        V,
        Y,
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

    logger = TensorBoardLogger("iclnoise", name="model_logs")

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
        filename='best-{epoch}-{val_loss:2f}-',
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
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())



    #model = LightningModel.load_from_checkpoint("lightning_radboud_noise_BLOCK/model_logs/version_1/checkpoints/best-epoch=20-val_loss=0.03-.ckpt", strict=False)
    #model = LightningModel.load_from_checkpoint("lightning_iclnoise_camus/model_logs/version_0/checkpoints/best-epoch=35-val_dice=0.949292-.ckpt", strict=False)
    logging.info("Training complete")

    # Test the model
    logging.info("Starting test phase...")

    model.eval()
    # 4. Move the model to the appropriate device
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    test_results = trainer.test(model, data_module.test_dataloader())
    logging.info(f"Test results: {test_results}")

    best_model_path = checkpoint_callback.best_model_path
    logging.info(f"Best model saved at: {best_model_path}")

    # TensorBoard reminder
    logging.info("Launch TensorBoard with the command: tensorboard --logdir=lightning_logs/")