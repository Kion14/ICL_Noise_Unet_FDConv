import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as quantization
from torch.cuda.amp import autocast



class UNet(nn.Module):
    def __init__(self, q=False, use_fp16=False):
        super(UNet, self).__init__()

        # Downward path
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(1024)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(1024)
        self.relu5_2 = nn.ReLU()

        # Upward path
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(512)
        self.relu6_1 = nn.ReLU()
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(512)
        self.relu6_2 = nn.ReLU()

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn7_1 = nn.BatchNorm2d(256)
        self.relu7_1 = nn.ReLU()
        self.conv7_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7_2 = nn.BatchNorm2d(256)
        self.relu7_2 = nn.ReLU()

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn8_1 = nn.BatchNorm2d(128)
        self.relu8_1 = nn.ReLU()
        self.conv8_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn8_2 = nn.BatchNorm2d(128)
        self.relu8_2 = nn.ReLU()

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn9_1 = nn.BatchNorm2d(64)
        self.relu9_1 = nn.ReLU()
        self.conv9_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn9_2 = nn.BatchNorm2d(64)
        self.relu9_2 = nn.ReLU()

        self.conv10 = nn.Conv2d(64, 1, kernel_size=1)

        self.__name__ = self.__class__.__name__
        self.q = q
        self.use_fp16 = use_fp16
        if q:
            self.quant = quantization.QuantStub()
            self.dequant = quantization.DeQuantStub()

    def forward(self, x):
        if self.q:
            x = self.quant(x)

        if self.use_fp16:
            with autocast():
                x = self._forward_impl(x)
        else:
            x = self._forward_impl(x)

        if self.q:
            x = self.dequant(x)

        return x

    def _forward_impl(self, x):
        # Downward path
        x1 = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        x1 = self.relu1_2(self.bn1_2(self.conv1_2(x1)))
        x1_pool = self.pool1(x1)

        x2 = self.relu2_1(self.bn2_1(self.conv2_1(x1_pool)))
        x2 = self.relu2_2(self.bn2_2(self.conv2_2(x2)))
        x2_pool = self.pool2(x2)

        x3 = self.relu3_1(self.bn3_1(self.conv3_1(x2_pool)))
        x3 = self.relu3_2(self.bn3_2(self.conv3_2(x3)))
        x3_pool = self.pool3(x3)

        x4 = self.relu4_1(self.bn4_1(self.conv4_1(x3_pool)))
        x4 = self.relu4_2(self.bn4_2(self.conv4_2(x4)))
        x4_pool = self.pool4(x4)

        x5 = self.relu5_1(self.bn5_1(self.conv5_1(x4_pool)))
        x5 = self.relu5_2(self.bn5_2(self.conv5_2(x5)))

        # Upward path
        x6 = self.up6(x5)
        x6 = torch.cat([x6, x4], dim=1)
        x6 = self.relu6_1(self.bn6_1(self.conv6_1(x6)))
        x6 = self.relu6_2(self.bn6_2(self.conv6_2(x6)))

        x7 = self.up7(x6)
        x7 = torch.cat([x7, x3], dim=1)
        x7 = self.relu7_1(self.bn7_1(self.conv7_1(x7)))
        x7 = self.relu7_2(self.bn7_2(self.conv7_2(x7)))

        x8 = self.up8(x7)
        x8 = torch.cat([x8, x2], dim=1)
        x8 = self.relu8_1(self.bn8_1(self.conv8_1(x8)))
        x8 = self.relu8_2(self.bn8_2(self.conv8_2(x8)))

        x9 = self.up9(x8)
        x9 = torch.cat([x9, x1], dim=1)
        x9 = self.relu9_1(self.bn9_1(self.conv9_1(x9)))
        x9 = self.relu9_2(self.bn9_2(self.conv9_2(x9)))

        x10 = self.conv10(x9)

        return x10


class WNet(nn.Module):
    def __init__(self, q=False, use_fp16=False):
        super(WNet, self).__init__()

        # First U-Net (Forward Path)
        self.unet1 = UNet(q=q, use_fp16=use_fp16)
        
        # Second U-Net (Reverse Path)
        self.unet2 = UNet(q=q, use_fp16=use_fp16)

        self.__name__ = self.__class__.__name__
        self.q = q
        self.use_fp16 = use_fp16

    def forward(self, x):
        # First U-Net forward pass
        out1 = self.unet1(x)
        
        # Second U-Net reverse pass, using the output of the first U-Net
        out2 = self.unet2(out1)

        return torch.sigmoid(out2)
    
