import torch
import torch.nn as nn
from torch.cuda.amp import autocast

class UNetPlusPlus(nn.Module):
    def __init__(self, q=False, use_fp16=False):
        super(UNetPlusPlus, self).__init__()

        # Encoder blocks
        self.conv0_0 = self.conv_block(1, 64)
        self.conv1_0 = self.conv_block(64, 128)
        self.conv2_0 = self.conv_block(128, 256)
        self.conv3_0 = self.conv_block(256, 512)
        self.conv4_0 = self.conv_block(512, 1024)

        # Decoder blocks
        self.conv0_1 = self.conv_block(64 + 64, 64)
        self.conv1_1 = self.conv_block(128 + 128, 128)
        self.conv2_1 = self.conv_block(256 + 256, 256)
        self.conv3_1 = self.conv_block(512 + 512, 512)

        self.conv0_2 = self.conv_block(64*2 + 64, 64)
        self.conv1_2 = self.conv_block(128*2 + 128, 128)
        self.conv2_2 = self.conv_block(256*2 + 256, 256)

        self.conv0_3 = self.conv_block(64*3 + 64, 64)
        self.conv1_3 = self.conv_block(128*3 + 128, 128)

        self.conv0_4 = self.conv_block(64*4 + 64, 64)

        # Upsampling layers
        self.up1_0 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4_0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
        self.__name__ = self.__class__.__name__
        self.q = q
        self.use_fp16 = use_fp16
        if q:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

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
        # Encoder path
        x0_0 = self.conv0_0(x)  # x0,0

        x1_0 = self.conv1_0(self.downsample(x0_0))  # x1,0
        x0_1 = self.conv0_1(torch.cat([self.up4_0(x1_0), x0_0], 1))  # x0,1

        x2_0 = self.conv2_0(self.downsample(x1_0))  # x2,0
        x1_1 = self.conv1_1(torch.cat([self.up3_0(x2_0), x1_0], 1))  # x1,1
        x0_2 = self.conv0_2(torch.cat([self.up4_0(x1_1), x0_0, x0_1], 1))  # x0,2

        x3_0 = self.conv3_0(self.downsample(x2_0))  # x3,0
        x2_1 = self.conv2_1(torch.cat([self.up2_0(x3_0), x2_0], 1))  # x2,1
        x1_2 = self.conv1_2(torch.cat([self.up3_0(x2_1), x1_0, x1_1], 1))  # x1,2
        x0_3 = self.conv0_3(torch.cat([self.up4_0(x1_2), x0_0, x0_1, x0_2], 1))  # x0,3

        x4_0 = self.conv4_0(self.downsample(x3_0))  # x4,0
        x3_1 = self.conv3_1(torch.cat([self.up1_0(x4_0), x3_0], 1))  # x3,1
        x2_2 = self.conv2_2(torch.cat([self.up2_0(x3_1), x2_0, x2_1], 1))  # x2,2
        x1_3 = self.conv1_3(torch.cat([self.up3_0(x2_2), x1_0, x1_1, x1_2], 1))  # x1,3
        x0_4 = self.conv0_4(torch.cat([self.up4_0(x1_3), x0_0, x0_1, x0_2, x0_3], 1))  # x0,4

        output = self.final(x0_4)
        return output

    def downsample(self, x):
        return nn.functional.max_pool2d(x, kernel_size=2, stride=2)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
