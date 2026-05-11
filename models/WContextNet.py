import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn.vmap import Vmap, vmap
from einops import rearrange
from .nn.noise_maps import residual_noise_map_2d, local_variance_map_2d,residual_noise_map_context, local_variance_map_context


def apply_conv_context(conv, x):
    # x: [B, L, C, H, W]
    B, L, C, H, W = x.shape
    x = rearrange(x, "b l c h w -> (b l) c h w")   # merge batch & context
    x = conv(x)                                    # Conv2d works
    x = rearrange(x, "(b l) c h w -> b l c h w", b=B, l=L)  # reshape back
    return x


# --- Your Original UNet Backbone (kept same as you wrote) ---
class UNet(nn.Module):
    def __init__(self, in_channels=1,noise_maps=False):
        super(UNet, self).__init__()
        conv_fn = getattr(nn, f'Conv{2}d')
        self.noise_maps=noise_maps
        if self.noise_maps:
            self.alpha_res = nn.Parameter(torch.tensor(0.5))
            self.alpha_var = nn.Parameter(torch.tensor(0.5))
            self.alpha_res_ctx = nn.Parameter(torch.tensor(0.5))
            self.alpha_var_ctx = nn.Parameter(torch.tensor(0.5))
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU()
        self.combine_conv_target_1 = Vmap(conv_fn(in_channels=128,
                                                out_channels=64,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)
                                        )
        self.combine_conv_context_1 = Vmap(conv_fn(in_channels=128,
                                         out_channels=64,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
                                         )
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.combine_conv_target_2 = Vmap(conv_fn(in_channels=256,
                                                out_channels=128,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)
                                        )
        self.combine_conv_context_2 = Vmap(conv_fn(in_channels=256,
                                         out_channels=128,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
                                         )

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.combine_conv_target_3 = Vmap(conv_fn(in_channels=512,
                                                out_channels=256,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)
                                        )
        self.combine_conv_context_3 = Vmap(conv_fn(in_channels=512,
                                         out_channels=256,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
                                         )
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.combine_conv_target_4 = Vmap(conv_fn(in_channels=1024,
                                                out_channels=512,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)
                                        )
        self.combine_conv_context_4 = Vmap(conv_fn(in_channels=1024,
                                         out_channels=512,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
                                         )

        self.conv5_1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(1024)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(1024)
        self.relu5_2 = nn.ReLU()
        self.combine_conv_target_5 = Vmap(conv_fn(in_channels=2048,
                                                out_channels=1024,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)
                                        )
        self.combine_conv_context_5 = Vmap(conv_fn(in_channels=2048,
                                         out_channels=1024,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
                                         )

        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up6_context = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        self.conv6_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn6_1 = nn.BatchNorm2d(512)
        self.relu6_1 = nn.ReLU()
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn6_2 = nn.BatchNorm2d(512)
        self.relu6_2 = nn.ReLU()
        self.combine_conv_target_6 = Vmap(conv_fn(in_channels=1024,
                                                out_channels=512,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)
                                        )
        self.combine_conv_context_6 = Vmap(conv_fn(in_channels=1024,
                                         out_channels=512,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
                                         )

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up7_context = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        self.conv7_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn7_1 = nn.BatchNorm2d(256)
        self.relu7_1 = nn.ReLU()
        self.conv7_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7_2 = nn.BatchNorm2d(256)
        self.relu7_2 = nn.ReLU()
        self.combine_conv_target_7 = Vmap(conv_fn(in_channels=512,
                                                out_channels=256,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)
                                        )
        self.combine_conv_context_7 = Vmap(conv_fn(in_channels=512,
                                         out_channels=256,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
                                         )

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up8_context = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.combine_conv_target_8 = Vmap(conv_fn(in_channels=256,
                                                out_channels=128,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)
                                        )
        self.combine_conv_context_8 = Vmap(conv_fn(in_channels=256,
                                         out_channels=128,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
                                         )

        self.conv8_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn8_1 = nn.BatchNorm2d(128)
        self.relu8_1 = nn.ReLU()
        self.conv8_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn8_2 = nn.BatchNorm2d(128)
        self.relu8_2 = nn.ReLU()
        

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up9_context = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.combine_conv_target_9 = Vmap(conv_fn(in_channels=128,
                                                out_channels=64,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1)
                                        )

        self.conv9_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn9_1 = nn.BatchNorm2d(64)
        self.relu9_1 = nn.ReLU()
        self.conv9_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn9_2 = nn.BatchNorm2d(64)
        self.relu9_2 = nn.ReLU()
        

        self.conv10 = nn.Conv2d(64, 1, kernel_size=1)


    def compute_maps(self, x):
        """Compute residual and local variance maps dynamically"""
        res_map = residual_noise_map_2d(x)  # [B,1,H,W]
        var_map = local_variance_map_2d(x)   # [B,1,H,W]
        return res_map, var_map
    
    def compute_context_maps(self, x):
        """Compute residual and local variance maps dynamically"""
        res_map = residual_noise_map_context(x)  # [B,1,H,W]
        var_map = local_variance_map_context(x)   # [B,1,H,W]
        return res_map, var_map

    def forward(self, x, context=None,context_arg=False):

        if context_arg:
            B, L, C, H, W = context.shape
        else:
            B, L, C, H, W = None,None,None,None,None

        # Normal UNet forward, but modulate features with noise maps if provided
        x1 = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        x1 = self.relu1_2(self.bn1_2(self.conv1_2(x1)))
        x1_pool = self.pool1(x1)

        if context_arg:
            context1 = apply_conv_context(self.conv1_1, context)
            context1 = self.bn1_1(context1.reshape(-1, *context1.shape[2:]))
            context1 = self.relu1_1(context1)
            context1_pool = self.pool1(context1)
            context1_pool = rearrange(context1_pool, "(b l) c h w -> b l c h w", b=B, l=L)

            if self.noise_maps:
                res_map, var_map = self.compute_maps(x1_pool)
                res_ctx_map, var_ctx_map = self.compute_context_maps(context1_pool)


                # --- Target features ---
                x1_pool = x1_pool * (1 + self.alpha_res * res_map)
                x1_pool = x1_pool * (1 + self.alpha_var * var_map)

                # --- Context features ---
                res_map_ctx = res_ctx_map.expand_as(context1_pool)
                var_map_ctx = var_ctx_map.expand_as(context1_pool)
                context1_pool = context1_pool * (1 + self.alpha_res_ctx * res_map_ctx)
                context1_pool = context1_pool * (1 + self.alpha_var_ctx * var_map_ctx)

            
            context_target = torch.concat(
                [context1_pool, x1_pool.unsqueeze(1).expand_as(context1_pool)], dim=2)  # B,L,2C,...

            # conv query with support
            target_update = self.combine_conv_target_1(context_target)  # B,L,C,...
            context_update = self.combine_conv_context_1(context_target)
            #print(x1_pool.shape)
            #print(target_update.shape)
            # average
            target_update = target_update.mean(dim=1, keepdim=False)  # B,C,...
            
            # resudual and activation
            x1_pool = F.gelu(x1_pool + target_update)
            context1_pool = F.gelu(context1_pool + context_update)

        x2 = self.relu2_1(self.bn2_1(self.conv2_1(x1_pool)))
        x2 = self.relu2_2(self.bn2_2(self.conv2_2(x2)))
        x2_pool = self.pool2(x2)

        if context_arg:
            context2 = apply_conv_context(self.conv2_1, context1_pool)
            context2 = self.bn2_1(context2.reshape(-1, *context2.shape[2:]))
            context2 = self.relu2_1(context2)
            context2_pool = self.pool2(context2)
            context2_pool = rearrange(context2_pool, "(b l) c h w -> b l c h w", b=B, l=L)
            context_target_2 = torch.concat(
                [context2_pool, x2_pool.unsqueeze(1).expand_as(context2_pool)], dim=2)  # B,L,2C,...
            # conv query with support
            target_update_2 = self.combine_conv_target_2(context_target_2)
            target_update_2 = target_update_2.mean(dim=1, keepdim=False)  # B,C,...
            context_update_2 = self.combine_conv_context_2(context_target_2)
            x2_pool = F.gelu(x2_pool + target_update_2)
            context2_pool = F.gelu(context2_pool + context_update_2)



        x3 = self.relu3_1(self.bn3_1(self.conv3_1(x2_pool)))
        x3 = self.relu3_2(self.bn3_2(self.conv3_2(x3)))
        x3_pool = self.pool3(x3)

        if context_arg:
            context3 = apply_conv_context(self.conv3_1, context2_pool)
            context3 = self.bn3_1(context3.reshape(-1, *context3.shape[2:]))
            context3 = self.relu3_2(context3)
            context3_pool = self.pool3(context3)
            context3_pool = rearrange(context3_pool, "(b l) c h w -> b l c h w", b=B, l=L)
            context_target_3 = torch.concat([context3_pool, x3_pool.unsqueeze(1).expand_as(context3_pool)], dim=2)  
            target_update_3 = self.combine_conv_target_3(context_target_3)
            target_update_3 = target_update_3.mean(dim=1, keepdim=False)  # B,C,...
            context_update_3 = self.combine_conv_context_3(context_target_3)
            x3_pool = F.gelu(x3_pool + target_update_3)
            context3_pool = F.gelu(context3_pool + context_update_3)

        x4 = self.relu4_1(self.bn4_1(self.conv4_1(x3_pool)))
        x4 = self.relu4_2(self.bn4_2(self.conv4_2(x4)))
        x4_pool = self.pool4(x4)

        if context_arg:
            context4 = apply_conv_context(self.conv4_1, context3_pool)
            context4 = self.bn4_1(context4.reshape(-1, *context4.shape[2:]))
            context4 = self.relu4_2(context4)
            context4_pool = self.pool4(context4)
            context4_pool = rearrange(context4_pool, "(b l) c h w -> b l c h w", b=B, l=L)
            context_target_4 = torch.concat([context4_pool, x4_pool.unsqueeze(1).expand_as(context4_pool)], dim=2)  
            target_update_4 = self.combine_conv_target_4(context_target_4)
            target_update_4 = target_update_4.mean(dim=1, keepdim=False)  # B,C,...
            context_update_4 = self.combine_conv_context_4(context_target_4)
            x4_pool = F.gelu(x4_pool + target_update_4)
            context4_pool = F.gelu(context4_pool + context_update_4)

        x5 = self.relu5_1(self.bn5_1(self.conv5_1(x4_pool)))
        x5 = self.relu5_2(self.bn5_2(self.conv5_2(x5)))

        if context_arg:
            context5 = apply_conv_context(self.conv5_1, context4_pool)
            context5 = self.bn5_1(context5.reshape(-1, *context5.shape[2:]))
            context5 = self.relu5_2(context5)
            context5 = rearrange(context5, "(b l) c h w -> b l c h w", b=B, l=L)
            if self.noise_maps:
                res_map, var_map = self.compute_maps(x5)
                res_ctx_map, var_ctx_map = self.compute_context_maps(context5)


                # --- Target features ---
                x5 = x5 * (1 + self.alpha_res * res_map)
                x5 = x5 * (1 + self.alpha_var * var_map)

                # --- Context features ---
                res_map_ctx = res_ctx_map.expand_as(context5)
                var_map_ctx = var_ctx_map.expand_as(context5)
                context5 = context5 * (1 + self.alpha_res_ctx * res_map_ctx)
                context5 = context5 * (1 + self.alpha_var_ctx * var_map_ctx)

            context_target_5 = torch.concat([context5, x5.unsqueeze(1).expand_as(context5)], dim=2)  
            target_update_5 = self.combine_conv_target_5(context_target_5)
            target_update_5 = target_update_5.mean(dim=1, keepdim=False)  # B,C,...
            context_update_5 = self.combine_conv_context_5(context_target_5)
            x5 = F.gelu(x5 + target_update_5)
            context5 = F.gelu(context5 + context_update_5)


        # Upward path
        x6 = self.up6(x5)
        x6 = torch.cat([x6, x4], dim=1)
        x6 = self.relu6_1(self.bn6_1(self.conv6_1(x6)))
        x6 = self.relu6_2(self.bn6_2(self.conv6_2(x6)))

        if context_arg:
            context6 = rearrange(context5, "b l c h w -> (b l) c h w")
            context4_pool = rearrange(context4_pool, "b l c h w -> (b l) c h w")
            context4_pool= self.up6_context(context4_pool)
            context6 = self.up6(context6)

            context6 = torch.cat([context6, context4_pool], dim=1)
   # merge batch & context
            context6 = self.conv6_1(context6)
            context6 = self.bn6_1(context6)
            context6 = self.relu6_2(context6)
            context6 = rearrange(context6, "(b l) c h w -> b l c h w", b=B, l=L)
            context_target_6 = torch.concat([context6, x6.unsqueeze(1).expand_as(context6)], dim=2)  
            target_update_6 = self.combine_conv_target_6(context_target_6)
            target_update_6 = target_update_6.mean(dim=1, keepdim=False)  # B,C,...
            context_update_6 = self.combine_conv_context_6(context_target_6)
            x6 = F.gelu(x6 + target_update_6)
            context6 = F.gelu(context6 + context_update_6)


        x7 = self.up7(x6)
        x7 = torch.cat([x7, x3], dim=1)
        x7 = self.relu7_1(self.bn7_1(self.conv7_1(x7)))
        x7 = self.relu7_2(self.bn7_2(self.conv7_2(x7)))

        if context_arg:
            context7 = rearrange(context6, "b l c h w -> (b l) c h w")
            context3_pool = rearrange(context3_pool, "b l c h w -> (b l) c h w")
            context3_pool= self.up7_context(context3_pool)
            context7 = self.up7(context7)

            context7 = torch.cat([context7, context3_pool], dim=1)
   # merge batch & context
            context7 = self.conv7_1(context7)
            context7 = self.bn7_1(context7)
            context7 = self.relu7_2(context7)
            context7 = rearrange(context7, "(b l) c h w -> b l c h w", b=B, l=L)
            context_target_7 = torch.concat([context7, x7.unsqueeze(1).expand_as(context7)], dim=2)  
            target_update_7 = self.combine_conv_target_7(context_target_7)
            target_update_7 = target_update_7.mean(dim=1, keepdim=False)  # B,C,...
            context_update_7 = self.combine_conv_context_7(context_target_7)
            x7 = F.gelu(x7 + target_update_7)
            context7 = F.gelu(context7 + context_update_7)



        x8 = self.up8(x7)
        x8 = torch.cat([x8, x2], dim=1)
        x8 = self.relu8_1(self.bn8_1(self.conv8_1(x8)))
        x8 = self.relu8_2(self.bn8_2(self.conv8_2(x8)))

        if context_arg:
            context8 = rearrange(context7, "b l c h w -> (b l) c h w")
            context2_pool = rearrange(context2_pool, "b l c h w -> (b l) c h w")
            context2_pool= self.up8_context(context2_pool)
            context8 = self.up8(context8)

            context8 = torch.cat([context8, context2_pool], dim=1)
   # merge batch & context
            context8 = self.conv8_1(context8)
            context8 = self.bn8_1(context8)
            context8 = self.relu8_2(context8)
            context8 = rearrange(context8, "(b l) c h w -> b l c h w", b=B, l=L)
            context_target_8 = torch.concat([context8, x8.unsqueeze(1).expand_as(context8)], dim=2)  
            target_update_8 = self.combine_conv_target_8(context_target_8)
            target_update_8 = target_update_8.mean(dim=1, keepdim=False)  # B,C,...
            context_update_8 = self.combine_conv_context_8(context_target_8)
            x8 = F.gelu(x8 + target_update_8)
            context8 = F.gelu(context8 + context_update_8)



        x9 = self.up9(x8)
        x9 = torch.cat([x9, x1], dim=1)
        x9 = self.relu9_1(self.bn9_1(self.conv9_1(x9)))
        x9 = self.relu9_2(self.bn9_2(self.conv9_2(x9)))

        if context_arg:
            context9 = rearrange(context8, "b l c h w -> (b l) c h w")
            context1_pool = rearrange(context1_pool, "b l c h w -> (b l) c h w")
            context1_pool= self.up9_context(context1_pool)
            context9 = self.up9(context9)

            context9 = torch.cat([context9, context1_pool], dim=1)
   # merge batch & context
            context9 = self.conv9_1(context9)
            context9 = self.bn9_1(context9)
            context9 = self.relu9_2(context9)
            context9 = rearrange(context9, "(b l) c h w -> b l c h w", b=B, l=L)
            context_target_9 = torch.concat([context9, x9.unsqueeze(1).expand_as(context9)], dim=2)  
            target_update_9 = self.combine_conv_target_9(context_target_9)
            target_update_9 = target_update_9.mean(dim=1, keepdim=False)  # B,C,...
            x9 = F.gelu(x9 + target_update_9)


        x10 = self.conv10(x9)

        return x10

# --- WNet with Context + Noise ---
class ContextNoiseWNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet1 = UNet(in_channels=1,noise_maps=False)
        self.unet2 = UNet(in_channels=1,noise_maps=False)
        # trainable scalars for modulation
        #self.alpha_res = nn.Parameter(torch.tensor(0.5))
        #self.beta_var = nn.Parameter(torch.tensor(0.5))

    def forward(self, target, context):
        # compute noise maps on target
        #res_map = residual_noise_map_2d(target)
        #var_map = local_variance_map_2d(target)

        # 1st UNet: segment target with modulation
        #out1 = self.unet1(target, res_map=res_map, var_map=var_map,
        #                  alpha=self.alpha_res, beta=self.beta_var)

        # 2nd UNet: refine using context
        #out2 = self.unet2(context, res_map=res_map, var_map=var_map,
        #                  alpha=self.alpha_res, beta=self.beta_var)
        out1 = self.unet1(x=target,context=context,context_arg=True)
        #out1 = torch.sigmoid(out1_feat)

        
        # Second U-Net reverse pass, using the output of the first U-Net
        out2 = self.unet2(out1,context=None,context_arg=False)


        return out2
