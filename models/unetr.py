import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ------------------ Basic Blocks ------------------

class SingleDeconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=2)

    def forward(self, x):
        return self.block(x)


class SingleConv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                               stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        return self.block(x)


class Conv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv2DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv2DBlock(in_planes, out_planes),
            SingleConv2DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


# ------------------ Transformer Components ------------------

class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = embed_dim // num_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.vis = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attn_probs = self.softmax(attn_scores)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_context_shape)

        out = self.out(context)
        out = self.proj_dropout(out)
        return out, None


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=768, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings2D(nn.Module):
    def __init__(self, input_dim, embed_dim, img_shape, patch_size, dropout):
        super().__init__()
        self.H, self.W = img_shape
        self.patch_size = patch_size
        self.n_patches = (self.H // patch_size) * (self.W // patch_size)

        self.patch_embeddings = nn.Conv2d(input_dim, embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)       # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)   # (B, n_patches, embed_dim)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(num_heads, embed_dim, dropout)
        self.mlp = PositionwiseFeedForward(embed_dim, 2048, dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, _ = self.attn(x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
        return x, _


class Transformer2D(nn.Module):
    def __init__(self, input_dim, embed_dim, img_shape, patch_size,
                 num_heads, num_layers, dropout, extract_layers):
        super().__init__()
        self.embeddings = Embeddings2D(input_dim, embed_dim, img_shape, patch_size, dropout)
        self.layer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.extract_layers = extract_layers

    def forward(self, x):
        features = []
        hidden_states = self.embeddings(x)
        for depth, blk in enumerate(self.layer):
            hidden_states, _ = blk(hidden_states)
            if depth + 1 in self.extract_layers:
                features.append(hidden_states)
        return features


# ------------------ UNETR 2D ------------------

class UNETR2D(nn.Module):
    def __init__(self, img_shape=(256, 256), input_dim=3, output_dim=1,
                 embed_dim=768, patch_size=16, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        self.H_p, self.W_p = img_shape[0] // patch_size, img_shape[1] // patch_size

        self.transformer = Transformer2D(input_dim, embed_dim, img_shape, patch_size,
                                         num_heads, self.num_layers, dropout, self.ext_layers)

        # Decoder
        self.decoder0 = nn.Sequential(
            Conv2DBlock(input_dim, 32),
            Conv2DBlock(32, 64)
        )

        self.decoder3 = nn.Sequential(
            Deconv2DBlock(embed_dim, 512),
            Deconv2DBlock(512, 256),
            Deconv2DBlock(256, 128)
        )

        self.decoder6 = nn.Sequential(
            Deconv2DBlock(embed_dim, 512),
            Deconv2DBlock(512, 256)
        )

        self.decoder9 = Deconv2DBlock(embed_dim, 512)
        self.decoder12_upsampler = SingleDeconv2DBlock(embed_dim, 512)

        self.decoder9_upsampler = nn.Sequential(
            Conv2DBlock(1024, 512),
            Conv2DBlock(512, 512),
            Conv2DBlock(512, 512),
            SingleDeconv2DBlock(512, 256)
        )

        self.decoder6_upsampler = nn.Sequential(
            Conv2DBlock(512, 256),
            Conv2DBlock(256, 256),
            SingleDeconv2DBlock(256, 128)
        )

        self.decoder3_upsampler = nn.Sequential(
            Conv2DBlock(256, 128),
            Conv2DBlock(128, 128),
            SingleDeconv2DBlock(128, 64)
        )

        self.decoder0_header = nn.Sequential(
            Conv2DBlock(128, 64),
            Conv2DBlock(64, 64),
            SingleConv2DBlock(64, output_dim, 1)
        )

    def forward(self, x,context=None,context_out=None):
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z

        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, self.H_p, self.W_p)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, self.H_p, self.W_p)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, self.H_p, self.W_p)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, self.H_p, self.W_p)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return output


# ------------------ Test ------------------
if __name__ == "__main__":
    model = UNETR2D(img_shape=(256, 256), input_dim=1, output_dim=1)
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(y.shape)  # -> (1, 1, 256, 256)
