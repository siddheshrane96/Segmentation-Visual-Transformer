# src/segformer/models/encoder.py
import torch
import torch.nn as nn
from .transformer import TransformerEncoder

class PatchEmbedding(nn.Module):
    """
    in_channel - input channels (3 for RGB image)
    d_model - dimension of encoded token
    patch_size - size of each patch in pixels (eg. 16)

    Hout = floor((Hin - f + 2p) / stride) + 1
    for f = 3, p=1 and s=2
    Hout = ((Hin - 1)/2)+1
    for 240x320:
    s1-> 239/2 + 1 -> 120 H/2
    s2-> 119/2 + 1 -> 60 H/4
    s3-> 59/2 + 1 -> 30 H/8
    s4 -> 29/2 + 1 -> 15 H/16

    """
    def __init__(self, in_channels, d_model, patch_size):
        super().__init__()
        assert patch_size == 16, 'Four 2x downsampling stages require patch_size=16'
        assert d_model % 8 == 0
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU()
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.GELU()
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128), nn.GELU()
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, d_model), nn.GELU()
        )
    def forward(self, X):
        s1 = self.stage1(X)  # B,32,H/2,W/2
        s2 = self.stage2(s1) # B,64,H/4,W/4
        s3 = self.stage3(s2) # B,128,H/8,W/8
        s4 = self.stage4(s3) # B,D_model,H/16,W/16
        tokens = s4.flatten(2).transpose(1,2) # shape (B, H/16 * W/16, D_model)!
        return tokens, s1, s2
    
class SegFormerEncoder(nn.Module):
    def __init__(self, image_size=(240,320), patch_size=16, d_model=192, num_heads=4, num_layers=4):
        super().__init__()
        self.H, self.W = image_size
        assert self.H % patch_size == 0 and self.W % patch_size == 0
        assert d_model % num_heads == 0

        self.grid_size = (self.H // patch_size, self.W // patch_size)
        self.patch_embed = PatchEmbedding(3, d_model, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size[0] * self.grid_size[1], d_model))
        self.pos_drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([
            TransformerEncoder(d_model, num_heads, d_model // num_heads, d_model // num_heads, 4 * d_model, dropout=0.1)
            for _ in range(num_layers)
        ])
        # Independent layernorm at each extracted block
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images):
        X, S1, S2 = self.patch_embed(images)
        assert X.shape[1:] == self.pos_embed.shape[1:]
        X = self.pos_drop(X + self.pos_embed)
        features = []
        for block, norm in zip(self.blocks, self.norms):
            X = block(X)
            features.append(norm(X))
        
        return features, S1, S2