# src/segformer/models/decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Reassemble(nn.Module):
    def __init__(self, d_model, out_channels, scale):
        super().__init__()
        assert out_channels % 8 == 0
        self.scale = scale
        # CHANGED (other AI §2): GroupNorm + GELU in projection and refinement.
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.GELU()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), nn.GELU()
        )

    def forward(self, x, hp, wp):
        b, n, d = x.shape
        assert n == hp * wp
        x = x.transpose(1, 2).reshape(b, d, hp, wp)
        x = self.proj(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return self.refine(x)

class FusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # CHANGED (other AI §3): add GroupNorm within residual conv branch.
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels), 
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels)
        )

    def forward(self, lateral, deeper):
        deeper = F.interpolate(deeper, size=lateral.shape[-2:],
                               mode='bilinear', align_corners=False)
        merged = lateral + deeper
        # CHANGED (our adjustment): keep the identity skip instead of GELU after addition.
        return merged + self.conv(merged)
    
class SegFormerDecoder(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.fuse3 = FusionBlock(channels)
        self.fuse2 = FusionBlock(channels)
        self.fuse1 = FusionBlock(channels)
        # Progressively fuse H/4 stem features, then H/2 stem features.
        self.up_half = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU()
        )
        self.refine_half = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.GELU()
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=1)
        )

    def forward(self, features, s1, s2, output_size):
        f1, f2, f3, f4 = features
        r3 = self.fuse3(f3, f4)
        r2 = self.fuse2(f2, r3)
        r1 = self.fuse1(f1, r2)  # B,64,H/4,W/4
        assert r1.shape == s2.shape, f'H/4 feature mismatch: {r1.shape} vs {s2.shape}'
        r1 = r1 + s2
        x = F.interpolate(r1, size=s1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.up_half(x)
        x = self.refine_half(torch.cat([x, s1], dim=1))
        x = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)
        return self.segmentation_head(x)