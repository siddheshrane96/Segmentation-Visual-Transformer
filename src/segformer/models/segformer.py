# src/segformer/models/segformer.py
import torch.nn as nn

from .encoder import SegFormerEncoder
from .decoder import Reassemble, SegFormerDecoder

class SegFormer(nn.Module):
    def __init__(self, image_size=(240,320), patch_size=16, d_model=192, decoder_channels=64):
        super().__init__()
        self.patch_size = patch_size
        self.encoder = SegFormerEncoder(image_size, patch_size, d_model, 4, 4)
        self.reassemble = nn.ModuleList([
            Reassemble(d_model, decoder_channels, 4.0),
            Reassemble(d_model, decoder_channels, 2.0),
            Reassemble(d_model, decoder_channels, 1.0),
            Reassemble(d_model, decoder_channels, 0.5),
        ])
        self.decoder = SegFormerDecoder(decoder_channels)

    def forward(self, images):
        _, _, h, w = images.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0
        hp, wp = h // self.patch_size, w // self.patch_size
        features, s1, s2 = self.encoder(images)
        spatial = [module(x, hp, wp) for module, x in zip(self.reassemble, features)]
        return self.decoder(spatial, s1, s2, output_size=(h, w))