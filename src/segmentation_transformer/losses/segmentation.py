import torch
import torch.nn as nn

class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        assert logits.ndim == 4
        assert targets.ndim == 3
        assert logits.shape[0] == targets.shape[0]
        assert logits.shape[-2:] == targets.shape[-2:]

        return self.criterion(logits, targets.long())