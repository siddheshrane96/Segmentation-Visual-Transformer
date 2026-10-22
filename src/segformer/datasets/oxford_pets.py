import random
import torch
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np

class OxfordPetsSegmentation(Dataset):
    def __init__(self, root="data", split="trainval", image_size=(240, 320), augment=False, download=False):
        super().__init__()
        self.dataset = OxfordIIITPet(root=root, split=split, target_types="segmentation", download=download)
        self.image_size = image_size
        self.augment = augment

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, mask = self.dataset[index]
        # Apply identical spatial transformations
        if self.augment and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.resize(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.image_size, interpolation=InterpolationMode.NEAREST)
        
        # Convert RGB to normalized tensor.
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)
        raw_mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        # Oxford labels: 1, 2, 3
        # Training labels: 0, 1, 2 where 0 is pet, 1 background and 2 is boundary
        raw_mask = torch.from_numpy(
            np.array(mask, dtype=np.int64).copy()
        ) - 1
        assert raw_mask.min() >= 0
        assert raw_mask.max() <= 2
        return {
            "image": image,
            "mask": raw_mask
        }