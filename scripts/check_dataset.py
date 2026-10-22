import torch
from segmentation_transformer.datasets.dataloaders import create_dataloaders
import numpy as np
import matplotlib.pyplot as plt

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_loader, val_loader, test_loader = create_dataloaders(batch_size=8)

print("Training images:", len(train_loader.dataset))
print("Validation images:", len(val_loader.dataset))
print("Testing images:", len(test_loader.dataset))

batch = next(iter(train_loader))
images = batch["image"]
masks = batch["mask"]

print("\nImage shape:", images.shape)
print("Mask shape:", masks.shape)
print("Image dtype:", images.dtype)
print("Mask dtype:", masks.dtype)
print("Mask labels:", torch.unique(masks))

image = images[0].permute(1,2,0).numpy()
image = np.clip((image*std) + mean, 0, 1)
mask = masks[0].numpy()


fig, axes = plt.subplots(1, 2, figsize=(15, 4))
axes[0].imshow(image)
axes[1].imshow(mask)
plt.show()