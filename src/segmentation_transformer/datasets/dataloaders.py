import torch
from torch.utils.data import DataLoader, Subset
from segmentation_transformer.datasets.oxford_pets import OxfordPetsSegmentation

def create_dataloaders(
    data_root="data",
    image_size=(240, 320),
    batch_size=8,
    num_workers=1,
    seed=42
):
    train_dataset =OxfordPetsSegmentation(
        root=data_root,
        split="trainval",
        image_size=image_size,
        augment=True,
        download=True
    )

    # Same images, but WITHOUT training augmentation.
    val_dataset = OxfordPetsSegmentation(
        root=data_root,
        split="trainval",
        image_size=image_size,
        augment=False,
        download=False
    )

    test_dataset = OxfordPetsSegmentation(
        root=data_root,
        split="test",
        image_size=image_size,
        augment=False,
        download=True
    )

    # Train/Validation split
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    train_size = int(0.8* len(indices)) # 80% train
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


