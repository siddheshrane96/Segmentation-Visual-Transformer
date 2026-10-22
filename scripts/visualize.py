import torch
import numpy as np
import matplotlib.pyplot as plt

from segformer.datasets.dataloaders import create_dataloaders
from segformer.models.segformer import SegFormer


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    _, val_loader, _  = create_dataloaders(
        batch_size=10
    )

    model = SegFormer(
        image_size=(240, 320),
        patch_size=16,
        d_model=192,
        decoder_channels=64
    ).to(device)

    checkpoint = torch.load(
        "checkpoints/segformer_pets_v1/best.pt",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    batch = next(iter(val_loader))

    images = batch["image"].to(device)
    targets = batch["mask"]

    with torch.no_grad():
        logits = model(images)
        predictions = logits.argmax(dim=1).cpu()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(
        10, 4,
        figsize=(5, 10)
    )

    for i in range(10):

        image = images[i].cpu().permute(1, 2, 0).numpy()
        image = np.clip(image * std + mean, 0, 1)

        gt = targets[i].numpy()
        pred = predictions[i].numpy()

        errors = gt != pred

        axes[i, 0].imshow(image)

        axes[i, 1].imshow(
            gt,
            cmap="tab10",
            vmin=0,
            vmax=2
        )

        axes[i, 2].imshow(
            pred,
            cmap="tab10",
            vmin=0,
            vmax=2
        )

        axes[i, 3].imshow(
            errors,
            cmap="gray"
        )

        if i == 0:
            axes[i, 0].set_title("RGB")
            axes[i, 1].set_title("Ground Truth")
            axes[i, 2].set_title("Prediction")
            axes[i, 3].set_title("Errors")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    # plt.show()
    plt.savefig(
        "segmentation_results.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":
    main()