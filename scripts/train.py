from pathlib import Path

import torch

from segformer.datasets.dataloaders import create_dataloaders
from segformer.models.segformer import SegFormer
from segformer.losses.segmentation import SegmentationLoss
from segformer.engine.trainer import run_epoch


def main():

    # ---------------------------------
    # Configuration
    # ---------------------------------

    IMAGE_SIZE = (240, 320)
    BATCH_SIZE = 4

    EPOCHS = 30  # Additional epochs to train
    LEARNING_RATE = 1e-5

    CHECKPOINT_DIR = Path("checkpoints/segformer_pets_v1")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"
    LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # ---------------------------------
    # Dataset
    # ---------------------------------

    train_loader, val_loader, test_loader = create_dataloaders(
        data_root="data",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # ---------------------------------
    # Model
    # ---------------------------------

    model = SegFormer(
        image_size=IMAGE_SIZE,
        patch_size=16,
        d_model=192,
        decoder_channels=64
    ).to(device)

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {num_parameters:,}")

    # ---------------------------------
    # Loss
    # ---------------------------------

    criterion = SegmentationLoss()

    # ---------------------------------
    # Optimizer
    # ---------------------------------

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if param.ndim < 2 or "pos_embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0}
        ],
        lr=LEARNING_RATE
    )

    # ---------------------------------
    # Resume from best checkpoint
    # ---------------------------------

    start_epoch = 0
    best_val_iou = -1.0

    if BEST_CHECKPOINT.exists():

        print(f"\nLoading checkpoint: {BEST_CHECKPOINT}")

        checkpoint = torch.load(
            BEST_CHECKPOINT,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # optimizer.load_state_dict(
        #     checkpoint["optimizer_state_dict"]
        # )

        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # # Manually override the learning rate for all parameter groups
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = LEARNING_RATE


        start_epoch = checkpoint["epoch"]
        best_val_iou = checkpoint["best_val_iou"]

        print(f"Resuming from epoch {start_epoch}")
        print(f"Previous best validation mIoU: {best_val_iou:.4f}")

    else:

        print("\nNo existing checkpoint found.")
        print("Training from scratch.")

    # ---------------------------------
    # Training
    # ---------------------------------

    end_epoch = start_epoch + EPOCHS

    print(
        f"\nTraining epochs {start_epoch + 1} "
        f"to {end_epoch}\n"
    )

    for epoch in range(start_epoch, end_epoch):

        train_results = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer
        )

        val_results = run_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch + 1:02d}/{end_epoch} | "
            f"Train Loss: {train_results['loss']:.4f} | "
            f"Train mIoU: {train_results['mean_iou']:.4f} | "
            f"Val Loss: {val_results['loss']:.4f} | "
            f"Val mIoU: {val_results['mean_iou']:.4f}"
        )

        print(
            "Validation class IoU:",
            [round(v, 4) for v in val_results["per_class_iou"]]
        )

        # ---------------------------------
        # Save checkpoint
        # ---------------------------------

        current_val_iou = val_results["mean_iou"]

        is_best = current_val_iou > best_val_iou

        if is_best:
            best_val_iou = current_val_iou

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_iou": best_val_iou
        }

        # Always save latest
        torch.save(
            checkpoint,
            LATEST_CHECKPOINT
        )

        # Save best only if improved
        if is_best:

            torch.save(
                checkpoint,
                BEST_CHECKPOINT
            )

            print("New best checkpoint saved!")

    print(f"\nBest validation mIoU: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()