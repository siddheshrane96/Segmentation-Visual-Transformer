import torch
from tqdm.auto import tqdm

from segformer.engine.metrics import SegmentationMetrics

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None
):
    training = optimizer is not None
    model.train(training)
    metrics = SegmentationMetrics(num_classes=3)

    total_loss = 0.0
    total_images = 0
    progress = tqdm(
        loader,
        desc="Training" if training else "Validation",
        leave=False
    )

    for batch in progress:
        images = batch["image"].to(
            device, non_blocking=True
        )
        targets = batch["mask"].to(
            device, non_blocking=True
        )
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                    error_if_nonfinite=True
                )
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size

        metrics.update(
            logits.detach(),
            targets
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    results = metrics.compute()
    results["loss"] = total_loss / total_images

    return results