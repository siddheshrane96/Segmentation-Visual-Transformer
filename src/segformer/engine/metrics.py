import torch


class SegmentationMetrics:

    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros(
            self.num_classes,
            self.num_classes,
            dtype=torch.int64
        )

    @torch.no_grad()
    def update(self, logits, targets):
        predictions = logits.argmax(dim=1)
        predictions = predictions.reshape(-1).cpu()
        targets = targets.reshape(-1).cpu()
        valid = (
            (targets >= 0) &
            (targets < self.num_classes)
        )
        predictions = predictions[valid]
        targets = targets[valid]

        indices = (targets * self.num_classes + predictions)
        counts = torch.bincount(indices, minlength=self.num_classes ** 2)

        self.confusion_matrix += counts.reshape(
            self.num_classes,
            self.num_classes
        )

    def compute(self):
        cm = self.confusion_matrix.float()
        tp = cm.diag()
        actual = cm.sum(dim=1)
        predicted = cm.sum(dim=0)
        union = actual + predicted - tp
        iou = tp / union.clamp_min(1)
        pixel_accuracy = tp.sum() / cm.sum().clamp_min(1)

        return {
            "pixel_accuracy": pixel_accuracy.item(),
            "mean_iou": iou.mean().item(),
            "per_class_iou": iou.tolist()
        }