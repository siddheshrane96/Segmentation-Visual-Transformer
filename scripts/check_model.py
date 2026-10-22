import torch

from segmentation_transformer.models.segmentation_transformer import SegmentationTransformer


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SegmentationTransformer(
        image_size=(240, 320),
        patch_size=16,
        d_model=192,
        decoder_channels=64
    ).to(device)

    images = torch.randn(
        2, 3, 240, 320,
        device=device
    )

    model.eval()

    with torch.no_grad():
        logits = model(images)

    print("Input:", images.shape)
    print("Logits:", logits.shape)

    predicted_mask = logits.argmax(dim=1)

    print("Predicted mask:", predicted_mask.shape)

    assert logits.shape == (2, 3, 240, 320)
    assert predicted_mask.shape == (2, 240, 320)

    print("Model forward pass successful!")


if __name__ == "__main__":
    main()