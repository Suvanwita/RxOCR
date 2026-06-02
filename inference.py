from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.model import CRNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RxOCR inference on one image.")
    parser.add_argument("image_path", type=Path, help="Path to the handwriting image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_crnn.pt"),
        help="Path to a trained CRNN checkpoint.",
    )
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--mean", type=float, default=0.5)
    parser.add_argument("--std", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if "model_state_dict" not in checkpoint:
        return {"model_state_dict": checkpoint}
    return checkpoint


def resize_image(image: Image.Image, image_height: int) -> Image.Image:
    width, height = image.size
    new_width = max(1, round(width * (image_height / height)))
    return image.resize((new_width, image_height), Image.BILINEAR)


def preprocess_pil_image(
    image: Image.Image, image_height: int, mean: float, std: float
) -> tuple[torch.Tensor, int, Image.Image]:
    image = image.convert("L")
    image = resize_image(image, image_height)
    image_tensor = torch.from_numpy(np.array(image, dtype="float32")).unsqueeze(0)
    image_tensor = image_tensor / 255.0
    image_tensor = (image_tensor - mean) / std
    return image_tensor.unsqueeze(0), image_tensor.shape[-1], image


def preprocess_image(
    image_path: Path, image_height: int, mean: float, std: float
) -> tuple[torch.Tensor, int]:
    image = Image.open(image_path)
    image_tensor, image_width, _ = preprocess_pil_image(image, image_height, mean, std)
    return image_tensor, image_width


def greedy_ctc_decode(
    log_probs: torch.Tensor, idx_to_char: list[str], input_length: int | None = None
) -> str:
    """Decode CTC probabilities with argmax, repeat collapse, and blank removal."""

    best_path = log_probs.argmax(dim=2).squeeze(1)
    if input_length is not None:
        best_path = best_path[:input_length]

    decoded_chars: list[str] = []
    previous_idx: int | None = None
    for idx_tensor in best_path:
        idx = int(idx_tensor)
        if idx != 0 and idx != previous_idx:
            decoded_chars.append(idx_to_char[idx])
        previous_idx = idx
    return "".join(decoded_chars)


@torch.no_grad()
def predict(
    model: CRNN,
    image: torch.Tensor,
    image_width: int,
    idx_to_char: list[str],
    device: torch.device,
) -> str:
    model.eval()
    image = image.to(device)
    logits = model(image)
    log_probs = F.log_softmax(logits, dim=2).cpu()
    input_length = min(log_probs.size(0), max(1, image_width // 4))
    return greedy_ctc_decode(log_probs, idx_to_char, input_length=input_length)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)

    idx_to_char = checkpoint.get("idx_to_char")
    if idx_to_char is None:
        raise ValueError("Checkpoint must contain idx_to_char to decode predictions")

    saved_args = checkpoint.get("args", {})
    image_height = args.image_height or saved_args.get("image_height", 64)
    hidden_size = args.hidden_size or saved_args.get("hidden_size", 256)

    model = CRNN(num_classes=len(idx_to_char), hidden_size=hidden_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    image, image_width = preprocess_image(
        image_path=args.image_path,
        image_height=image_height,
        mean=args.mean,
        std=args.std,
    )
    prediction = predict(model, image, image_width, idx_to_char, device)
    print(prediction)


if __name__ == "__main__":
    main()
