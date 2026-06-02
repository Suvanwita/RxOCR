from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.data_loader import IAMDataset, iam_collate_fn
from src.model import CRNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RxOCR CRNN with CTC loss.")
    parser.add_argument("--data-root", type=Path, default=Path("data/iam"))
    parser.add_argument("--train-annotation", type=Path, default=None)
    parser.add_argument("--val-annotation", type=Path, default=None)
    parser.add_argument("--train-image-dir", type=Path, default=None)
    parser.add_argument("--val-image-dir", type=Path, default=None)
    parser.add_argument("--train-split", type=Path, default=None)
    parser.add_argument("--val-split", type=Path, default=None)
    parser.add_argument("--level", choices=["lines", "words"], default="lines")
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def crnn_input_lengths(image_widths: torch.Tensor) -> torch.Tensor:
    """Compute output time steps after the CRNN CNN width downsampling."""

    return torch.clamp(image_widths // 4, min=1).to(dtype=torch.long)


def greedy_decode(
    logits: torch.Tensor, input_lengths: torch.Tensor, dataset: IAMDataset
) -> list[str]:
    """Decode CTC logits using argmax, blank removal, and repeat collapsing."""

    predictions = logits.argmax(dim=2).permute(1, 0)
    decoded = []
    for sequence, length in zip(predictions, input_lengths):
        decoded.append(
            dataset.sequence_to_text(
                sequence[: int(length)].tolist(), collapse_repeats=True
            )
        )
    return decoded


def edit_distance(source: str, target: str) -> int:
    previous = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current = [i]
        for j, target_char in enumerate(target, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + int(source_char != target_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def character_error_rate(predictions: list[str], references: list[str]) -> float:
    total_edits = 0
    total_chars = 0
    for prediction, reference in zip(predictions, references):
        total_edits += edit_distance(prediction, reference)
        total_chars += len(reference)
    return total_edits / max(1, total_chars)


def create_datasets(args: argparse.Namespace) -> tuple[IAMDataset, IAMDataset]:
    train_annotation = args.train_annotation
    val_annotation = args.val_annotation
    train_image_dir = args.train_image_dir
    val_image_dir = args.val_image_dir

    if train_annotation is None and (args.data_root / "Train_Label.csv").exists():
        train_annotation = args.data_root / "Train_Label.csv"
        train_image_dir = train_image_dir or args.data_root / "Train_Set"
    if val_annotation is None and (args.data_root / "Test_Label.csv").exists():
        val_annotation = args.data_root / "Test_Label.csv"
        val_image_dir = val_image_dir or args.data_root / "Test_Set"

    train_dataset = IAMDataset(
        root_dir=args.data_root,
        annotation_file=train_annotation,
        split_file=args.train_split,
        level=args.level,
        image_dir=train_image_dir,
        image_height=args.image_height,
    )
    val_dataset = IAMDataset(
        root_dir=args.data_root,
        annotation_file=val_annotation,
        split_file=args.val_split,
        level=args.level,
        image_dir=val_image_dir,
        image_height=args.image_height,
        charset=train_dataset.idx_to_char[1:],
    )
    return train_dataset, val_dataset


def train_one_epoch(
    model: CRNN,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip_grad: float,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"].to(device)
        input_lengths = crnn_input_lengths(batch["image_widths"]).to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        log_probs = F.log_softmax(logits, dim=2)
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate(
    model: CRNN,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    dataset: IAMDataset,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_predictions: list[str] = []
    all_references: list[str] = []

    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"].to(device)
        input_lengths = crnn_input_lengths(batch["image_widths"])
        device_input_lengths = input_lengths.to(device)

        logits = model(images)
        log_probs = F.log_softmax(logits, dim=2)
        loss = criterion(log_probs, labels, device_input_lengths, label_lengths)

        total_loss += loss.item()
        all_predictions.extend(greedy_decode(logits.cpu(), input_lengths, dataset))
        all_references.extend(batch["texts"])

    cer = character_error_rate(all_predictions, all_references)
    return total_loss / max(1, len(loader)), cer


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = create_datasets(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=iam_collate_fn,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=iam_collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = CRNN(
        num_classes=train_dataset.num_classes,
        hidden_size=args.hidden_size,
    ).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_cer = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            clip_grad=args.clip_grad,
        )
        val_loss, val_cer = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            dataset=val_dataset,
            device=device,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_CER={val_cer:.4f}"
        )

        if val_cer < best_cer:
            best_cer = val_cer
            checkpoint_path = args.checkpoint_dir / "best_crnn.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "num_classes": train_dataset.num_classes,
                    "idx_to_char": train_dataset.idx_to_char,
                    "best_cer": best_cer,
                    "args": vars(args),
                },
                checkpoint_path,
            )


if __name__ == "__main__":
    main()
