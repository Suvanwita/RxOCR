"""Compatibility import for the RxOCR IAM data loader."""

from src.data_loader import IAMDataset, iam_collate_fn

__all__ = ["IAMDataset", "iam_collate_fn"]
