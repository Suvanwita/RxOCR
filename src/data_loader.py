from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


ImageTransform = Callable[[Image.Image], torch.Tensor]


@dataclass(frozen=True)
class IAMSample:
    """Single IAM handwriting sample."""

    sample_id: str
    image_path: Path
    text: str


class IAMDataset(Dataset):
    """Dataset for IAM handwriting line or word images.

    Expected IAM layout:

    data/iam/
      ascii/lines.txt
      lines/a01/a01-000u/a01-000u-00.png

    Word-level loading is also supported by setting ``level="words"`` and
    using the IAM ``ascii/words.txt`` file plus the ``words/`` image folder.
    Character id ``0`` is reserved for the CTC blank token, so encoded labels
    start at ``1``.
    """

    blank_token = "<BLANK>"

    def __init__(
        self,
        root_dir: str | Path,
        annotation_file: str | Path | None = None,
        split_file: str | Path | None = None,
        level: str = "lines",
        image_height: int = 64,
        image_width: int | None = None,
        mean: float = 0.5,
        std: float = 0.5,
        charset: Sequence[str] | None = None,
        transform: ImageTransform | None = None,
        skip_bad_samples: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.level = level
        self.image_height = image_height
        self.image_width = image_width
        self.mean = mean
        self.std = std
        self.transform = transform

        if level not in {"lines", "words"}:
            raise ValueError('level must be either "lines" or "words"')

        self.annotation_file = (
            Path(annotation_file)
            if annotation_file is not None
            else self.root_dir / "ascii" / f"{level}.txt"
        )
        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"IAM annotation file not found: {self.annotation_file}"
            )

        split_ids = self._read_split_ids(split_file)
        self.samples = self._parse_annotations(split_ids, skip_bad_samples)
        if not self.samples:
            raise ValueError("No IAM samples were found for the requested configuration")

        characters = sorted(set(charset or self._collect_charset(self.samples)))
        self.idx_to_char = [self.blank_token, *characters]
        self.char_to_idx = {char: idx for idx, char in enumerate(self.idx_to_char)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = self._load_image(sample.image_path)
        label = torch.tensor(self.text_to_sequence(sample.text), dtype=torch.long)

        return {
            "image": image,
            "label": label,
            "label_length": torch.tensor(label.numel(), dtype=torch.long),
            "text": sample.text,
            "sample_id": sample.sample_id,
        }

    @property
    def num_classes(self) -> int:
        """Number of output classes, including the CTC blank class."""

        return len(self.idx_to_char)

    def text_to_sequence(self, text: str) -> list[int]:
        """Encode a text label into character ids for CTC loss."""

        try:
            return [self.char_to_idx[char] for char in text]
        except KeyError as exc:
            missing = exc.args[0]
            raise ValueError(f"Character {missing!r} is not in the dataset charset") from exc

    def sequence_to_text(self, sequence: Iterable[int], collapse_repeats: bool = False) -> str:
        """Decode character ids, optionally applying CTC repeat collapsing."""

        chars: list[str] = []
        previous_idx: int | None = None
        for idx in sequence:
            idx = int(idx)
            if idx == 0:
                previous_idx = idx
                continue
            if collapse_repeats and idx == previous_idx:
                continue
            chars.append(self.idx_to_char[idx])
            previous_idx = idx
        return "".join(chars)

    def _parse_annotations(
        self, split_ids: set[str] | None, skip_bad_samples: bool
    ) -> list[IAMSample]:
        samples: list[IAMSample] = []
        with self.annotation_file.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                sample = self._parse_annotation_line(line, skip_bad_samples)
                if sample is None:
                    continue
                if split_ids is not None and sample.sample_id not in split_ids:
                    continue
                if not sample.image_path.exists():
                    if skip_bad_samples:
                        continue
                    raise FileNotFoundError(f"IAM image not found: {sample.image_path}")
                samples.append(sample)
        return samples

    def _parse_annotation_line(
        self, line: str, skip_bad_samples: bool
    ) -> IAMSample | None:
        parts = line.split(maxsplit=8)
        if len(parts) < 9:
            raise ValueError(f"Malformed IAM annotation line: {line}")

        sample_id = parts[0]
        status = parts[1]
        if status != "ok" and skip_bad_samples:
            return None

        if self.level == "lines":
            text = parts[8].replace("|", " ")
        else:
            text = parts[8]

        return IAMSample(
            sample_id=sample_id,
            image_path=self._image_path_for_id(sample_id),
            text=text,
        )

    def _image_path_for_id(self, sample_id: str) -> Path:
        if self.level == "lines":
            form_id = "-".join(sample_id.split("-")[:2])
            folder_id = sample_id.split("-")[0]
            return self.root_dir / "lines" / folder_id / form_id / f"{sample_id}.png"

        folder_id = sample_id.split("-")[0]
        form_id = "-".join(sample_id.split("-")[:2])
        return self.root_dir / "words" / folder_id / form_id / f"{sample_id}.png"

    def _load_image(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("L")

        if self.transform is not None:
            tensor = self.transform(image)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            return tensor.float()

        image = self._resize_image(image)
        image_tensor = torch.from_numpy(np.array(image, dtype="float32")).unsqueeze(0)
        image_tensor = image_tensor / 255.0
        return (image_tensor - self.mean) / self.std

    def _resize_image(self, image: Image.Image) -> Image.Image:
        if self.image_width is not None:
            return image.resize((self.image_width, self.image_height), Image.BILINEAR)

        width, height = image.size
        new_width = max(1, round(width * (self.image_height / height)))
        return image.resize((new_width, self.image_height), Image.BILINEAR)

    @staticmethod
    def _read_split_ids(split_file: str | Path | None) -> set[str] | None:
        if split_file is None:
            return None

        path = Path(split_file)
        with path.open("r", encoding="utf-8") as file:
            return {
                line.strip()
                for line in file
                if line.strip() and not line.startswith("#")
            }

    @staticmethod
    def _collect_charset(samples: Sequence[IAMSample]) -> set[str]:
        return {char for sample in samples for char in sample.text}


def iam_collate_fn(batch: Sequence[dict[str, torch.Tensor | str]]) -> dict[str, object]:
    """Pad IAM samples into a CTC-friendly mini-batch."""

    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]
    label_lengths = torch.stack([item["label_length"] for item in batch])
    texts = [item["text"] for item in batch]
    sample_ids = [item["sample_id"] for item in batch]

    max_width = max(image.shape[-1] for image in images)
    padded_images = []
    image_widths = []
    for image in images:
        width = image.shape[-1]
        image_widths.append(width)
        padded_images.append(F.pad(image, (0, max_width - width, 0, 0), value=0.0))

    return {
        "images": torch.stack(padded_images),
        "labels": torch.cat(labels),
        "label_lengths": label_lengths,
        "image_widths": torch.tensor(image_widths, dtype=torch.long),
        "texts": texts,
        "sample_ids": sample_ids,
    }
