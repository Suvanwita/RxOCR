# RxOCR

PyTorch project scaffold for handwritten text recognition.

## IAM Data Layout

Place the IAM Handwriting Database under `data/iam`:

```text
data/iam/
  ascii/lines.txt
  lines/a01/a01-000u/a01-000u-00.png
```

## Dataset Usage

```python
from torch.utils.data import DataLoader
from src.data_loader import IAMDataset, iam_collate_fn
from src.model import CRNN

dataset = IAMDataset(root_dir="data/iam", level="lines", image_height=64)
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=iam_collate_fn)
model = CRNN(num_classes=dataset.num_classes)

batch = next(iter(loader))
images = batch["images"]
labels = batch["labels"]
label_lengths = batch["label_lengths"]
logits = model(images)
```

## Training

```bash
python3 train.py --data-root data/iam --epochs 20 --batch-size 16
```

If `data/iam/Train_Label.csv`, `data/iam/Test_Label.csv`, `data/iam/Train_Set`,
and `data/iam/Test_Set` exist, the script uses them automatically.

Use `--train-split` and `--val-split` if you want separate IAM split files.

## Inference

```bash
python3 inference.py path/to/image.jpg --checkpoint checkpoints/best_crnn.pt
```

Character id `0` is reserved for the CTC blank token. Dataset labels contain
only real character ids, starting at `1`.
