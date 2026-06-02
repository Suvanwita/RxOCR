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

Character id `0` is reserved for the CTC blank token. Dataset labels contain
only real character ids, starting at `1`.
