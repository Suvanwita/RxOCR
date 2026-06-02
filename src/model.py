from __future__ import annotations

import torch
from torch import nn


class CRNN(nn.Module):
    """Convolutional recurrent neural network for CTC-based HTR.

    Input shape:
        ``(batch, channels, height, width)``

    Output shape:
        ``(time, batch, num_classes)``, ready for ``nn.CTCLoss``.
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            self._conv_block(input_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            self._conv_block(128, 256),
            self._conv_block(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            self._conv_block(256, 512, use_batch_norm=True),
            self._conv_block(512, 512, use_batch_norm=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(512, 512, kernel_size=(4, 1), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.sequence_model = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=False,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn(images)
        if features.size(2) != 1:
            raise ValueError(
                "CRNN CNN backbone expects height 64 inputs so feature height becomes 1; "
                f"got feature height {features.size(2)}"
            )

        features = features.squeeze(2)
        sequence = features.permute(2, 0, 1)
        recurrent_output, _ = self.sequence_model(sequence)
        return self.classifier(recurrent_output)

    @staticmethod
    def _conv_block(
        in_channels: int, out_channels: int, use_batch_norm: bool = False
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        ]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)


def build_crnn(num_classes: int, **kwargs: object) -> CRNN:
    """Factory helper for constructing the default CRNN."""

    return CRNN(num_classes=num_classes, **kwargs)

