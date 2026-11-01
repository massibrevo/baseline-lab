"""Original inference implementation of the published F3ED LCL/MLC architecture.

Liu et al., F3Set, ICLR 2025. RegNet is supplied by timm (Apache-2.0).
The temporal shift operation follows MIT HAN Lab's MIT-licensed TSM algorithm.
No upstream training scripts or remotely executed checkpoint code are required.
The contextual refinement head needs verified handedness and is not used here.
"""

from typing import Any

import torch
from torch import nn


class TemporalShift(nn.Module):
    def __init__(self, convolution):
        super().__init__()
        self.net = convolution

    def forward(self, images):
        n, channels, height, width = images.shape
        sequence = images.reshape(-1, 96, channels, height, width)
        fold = channels // 8
        shifted = torch.zeros_like(sequence)
        shifted[:, :-2, :fold] = sequence[:, 2:, :fold]
        shifted[:, 2:, fold : 2 * fold] = sequence[:, :-2, fold : 2 * fold]
        shifted[:, :, 2 * fold :] = sequence[:, :, 2 * fold :]
        return self.net(shifted.reshape(n, channels, height, width))


class Projection(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self._fc_out = nn.Linear(inputs, outputs)

    def forward(self, values):
        return self._fc_out(values)


class RecurrentHead(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self._gru = nn.GRU(inputs, 368, batch_first=True, bidirectional=True)
        self._fc_out = Projection(736, outputs)

    def forward(self, values):
        encoded, _ = self._gru(values)
        return self._fc_out(encoded)


class StrokeNetwork(nn.Module):
    def __init__(self):
        import timm

        super().__init__()
        features: Any = timm.create_model("regnety_002", pretrained=False)
        features.head.fc = nn.Identity()
        for stage in [features.s1, features.s2, features.s3, features.s4]:
            for block in stage.children():
                block.conv1 = TemporalShift(block.conv1)
        self._glb_feat = features
        self._head = RecurrentHead(368, 368)
        self._coarse_pred = nn.Linear(368, 2)
        self._fine_pred = nn.Linear(368, 29)
        self._ctx = RecurrentHead(30, 30)

    def forward(self, images):
        batch, steps, channels, height, width = images.shape
        spatial = self._glb_feat(images.reshape(-1, channels, height, width))
        encoded = self._head(spatial.reshape(batch, steps, 368))
        return self._coarse_pred(encoded), self._fine_pred(encoded)
