"""SE-RGCN: Spatially-Encoded Relational Graph Convolutional Network.

Skeleton — implementation deferred. The class shell is established here so the
training loop, configs, and unit tests can import a stable symbol while the
distance matrix and baselines are being built.
"""

from __future__ import annotations

import torch
from torch import nn


class SERGCN(nn.Module):
    """Spatially-Encoded Relational GCN over multiplex dyad-year graphs.

    Components (to be implemented):
      - Mandala spatial kernel  S(d) = cos(beta * d) * exp(-alpha * d)
      - Per-relation weight matrices  W_military, W_trade, W_spatial
      - Curriculum / annealing schedule for trade-relation contribution
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs) -> torch.Tensor:  # noqa: D401, ANN002, ANN003
        raise NotImplementedError("SERGCN.forward is not yet implemented.")
